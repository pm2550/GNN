#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf

# 路径与参数
BASE = Path('/workplace/models/public/inceptionv3_8seg_uniform')
OUT = BASE / 'full_split_pipeline'
TFL_DIR = OUT / 'tflite'
TPU_DIR = OUT / 'tpu'
LOG = OUT / 'compilation_log.txt'
COMPILER = '/workplace/edgetpu_compiler'

LOW, HIGH = 2.0, 6.0
SEGMENTS = 8
TIMEOUT_S = 360


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    TFL_DIR.mkdir(parents=True, exist_ok=True)
    TPU_DIR.mkdir(parents=True, exist_ok=True)


def log_write(s: str):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(s)


def to_int8(model_):
    conv = tf.lite.TFLiteConverter.from_keras_model(model_)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    ishape = model_.input.shape[1:]

    def rep():
        for _ in range(64):
            yield [np.random.rand(1, *ishape).astype(np.float32)]

    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv._experimental_disable_per_channel = True
    conv._experimental_new_quantizer = False
    return conv.convert()


def compile_tpu(tfl_path: Path, tag: str):
    res = subprocess.run([COMPILER, '-o', str(TPU_DIR), str(tfl_path)],
                         capture_output=True, text=True, timeout=TIMEOUT_S)
    edgetpu_path = TPU_DIR / (tfl_path.stem + '_edgetpu.tflite')
    mb = edgetpu_path.stat().st_size / 1024 / 1024 if edgetpu_path.exists() else None
    log_write(f'[{tag}] compile {tfl_path.name}\n')
    if res.stdout:
        log_write(f'STDOUT:\n{res.stdout}\n')
    if res.stderr:
        log_write(f'STDERR:\n{res.stderr}\n')
    log_write('-' * 60 + '\n')
    return mb, edgetpu_path.exists()


def probe_size(m_in, t_out, name_tag):
    try:
        sub = tf.keras.Model(m_in, t_out)
        tfl = to_int8(sub)
        p = TFL_DIR / f'_{name_tag}.tflite'
        p.write_bytes(tfl)
        mb, ok = compile_tpu(p, f'probe:{name_tag}')
        return mb if ok else None
    except Exception:
        return None


def finalize_segment(m_in, t_out, seg_idx):
    sub = tf.keras.Model(m_in, t_out, name=f'seg{seg_idx}')
    tfl = to_int8(sub)
    p = TFL_DIR / f'seg{seg_idx}_int8.tflite'
    p.write_bytes(tfl)
    mb, ok = compile_tpu(p, f'finalize:seg{seg_idx}')
    return mb, ok


def pick_candidates(model):
    # 仅无后缀的 mixed* 且 4D 输出（模块/阶段出口，分支已融合）
    cands = []
    for l in model.layers:
        if l.name.startswith('mixed') and '_' not in l.name:
            out = getattr(l, 'output', None)
            if out is not None and len(getattr(out, 'shape', ())) == 4:
                cands.append(l)
    return cands


def greedy_with_backtrack(model):
    ensure_dirs()
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write(f'PIPELINE greedy split time={datetime.now()}\n')
        f.write('=' * 70 + '\n\n')

    # 整模大小
    full_tfl = to_int8(model)
    full_tfl_path = TFL_DIR / 'full_int8.tflite'
    full_tfl_path.write_bytes(full_tfl)
    full_mb, ok = compile_tpu(full_tfl_path, 'full_model')
    if not ok or full_mb is None:
        print('整模编译失败')
        sys.exit(1)
    print(f'FULL TPU: {full_mb:.2f}MB')
    log_write(f'FULL TPU SIZE: {full_mb:.2f}MB\n' + '-' * 60 + '\n')

    cands = pick_candidates(model)
    assert len(cands) >= SEGMENTS - 1, '候选数量不足以切成8段'

    tensors = [model.input] + [l.output for l in cands] + [model.output]

    def probe_from(prev_idx, seg_idx, target):
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        results = []
        for i in range(prev_idx + 1, len(cands)):
            name = cands[i].name
            mb = probe_size(m_in, tensors[i + 1], f'seg{seg_idx}_{name}')
            if mb is not None:
                results.append((i, name, mb))
        return results

    def try_shift_previous_to_absorb(prev_seg_k, decisions):
        if prev_seg_k < 0:
            return False
        prev_prev_idx = decisions[prev_seg_k - 1][0] if prev_seg_k - 1 >= 0 else -1
        prev_idx = decisions[prev_seg_k][0]
        m_in = tensors[0] if prev_prev_idx < 0 else tensors[prev_prev_idx + 1]
        for j in range(prev_idx + 1, len(cands)):
            mb_j = probe_size(m_in, tensors[j + 1], f'shift_prev_seg{prev_seg_k + 1}_{cands[j].name}')
            if mb_j is None:
                continue
            if mb_j <= HIGH:
                decisions[prev_seg_k] = (j, mb_j)
                print(f'  回退：seg{prev_seg_k + 1} 前移到 {cands[j].name} = {mb_j:.2f}MB')
                log_write(f'BACKTRACK: seg{prev_seg_k + 1} -> {cands[j].name} = {mb_j:.2f}MB\n')
                return True
        return False

    decisions = []  # [(cand_idx, mb)]
    prev_idx = -1
    seg_idx = 1
    remaining_mb = full_mb

    while seg_idx <= SEGMENTS:
        segments_left = SEGMENTS - len(decisions)
        # 尾段：直接到输出，若超界触发回退机制2
        if segments_left == 1:
            m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            mb, ok2 = finalize_segment(m_in, tensors[-1], seg_idx)
            if not ok2:
                print('尾段编译失败')
                sys.exit(2)
            if not (LOW <= mb <= HIGH):
                print(f'尾段 {mb:.2f}MB 不合规，启动回退机制2')
                ret_ok = False
                k = len(decisions) - 1
                while k >= 0 and not ret_ok:
                    moved = try_shift_previous_to_absorb(k, decisions)
                    if moved:
                        prev_idx = decisions[-1][0]
                        m_in = tensors[prev_idx + 1]
                        mb, ok2 = finalize_segment(m_in, tensors[-1], seg_idx)
                        if ok2 and (LOW <= mb <= HIGH):
                            ret_ok = True
                            break
                    else:
                        k -= 1
                if not ret_ok:
                    print('回退未能使尾段合规')
                    sys.exit(3)
            decisions.append(('OUTPUT', mb))
            print(f'seg{seg_idx}: OUTPUT = {mb:.2f}MB ✓')
            log_write(f'Finalize seg{seg_idx}: OUTPUT = {mb:.2f}MB\n' + '-' * 60 + '\n')
            break

        # 动态目标=剩余TPU大小/剩余段数
        target = remaining_mb / segments_left
        probes = probe_from(prev_idx, seg_idx, target)
        if not probes:
            print('无可探测候选')
            sys.exit(4)

        # 选择最接近目标的候选，并按回退/前进找第一个合规（2-6MiB）
        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
        ideal_i = probes.index(probes_sorted[0])

        pick = None
        for p in range(ideal_i, -1, -1):
            i, name, mb = probes[p]
            if LOW <= mb <= HIGH:
                pick = (i, name, mb)
                break
        if pick is None:
            for p in range(ideal_i + 1, len(probes)):
                i, name, mb = probes[p]
                if LOW <= mb <= HIGH:
                    pick = (i, name, mb)
                    break

        # 若当前段最小候选仍 > HIGH，触发回退机制2：回退上一刀向后移，缩小当前跨度
        min_mb = min(x[2] for x in probes)
        if pick is None and min_mb > HIGH:
            print(f'第{seg_idx}段 min={min_mb:.2f}MB > 6MiB，启动回退机制2')
            ret_ok = False
            k = len(decisions) - 1
            while k >= 0 and not ret_ok:
                moved = try_shift_previous_to_absorb(k, decisions)
                if moved:
                    prev_idx = decisions[-1][0]
                    probes = probe_from(prev_idx, seg_idx, target)
                    if probes:
                        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
                        ideal_i = 0
                        pick = None
                        for p in range(ideal_i, -1, -1):
                            i, name, mb = probes[p]
                            if LOW <= mb <= HIGH:
                                pick = (i, name, mb)
                                break
                        if pick is None:
                            for p in range(ideal_i + 1, len(probes)):
                                i, name, mb = probes[p]
                                if LOW <= mb <= HIGH:
                                    pick = (i, name, mb)
                                    break
                        if pick is not None:
                            ret_ok = True
                            break
                else:
                    k -= 1
            if not ret_ok:
                print('回退机制2失败，无法吃下溢出')
                sys.exit(5)

        if pick is None:
            print(f'第{seg_idx}段无合规候选')
            sys.exit(6)

        # 固化该段
        i, name, mb = pick
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        mb_fix, ok2 = finalize_segment(m_in, tensors[i + 1], seg_idx)
        if not ok2:
            print('当前段固化失败')
            sys.exit(7)
        if not (LOW <= mb_fix <= HIGH):
            print('固化后大小异常')
            sys.exit(8)

        prev_idx = i
        remaining_mb -= mb_fix
        print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓')
        log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB\n' + '-' * 60 + '\n')
        seg_idx += 1

    # 汇总
    tpu_files = sorted(TPU_DIR.glob('seg*_int8_edgetpu.tflite'))
    ok_cnt = 0
    for f in tpu_files:
        mb = f.stat().st_size / 1024 / 1024
        ok = LOW <= mb <= HIGH
        ok_cnt += int(ok)
        print(f'{f.name}: {mb:.2f}MB {"OK" if ok else "NG"}')
    print(f'合规: {ok_cnt}/{len(tpu_files)}')


def main():
    print('=== 按 PIPELINE 规则切 InceptionV3 为 8 段 (2–6MiB) ===')
    m = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)
    greedy_with_backtrack(m)


if __name__ == '__main__':
    main()


