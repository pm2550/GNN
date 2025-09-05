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

    # 统一的当前段逆向回退并即刻固化
    def backtrack_current_and_finalize(cur_prev_idx, cur_end_idx, cur_seg_idx, segments_left, remaining_mb_local):
        m_in_local = tensors[0] if cur_prev_idx < 0 else tensors[cur_prev_idx + 1]
        for j in range(cur_end_idx - 1, cur_prev_idx, -1):
            name_j = cands[j].name
            mb_j = probe_size(m_in_local, tensors[j + 1], f'bt_seg{cur_seg_idx}_{name_j}')
            if mb_j is None:
                continue
            if LOW <= mb_j <= HIGH:
                # 可用性护栏：回退命中窗内后，需确保后续还能形成 n-1 合规段（末段连OUTPUT）
                rem_after = remaining_mb_local - mb_j
                seg_left_after = segments_left - 1
                # 检查下一个段是否仍有>LOW窗内候选，且末段可连OUTPUT>LOW
                def _availability_ok_after_pick_simple(i_pick, mb_pick, seg_idx_now, rem_after_pick, seg_left_now):
                    if seg_left_now <= 1:
                        return True
                    # 估一估下一段
                    prev_idx_next = i_pick
                    m_in_next = tensors[0] if prev_idx_next < 0 else tensors[prev_idx_next + 1]
                    # 简化探测：顺序向后，遇>HIGH早停，收集>LOW窗内
                    inwin = []
                    for k2 in range(prev_idx_next + 1, len(cands)):
                        name2 = cands[k2].name
                        mb2 = probe_size(m_in_next, tensors[k2 + 1], f'avail_seg{seg_idx_now+1}_{name2}')
                        if mb2 is None:
                            continue
                        if mb2 > HIGH:
                            break
                        if mb2 > LOW:
                            inwin.append((k2, name2, mb2))
                    if not inwin:
                        return False
                    if seg_left_now == 2:
                        # seg7 有>LOW窗内 + seg8(OUTPUT) > LOW 即可
                        for (k2, nm2, mb2) in inwin:
                            m_in_last = tensors[k2 + 1]
                            mb8 = probe_size(m_in_last, tensors[-1], 'avail_last')
                            if mb8 is not None and mb8 > LOW:
                                return True
                        return False
                    # seg_left_now>=3：只要下一段能取>LOW窗内，递归近似通过
                    return True
                if not _availability_ok_after_pick_simple(j, mb_j, cur_seg_idx, rem_after, seg_left_after):
                    continue
                mb_bt, ok_bt = finalize_segment(m_in_local, tensors[j + 1], cur_seg_idx)
                if ok_bt and LOW <= mb_bt <= HIGH:
                    print(f'  回退固化: seg{cur_seg_idx} -> {name_j} = {mb_bt:.2f}MB')
                    return j, name_j, mb_bt
        return None

    # 让上一段“向前（更早）”移动，为当前段腾出空间（应对全<LOW）
    def shift_previous_backward_to_give_space(prev_seg_k):
        if prev_seg_k < 1:
            return None
        prev_prev_idx = decisions[prev_seg_k - 2][0] if prev_seg_k - 2 >= 0 else -1
        cur_idx = decisions[prev_seg_k - 1][0]
        m_in_prev = tensors[0] if prev_prev_idx < 0 else tensors[prev_prev_idx + 1]
        for j in range(cur_idx - 1, prev_prev_idx, -1):
            name_j = cands[j].name
            mb_j = probe_size(m_in_prev, tensors[j + 1], f'shift_prev_back_seg{prev_seg_k}_{name_j}')
            if mb_j is None:
                continue
            if LOW <= mb_j <= HIGH:
                mb_fix, ok_fix = finalize_segment(m_in_prev, tensors[j + 1], prev_seg_k)
                if ok_fix and LOW <= mb_fix <= HIGH:
                    decisions[prev_seg_k - 1] = (j, mb_fix)
                    print(f'  回退：seg{prev_seg_k} 前移(更早)到 {name_j} = {mb_fix:.2f}MB')
                    return j, mb_fix
        return None

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

        # 选择最接近目标的候选，并按回退/前进找第一个合规（LOW-HIGH）
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

        # 若当前段最小候选仍 > HIGH，优先对“当前段”逆向回退并即刻固化；失败再启用旧的回退机制2
        min_mb = min(x[2] for x in probes)
        if pick is None and min_mb > HIGH:
            print(f'第{seg_idx}段 min={min_mb:.2f}MB > {HIGH}MiB，当前段逆向回退')
            bt = backtrack_current_and_finalize(prev_idx, probes[0][0], seg_idx, segments_left, remaining_mb)
            if bt is not None:
                i, name, mb_fix = bt
                prev_idx = i
                remaining_mb -= mb_fix
                print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓ (回退)')
                log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB (bt)\n' + '-' * 60 + '\n')
                seg_idx += 1
                continue
            print(f'第{seg_idx}段回退未命中，级联回退上一段向更早切点')
            k = len(decisions)  # 上一段编号
            success = False
            while k >= 1:
                moved = shift_previous_backward_to_give_space(k)
                if moved is not None:
                    prev_idx = decisions[-1][0]
                    # 重新探测当前段并固化
                    target = remaining_mb / (SEGMENTS - len(decisions))
                    probes = []
                    m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
                    for i2 in range(prev_idx + 1, len(cands)):
                        name2 = cands[i2].name
                        mb2 = probe_size(m_in, tensors[i2 + 1], f'seg{seg_idx}_{name2}')
                        if mb2 is None:
                            continue
                        if mb2 > HIGH:
                            break
                        if mb2 > LOW:
                            probes.append((i2, name2, mb2))
                    if probes:
                        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
                        i, name, mb = probes_sorted[0]
                        mb_fix, ok2 = finalize_segment(m_in, tensors[i + 1], seg_idx)
                        if ok2 and LOW <= mb_fix <= HIGH:
                            prev_idx = i
                            remaining_mb -= mb_fix
                            print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓ (回退)')
                            log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB (bt)\n' + '-' * 60 + '\n')
                            seg_idx += 1
                            success = True
                            break
                k -= 1
            if success:
                continue
            print('级联回退失败，无法吃下溢出')
            sys.exit(5)

        # 若所有候选都 < LOW：让上一段向“更早”回退为当前段腾空间
        all_small = all(mb < LOW for (_i, _n, mb) in probes) if probes else False
        if pick is None and all_small:
            print(f'第{seg_idx}段候选全<LOW，回退上一段向更早切点')
            # 从紧邻上一段开始逐段向前尝试回退
            k = len(decisions)
            success = False
            while k >= 1:
                moved = shift_previous_backward_to_give_space(k)
                if moved is not None:
                    # 更新 prev_idx 并重新探测当前段
                    prev_idx = decisions[-1][0]
                    target = remaining_mb / (SEGMENTS - len(decisions))
                    probes = []
                    m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
                    for i2 in range(prev_idx + 1, len(cands)):
                        name2 = cands[i2].name
                        mb2 = probe_size(m_in, tensors[i2 + 1], f'seg{seg_idx}_{name2}')
                        if mb2 is None:
                            continue
                        if mb2 > HIGH:
                            break
                        if mb2 > LOW:
                            probes.append((i2, name2, mb2))
                    if probes:
                        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
                        i, name, mb = probes_sorted[0]
                        mb_fix, ok2 = finalize_segment(m_in, tensors[i + 1], seg_idx)
                        if ok2 and LOW <= mb_fix <= HIGH:
                            prev_idx = i
                            remaining_mb -= mb_fix
                            print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓ (回退)')
                            log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB (bt)\n' + '-' * 60 + '\n')
                            seg_idx += 1
                            success = True
                            break
                k -= 1
            if success:
                continue
            print(f'第{seg_idx}段回退失败，无法提升到>LOW')
            sys.exit(6)

        if pick is None:
            # 无合规候选：同样执行“当前段逆向回退并即刻固化”
            print(f'第{seg_idx}段无合规候选，当前段逆向回退')
            # 若 probes 非空，使用 probes 最小索引作为当前终点；否则用 prev_idx+1 当作基准
            cur_end_idx = probes[0][0] if probes else (prev_idx + 1)
            bt = backtrack_current_and_finalize(prev_idx, cur_end_idx, seg_idx, segments_left, remaining_mb)
            if bt is not None:
                i, name, mb_fix = bt
                prev_idx = i
                remaining_mb -= mb_fix
                print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓ (回退)')
                log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB (bt)\n' + '-' * 60 + '\n')
                seg_idx += 1
                continue
            print(f'第{seg_idx}段回退失败，无法找到窗内候选')
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

        # 余均值护栏：若选后剩余平均 < LOW，执行“当前段优先”的逆向回退并即刻固化，然后从下一段继续
        if SEGMENTS - len(decisions) > 1:
            avg_rest = (remaining_mb - mb_fix) / (SEGMENTS - len(decisions) - 1)
            if avg_rest < LOW:
                print(f'  余均值护栏触发：avg_rest={avg_rest:.2f} < LOW，回退当前段')
                bt = backtrack_current_and_finalize(prev_idx, i, seg_idx, segments_left, remaining_mb)
                if bt is not None:
                    i, name, mb_fix = bt
                else:
                    print('  当前段回退失败')
                

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


