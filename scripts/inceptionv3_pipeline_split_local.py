#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= 统一静音（必须在 import tensorflow 前） =========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 屏蔽大多数 TF C++ INFO/WARN
# ==========================================================

import sys
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # 静音 absl

import tensorflow as tf

# 路径与参数（由 main 动态设定到对应模型目录）
ROOT = Path(__file__).resolve().parents[1]
BASE = None
OUT = None
TFL_DIR = None
TPU_DIR = None
LOG_DIR = None
LOG = None
COMPILER = str(ROOT / 'edgetpu_compiler')

LOW, HIGH = 1.0, 6.91
SEGMENTS = 8
TIMEOUT_S = 360

# 全局模型配置（默认 InceptionV3，运行时可通过 --model 覆盖）
MODEL_NAME = 'InceptionV3'
PREPROCESS_FN = tf.keras.applications.inception_v3.preprocess_input
INPUT_SIZE_HW = (299, 299)

MODEL_REGISTRY = {
    'InceptionV3': {
        'ctor': tf.keras.applications.InceptionV3,
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
        'input_size': (299, 299),
    },
    'ResNet50': {
        'ctor': tf.keras.applications.ResNet50,
        'preprocess': tf.keras.applications.resnet.preprocess_input,
        'input_size': (224, 224),
    },
    'ResNet101': {
        'ctor': tf.keras.applications.ResNet101,
        'preprocess': tf.keras.applications.resnet.preprocess_input,
        'input_size': (224, 224),
    },
    'DenseNet201': {
        'ctor': tf.keras.applications.DenseNet201,
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'input_size': (224, 224),
    },
    'Xception': {
        'ctor': tf.keras.applications.Xception,
        'preprocess': tf.keras.applications.xception.preprocess_input,
        'input_size': (299, 299),
    },
}

# 用于过滤无效日志的正则
DROP_PATTERNS = [
    r'Estimated count of arithmetic ops', r' MACs$',
    r'Ignored output_format', r'Ignored drop_control_dependency',
    r'NUMA node', r'Your kernel may have been built without NUMA support',
    r'Please consider providing the trackable_obj'
]
DROP_RE = re.compile('|'.join(DROP_PATTERNS))

def sanitize_log(s: str) -> str:
    return '\n'.join(line for line in s.splitlines() if not DROP_RE.search(line))

def chain_inference_check(base_model, preprocess_fn, input_size_hw, samples=5):
    """串联合成检查：逐段 TFLite 推理串联，与 Keras 全模输出比较。
    返回：{num_samples, top1_match, avg_abs_diff}
    """
    import numpy as np
    import tensorflow as tf
    h, w = input_size_hw
    # 读取段文件（按 seg 序）——优先使用 CPU TFLite，避免 edgetpu-custom-op 无法在 CPU 解释器加载
    seg_paths = sorted(TFL_DIR.glob('seg*_int8.tflite'))
    # 如确实只存在 TPU 版，再使用 TPU 版（需要外部 delegate 才能跑，通常仅做形状对齐检查）
    if not seg_paths:
        seg_paths = sorted(TPU_DIR.glob('seg*_int8_edgetpu.tflite'))

    interpreters = []
    for p in seg_paths:
        intr = tf.lite.Interpreter(model_path=str(p))
        intr.allocate_tensors()
        interpreters.append(intr)

    def run_chain(img):
        x = preprocess_fn(img.astype(np.float32))[None, ...]
        fm = x
        for intr in interpreters:
            inp = intr.get_input_details()[0]
            out = intr.get_output_details()[0]
            intr.set_tensor(inp['index'], fm.astype(inp['dtype']))
            intr.invoke()
            fm = intr.get_tensor(out['index'])
        return fm

    top1_match = 0
    diffs = []
    for i in range(samples):
        img = np.random.randint(0,255,(h,w,3),dtype=np.uint8)
        ref = base_model(preprocess_fn(img.astype(np.float32))[None,...], training=False).numpy()
        got = run_chain(img)
        diffs.append(np.mean(np.abs(ref - got)))
        if ref.argmax()==got.argmax():
            top1_match += 1

    return {
        'num_samples': samples,
        'top1_match': top1_match,
        'avg_abs_diff': float(np.mean(diffs))
    }

def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    TFL_DIR.mkdir(parents=True, exist_ok=True)
    TPU_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_write(s: str):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(s)

# ============ 代表性数据（校准） =========

def _gen_random_imgs(N=128, h=299, w=299, preprocess_fn=None):
    # 随机“伪图像”，走与真实图片相同预处理链路，稳定通道/分布
    for _ in range(N):
        img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8).astype(np.float32)
        x = preprocess_fn(img)[None, ...] if preprocess_fn is not None else img[None, ...]
        yield x

def rep_full_random(N=128, h=299, w=299, preprocess_fn=None):
    def gen():
        for x in _gen_random_imgs(N, h, w, preprocess_fn):
            yield [x.astype(np.float32)]
    return gen

def make_rep_from_prev(base_model, prev_tensor, N=128, h=299, w=299, preprocess_fn=None):
    """
    用基模型从输入前传到 prev_tensor，收集中间特征作为校准数据，
    保证 shape=(1,H,W,C) 与通道 C 与子图输入完全一致。
    """
    feeder = tf.keras.Model(base_model.input, prev_tensor)
    def gen():
        n = 0
        for x in _gen_random_imgs(N, h, w, preprocess_fn):
            y = feeder(x, training=False).numpy()  # 中间特征
            yield [y.astype(np.float32)]
            n += 1
            if n >= N: break
    return gen

# ============ 自动通道守护 + 回退量化 ============

CONV_ERR_PAT = re.compile(
    r'input_channel % filter_input_channel != 0|Node number \d+ \(CONV_2D\) failed to prepare'
)

def _build_converter_from_cf(model_sub, rep_fn, legacy=False):
    # 明确输入签名，避免量化器误判通道
    ish = model_sub.input.shape[1:]
    spec = tf.TensorSpec([1, *ish], tf.float32, name='input')

    @tf.function
    def f(x): return model_sub(x)

    cf = f.get_concrete_function(spec)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], trackable_obj=model_sub)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_fn
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    if legacy:
        # 仅对问题段降级：禁 per-channel + 旧量化器
        conv._experimental_disable_per_channel = True
        if hasattr(conv, "_experimental_new_quantizer"):
            conv._experimental_new_quantizer = False
    return conv

def _analyze_int8_and_channels(tflite_bytes, expected_c):
    intr = tf.lite.Interpreter(model_content=tflite_bytes)
    intr.allocate_tensors()
    inp = intr.get_input_details()[0]
    ok_dtype = (inp['dtype'] == np.int8)
    ok_chan = (inp['shape'][-1] == expected_c)
    any_float = any(t['dtype'] in (np.float32, np.float16) for t in intr.get_tensor_details())
    return ok_dtype and ok_chan, any_float, int(inp['shape'][-1])

def expected_in_channels(model_sub):
    return int(model_sub.input.shape[-1])

def convert_with_auto_channel_guard(model_sub, rep_fn):
    exp_c = expected_in_channels(model_sub)
    try:
        t0 = _build_converter_from_cf(model_sub, rep_fn, legacy=False).convert()
        ok, any_float, got_c = _analyze_int8_and_channels(t0, exp_c)
        if ok and not any_float:
            return t0, 'per_channel'
        # 有浮点残留或通道不符 → 尝试对该段回退
        t1 = _build_converter_from_cf(model_sub, rep_fn, legacy=True).convert()
        ok2, any_float2, got_c2 = _analyze_int8_and_channels(t1, exp_c)
        return t1, ('legacy_no_per_channel' if ok2 else 'legacy_float_left')
    except RuntimeError as e:
        if CONV_ERR_PAT.search(str(e)):
            t1 = _build_converter_from_cf(model_sub, rep_fn, legacy=True).convert()
            return t1, 'legacy_after_conv_err'
        raise

# ============ Edge TPU 编译（带日志过滤） ============

def compile_tpu(tfl_path: Path, tag: str):
    res = subprocess.run([COMPILER, '-o', str(TPU_DIR), str(tfl_path)],
                         capture_output=True, text=True, timeout=TIMEOUT_S)
    edgetpu_path = TPU_DIR / (tfl_path.stem + '_edgetpu.tflite')
    mb = edgetpu_path.stat().st_size / 1024 / 1024 if edgetpu_path.exists() else None
    log_write(f'[{tag}] compile {tfl_path.name}\n')
    if res.stdout:
        log_write(f'STDOUT:\n{sanitize_log(res.stdout)}\n')
    if res.stderr:
        log_write(f'STDERR:\n{sanitize_log(res.stderr)}\n')
    log_write('-' * 60 + '\n')
    # 另存每段原始日志
    ts = time.strftime('%Y%m%d-%H%M%S')
    per_log = LOG_DIR / f'{ts}_{tag.replace(" ","_")}.log'
    with open(per_log, 'w', encoding='utf-8') as f:
        if res.stdout:
            f.write('STDOUT\n')
            f.write(res.stdout)
            f.write('\n')
        if res.stderr:
            f.write('STDERR\n')
            f.write(res.stderr)
            f.write('\n')
    return mb, edgetpu_path.exists()

# ============ 你的分段逻辑（加入缓存/自动量化） ============

# 探测结果缓存，避免重复转/编（键用张量名对）
PROBE_CACHE = {}

def pick_candidates(model, model_name):
    # 放宽：模块/阶段出口，2D/3D/4D 都可；末端纳入 avg_pool
    cands = []
    for l in model.layers:
        name = l.name
        out = getattr(l, 'output', None)
        if out is None:
            continue
        rank = len(getattr(out, 'shape', ()))
        if rank not in (2, 3, 4):
            continue

        if model_name == 'InceptionV3':
            is_mixed = name.startswith('mixed') and '_' not in name
            is_avg_pool = (name == 'avg_pool')
            if is_mixed or is_avg_pool:
                cands.append(l)
        elif model_name.startswith('ResNet'):
            if re.match(r'conv[2-5]_block\d+_out$', name) or name == 'avg_pool':
                cands.append(l)
        elif model_name == 'DenseNet201':
            if re.match(r'pool[2-4]_pool$', name) or re.match(r'conv\d+_block\d+_concat$', name) or name == 'avg_pool':
                cands.append(l)
        elif model_name == 'Xception':
            if (name.startswith('block') and (name.endswith('add') or 'pool' in name)) or name == 'avg_pool':
                cands.append(l)
    return cands

def probe_size(base_model, m_in, t_out, name_tag, h, w, preprocess_fn):
    key = (getattr(m_in, 'name', str(m_in)), getattr(t_out, 'name', str(t_out)))
    if key in PROBE_CACHE:
        return PROBE_CACHE[key]
    try:
        sub = tf.keras.Model(m_in, t_out)
        rep_fn = make_rep_from_prev(base_model, m_in, N=64, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl, mode = convert_with_auto_channel_guard(sub, rep_fn)
        p = TFL_DIR / f'_{name_tag}.tflite'
        p.write_bytes(tfl)
        mb, ok = compile_tpu(p, f'probe:{name_tag} ({mode})')
        val = (mb if ok else None)
        PROBE_CACHE[key] = val
        return val
    except Exception as e:
        PROBE_CACHE[key] = None
        return None

def finalize_segment(base_model, m_in, t_out, seg_idx, h, w, preprocess_fn):
    sub = tf.keras.Model(m_in, t_out, name=f'seg{seg_idx}')
    rep_fn = make_rep_from_prev(base_model, m_in, N=128, h=h, w=w, preprocess_fn=preprocess_fn)
    tfl, mode = convert_with_auto_channel_guard(sub, rep_fn)
    p = TFL_DIR / f'seg{seg_idx}_int8.tflite'
    p.write_bytes(tfl)
    mb, ok = compile_tpu(p, f'finalize:seg{seg_idx} ({mode})')
    return mb, ok

def greedy_with_backtrack(model, model_name, preprocess_fn, input_size_hw):
    ensure_dirs()
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write(f'PIPELINE greedy split time={datetime.now()}\n')
        f.write('=' * 70 + '\n\n')

    # 整模大小（用于估算目标；不要求编 EdgeTPU）
    h, w = input_size_hw
    rep_fn_full = rep_full_random(N=128, h=h, w=w, preprocess_fn=preprocess_fn)
    full_tfl, mode = convert_with_auto_channel_guard(model, rep_fn_full)
    full_tfl_path = TFL_DIR / 'full_int8.tflite'
    full_tfl_path.write_bytes(full_tfl)

    # 这里编译一次整模仅为拿到 EdgeTPU 后大小估计；如不需要可跳过
    full_mb, ok = compile_tpu(full_tfl_path, f'full_model ({mode})')
    if not ok or full_mb is None:
        print('整模编译失败')
        sys.exit(1)
    print(f'FULL TPU: {full_mb:.2f}MB')
    log_write(f'FULL TPU SIZE: {full_mb:.2f}MB\n' + '-' * 60 + '\n')

    cands = pick_candidates(model, model_name)
    assert len(cands) >= SEGMENTS - 1, '候选数量不足以切成8段'

    tensors = [model.input] + [l.output for l in cands] + [model.output]

    def probe_from(prev_idx, seg_idx, target):
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        results = []
        for i in range(prev_idx + 1, len(cands)):
            name = cands[i].name
            mb = probe_size(model, m_in, tensors[i + 1], f'seg{seg_idx}_{name}', h, w, preprocess_fn)
            if mb is not None:
                results.append((i, name, mb))
        i = len(cands)
        name = 'OUTPUT'
        mb = probe_size(model, m_in, tensors[-1], f'seg{seg_idx}_OUTPUT', h, w, preprocess_fn)
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
            mb_j = probe_size(model, m_in, tensors[j + 1], f'shift_prev_seg{prev_seg_k + 1}_{cands[j].name}', h, w, preprocess_fn)
            if mb_j is None:
                continue
            if mb_j <= HIGH:
                decisions[prev_seg_k] = (j, mb_j)
                print(f'  回退：seg{prev_seg_k + 1} 前移到 {cands[j].name} = {mb_j:.2f}MB')
                log_write(f'BACKTRACK: seg{prev_seg_k + 1} -> {cands[j].name} = {mb_j:.2f}MB\n')
                return True
        return False

    def try_shift_previous_shrink(prev_seg_k, decisions):
        if prev_seg_k < 0:
            return False
        prev_prev_idx = decisions[prev_seg_k - 1][0] if prev_seg_k - 1 >= 0 else -1
        prev_idx = decisions[prev_seg_k][0]
        m_in = tensors[0] if prev_prev_idx < 0 else tensors[prev_prev_idx + 1]
        for j in range(prev_idx - 1, prev_prev_idx, -1):
            mb_j = probe_size(model, m_in, tensors[j + 1], f'shrink_prev_seg{prev_seg_k + 1}_{cands[j].name}', h, w, preprocess_fn)
            if mb_j is None:
                continue
            if LOW <= mb_j <= HIGH:
                decisions[prev_seg_k] = (j, mb_j)
                print(f'  过小回退：seg{prev_seg_k + 1} 后移到 {cands[j].name} = {mb_j:.2f}MB')
                log_write(f'SMALL_BACKTRACK: seg{prev_seg_k + 1} -> {cands[j].name} = {mb_j:.2f}MB\n')
                return True
        return False

    decisions = []
    prev_idx = -1
    seg_idx = 1
    remaining_mb = full_mb
    cut_layer_names = []

    while seg_idx <= SEGMENTS:
        segments_left = SEGMENTS - len(decisions)
        if segments_left == 1:
            m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            mb, ok2 = finalize_segment(model, m_in, tensors[-1], seg_idx, h, w, preprocess_fn)
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
                        mb, ok2 = finalize_segment(model, m_in, tensors[-1], seg_idx, h, w, preprocess_fn)
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

        target = remaining_mb / segments_left
        probes = probe_from(prev_idx, seg_idx, target)
        if not probes:
            print('无可探测候选')
            sys.exit(4)

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
            min_mb = min(x[2] for x in probes)
            max_mb = max(x[2] for x in probes)
            if max_mb < LOW:
                print(f'第{seg_idx}段全候选 < LOW，启动过小回退')
                ret_ok = False
                k = len(decisions) - 1
                while k >= 0 and not ret_ok:
                    moved = try_shift_previous_shrink(k, decisions)
                    if moved:
                        prev_idx = decisions[-1][0]
                        probes = probe_from(prev_idx, seg_idx, target)
                        if not probes:
                            k -= 1
                            continue
                        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
                        idx0 = probes.index(probes_sorted[0])
                        pick = None
                        for p in range(idx0, -1, -1):
                            i, name, mb = probes[p]
                            if LOW <= mb <= HIGH:
                                pick = (i, name, mb)
                                break
                        if pick is None:
                            for p in range(idx0 + 1, len(probes)):
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
                    print(f'过小回退失败，无法为第{seg_idx}段找到合规候选')
                    sys.exit(6)
            elif min_mb > HIGH:
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
                            idx0 = 0
                            pick = None
                            for p in range(idx0, -1, -1):
                                i, name, mb = probes[p]
                                if LOW <= mb <= HIGH:
                                    pick = (i, name, mb)
                                    break
                            if pick is None:
                                for p in range(idx0 + 1, len(probes)):
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
            else:
                print(f'第{seg_idx}段无窗口解（min<{LOW} < {HIGH}<max），尝试跨段补偿')
                ret_ok = False
                k = len(decisions) - 1
                while k >= 0 and not ret_ok:
                    moved_small = try_shift_previous_shrink(k, decisions)
                    moved_large = moved_small or try_shift_previous_to_absorb(k, decisions)
                    if moved_large:
                        prev_idx = decisions[-1][0]
                        probes = probe_from(prev_idx, seg_idx, target)
                        if not probes:
                            k -= 1
                            continue
                        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
                        idx0 = probes.index(probes_sorted[0])
                        pick = None
                        for p in range(idx0, -1, -1):
                            i, name, mb = probes[p]
                            if LOW <= mb <= HIGH:
                                pick = (i, name, mb)
                                break
                        if pick is None:
                            for p in range(idx0 + 1, len(probes)):
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
                    print(f'跨段补偿失败，无法为第{seg_idx}段找到合规候选')
                    sys.exit(6)

        i, name, mb = pick
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        mb_fix, ok2 = finalize_segment(model, m_in, tensors[i + 1], seg_idx, h, w, preprocess_fn)
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
        if name != 'OUTPUT':
            cut_layer_names.append(name)
        seg_idx += 1

    tpu_files = sorted(TPU_DIR.glob('seg*_int8_edgetpu.tflite'))
    ok_cnt = 0
    for f in tpu_files:
        mb = f.stat().st_size / 1024 / 1024
        ok = LOW <= mb <= HIGH
        ok_cnt += int(ok)
        print(f'{f.name}: {mb:.2f}MB {"OK" if ok else "NG"}')
    print(f'合规: {ok_cnt}/{len(tpu_files)}')
    summary = {
        'segments': SEGMENTS,
        'low': LOW,
        'high': HIGH,
        'full_mb': full_mb,
        'ok_segments': ok_cnt,
        'total_segments': len(tpu_files),
        'model': MODEL_NAME,
        'cut_names': cut_layer_names
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    try:
        check = chain_inference_check(model, preprocess_fn, (h, w))
        (OUT / 'chain_check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
        print(f"chain_check: top1_match={check['top1_match']}/{check['num_samples']}, avg_abs_diff={check['avg_abs_diff']:.6f}")
    except Exception as e:
        print('chain_check failed:', e)

def main():
    # 先声明 global，避免在本函数中“声明前使用”
    global SEGMENTS, LOW, HIGH, OUT, TFL_DIR, TPU_DIR, LOG_DIR, LOG, BASE, MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', type=int, default=SEGMENTS)
    parser.add_argument('--low', type=float, default=LOW)
    parser.add_argument('--high', type=float, default=HIGH)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--model', type=str, default='InceptionV3', choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    seg = args.segments
    low = args.low
    high = args.high
    suffix = args.suffix

    # 应用参数到全局路径与阈值
    SEGMENTS = seg
    LOW = low
    HIGH = high

    MODEL_NAME = args.model
    reg = MODEL_REGISTRY[MODEL_NAME]
    ctor = reg['ctor']
    preprocess_fn = reg['preprocess']
    h, w = reg['input_size']

    # 输出路径按模型名组织
    model_key = MODEL_NAME.lower()
    BASE = ROOT / 'models_local' / 'public' / f'{model_key}_8seg_uniform_local'
    OUT = BASE / 'full_split_pipeline_local'
    TFL_DIR = OUT / 'tflite'
    TPU_DIR = OUT / 'tpu'
    LOG_DIR = OUT / 'logs'
    LOG = OUT / 'compilation_log.txt'

    print(f'=== PIPELINE split {MODEL_NAME}: segments={SEGMENTS} S=[{LOW},{HIGH}] ===')
    m = ctor(weights='imagenet', include_top=True)
    greedy_with_backtrack(m, MODEL_NAME, preprocess_fn, (h, w))

if __name__ == '__main__':
    main()
