#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= 统一静音 =========
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
    edgetpu_path = TPU_DIR / (tfl_path.stem + '_edgetpu.tflite')
    # 防止旧文件掩盖失败：先清理同名输出
    try:
        if edgetpu_path.exists():
            edgetpu_path.unlink()
    except Exception:
        pass
    res = subprocess.run([COMPILER, '-o', str(TPU_DIR), str(tfl_path)],
                         capture_output=True, text=True, timeout=TIMEOUT_S)
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

# ============ 分段逻辑（加入缓存/自动量化） ============

# 探测结果缓存，避免重复转/编（键用张量名对）
PROBE_CACHE = {}

def pick_candidates(model, model_name):
    # 放宽：模块/阶段出口，2D/3D/4D 都可；统一纳入常见汇合/池化类锚点
    cands = []
    seen = set()
    for l in model.layers:
        name = l.name
        out = getattr(l, 'output', None)
        if out is None:
            continue
        rank = len(getattr(out, 'shape', ()))
        if rank not in (2, 3, 4):
            continue

        # 全局通用候选：Add / Concatenate / 各类 Pool（包含 GlobalAveragePooling2D）/ keras 命名的 avg_pool
        is_add = isinstance(l, tf.keras.layers.Add)
        is_concat = isinstance(l, tf.keras.layers.Concatenate)
        is_pool = isinstance(l, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D, tf.keras.layers.GlobalAveragePooling2D)) or name == 'avg_pool'
        if is_add or is_concat or is_pool:
            if id(l) not in seen:
                cands.append(l)
                seen.add(id(l))

        if model_name == 'InceptionV3':
            is_mixed = name.startswith('mixed') and '_' not in name
            is_avg_pool = (name == 'avg_pool')
            if is_mixed or is_avg_pool:
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name.startswith('ResNet'):
            if re.match(r'conv[2-5]_block\d+_out$', name) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name == 'DenseNet201':
            if re.match(r'pool[2-4]_pool$', name) or re.match(r'conv\d+_block\d+_concat$', name) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name == 'Xception':
            # 方案A+：Add/Pooling/avg_pool + 无残差尾段 block14 的激活作为候选
            # 1) 残差汇合/池化层（单输入安全）
            if isinstance(l, (tf.keras.layers.Add, tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
                continue
            # 2) 无残差尾段：block14 的激活点（单输入，便于细分尾部）
            if name.startswith('block14_') and name.endswith('_act'):
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
    return cands

def probe_size(base_model, m_in, t_out, name_tag, h, w, preprocess_fn):
    key = (getattr(m_in, 'name', str(m_in)), getattr(t_out, 'name', str(t_out)))
    if key in PROBE_CACHE:
        return PROBE_CACHE[key]
    try:
        sub = tf.keras.Model(m_in, t_out)
        rep_fn = make_rep_from_prev(base_model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl, mode = convert_with_auto_channel_guard(sub, rep_fn)
        tfl_mb = len(tfl) / 1024 / 1024
        
        # 如果 TFLite 大小 < 6MB，跳过 EdgeTPU 编译直接使用 TFLite 大小
        if tfl_mb < 6.0:
            val = tfl_mb
            print(f'  probe {name_tag}: TFLite={tfl_mb:.2f}MB <6MB, 跳过EdgeTPU编译')
        else:
            p = TFL_DIR / f'_{name_tag}.tflite'
            p.write_bytes(tfl)
            mb, ok = compile_tpu(p, f'probe:{name_tag} ({mode})')
            if ok:
                val = mb
            else:
                # EdgeTPU 编译失败时，使用 TFLite 大小作为备选
                val = tfl_mb
                print(f'  probe {name_tag}: EdgeTPU编译失败，使用TFLite大小 {tfl_mb:.2f}MB')
        
        PROBE_CACHE[key] = val
        return val
    except ValueError as e:
        if "`inputs` not connected to `outputs`" in str(e):
            # Xception跳跃连接问题：尝试回退到最近的Add层
            result = _try_fallback_to_add_layer(base_model, m_in, t_out, name_tag, h, w, preprocess_fn)
            PROBE_CACHE[key] = result
            return result
        else:
            PROBE_CACHE[key] = None
            return None
    except Exception as e:
        PROBE_CACHE[key] = None
        return None

def _try_fallback_to_add_layer(base_model, m_in, t_out, name_tag, h, w, preprocess_fn):
    """
    当遇到跳跃/多输入合流问题时，尝试回退到最近的 Merge 层（Add/Concatenate/Multiply/Average/Maximum/Minimum）
    """
    # 找到t_out对应的层
    t_out_layer = None
    for layer in base_model.layers:
        if hasattr(layer, 'output') and layer.output is t_out:
            t_out_layer = layer
            break
    
    if not t_out_layer:
        print(f'  {name_tag}: 无法找到输出层，回退失败')
        return None
    
    # 定位t_out_layer的索引
    t_out_layer_idx = None
    for i, layer in enumerate(base_model.layers):
        if layer is t_out_layer:
            t_out_layer_idx = i
            break
    
    if t_out_layer_idx is None:
        print(f'  {name_tag}: 无法找到层索引，回退失败')
        return None
    
    # 先尝试回退到“模型自身的合法切点”（避免跨合流）：
    def is_model_anchor(layer_name: str) -> bool:
        try:
            if 'MODEL_NAME' in globals():
                mn = MODEL_NAME
            else:
                mn = ''
        except Exception:
            mn = ''
        if mn == 'InceptionV3':
            return layer_name.startswith('mixed') and ('_' not in layer_name)
        if mn.startswith('ResNet'):
            return re.match(r'conv[2-5]_block\d+_out$', layer_name) is not None
        if mn == 'DenseNet201':
            return re.match(r'pool[2-4]_pool$', layer_name) is not None or re.match(r'conv\d+_block\d+_concat$', layer_name) is not None
        if mn == 'Xception':
            return layer_name.startswith('block14_') and layer_name.endswith('_act')
        return False

    for i in range(t_out_layer_idx - 1, -1, -1):
        layer = base_model.layers[i]
        nm = getattr(layer, 'name', '')
        if not is_model_anchor(nm):
            continue
        try:
            fallback_sub = tf.keras.Model(layer.output, t_out)
            rep_fn = make_rep_from_prev(base_model, layer.output, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
            tfl, mode = convert_with_auto_channel_guard(fallback_sub, rep_fn)
            tfl_mb = len(tfl) / 1024 / 1024
            print(f'  {name_tag}: 回退到锚点 {nm}，大小={tfl_mb:.2f}MB')
            return tfl_mb
        except Exception:
            continue

    # 从 t_out 向前搜索最近的合流层：
    # - Xception：仅允许 Add（避免多输入路径不全导致的构图问题）
    # - 其他模型：允许 Merge（Add/Concatenate/Multiply/Average/Maximum/Minimum）
    if 'MODEL_NAME' in globals() and MODEL_NAME == 'Xception':
        allowed_merge = (tf.keras.layers.Add,)
    else:
        allowed_merge = (
            tf.keras.layers.Add,
            tf.keras.layers.Concatenate,
            tf.keras.layers.Multiply,
            tf.keras.layers.Average,
            tf.keras.layers.Maximum,
            tf.keras.layers.Minimum,
        )

    # 向前寻找合适的回退点
    for i in range(t_out_layer_idx - 1, -1, -1):
        layer = base_model.layers[i]
        if isinstance(layer, allowed_merge):
            try:
                # 尝试从这个 Merge 层到目标层创建子模型
                fallback_sub = tf.keras.Model(layer.output, t_out)
                rep_fn = make_rep_from_prev(base_model, layer.output, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
                tfl, mode = convert_with_auto_channel_guard(fallback_sub, rep_fn)
                tfl_mb = len(tfl) / 1024 / 1024
                print(f'  {name_tag}: 回退到 {layer.name}，大小={tfl_mb:.2f}MB')
                return tfl_mb
            except Exception as e:
                continue
    
    print(f'  {name_tag}: 无法找到合适的 Merge 回退点')
    return None

def finalize_segment(base_model, m_in, t_out, seg_idx, h, w, preprocess_fn):
    def _try_build(m_in_, t_out_):
        sub_ = tf.keras.Model(m_in_, t_out_, name=f'seg{seg_idx}')
        rep_fn_ = make_rep_from_prev(base_model, m_in_, N=32, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl_, mode_ = convert_with_auto_channel_guard(sub_, rep_fn_)
        p_ = TFL_DIR / f'seg{seg_idx}_int8.tflite'
        p_.write_bytes(tfl_)
        mb_, ok_ = compile_tpu(p_, f'finalize:seg{seg_idx} ({mode_})')
        return mb_, ok_

    try:
        return _try_build(m_in, t_out)
    except Exception as e:
        # 段内收缩：从 t_out 向后回溯，寻找最接近且可构建的终点
        print(f'  finalize seg{seg_idx}: 原终点不可达，段内收缩…')
        # 找到 t_out 对应层索引
        t_out_layer_idx = None
        for idx, layer in enumerate(base_model.layers):
            if hasattr(layer, 'output') and (layer.output is t_out):
                t_out_layer_idx = idx
                break
        if t_out_layer_idx is None:
            return None, False
        # 向后回溯，尝试更早终点
        for j in range(t_out_layer_idx - 1, -1, -1):
            layer_j = base_model.layers[j]
            tout_j = getattr(layer_j, 'output', None)
            if tout_j is None:
                continue
            try:
                mb_j, ok_j = _try_build(m_in, tout_j)
                if ok_j and LOW <= mb_j <= HIGH:
                    print(f'  finalize seg{seg_idx}: 收缩到 {layer_j.name} = {mb_j:.2f}MB')
                    return mb_j, ok_j
            except Exception:
                continue
        return None, False

def greedy_with_backtrack(model, model_name, preprocess_fn, input_size_hw):
    ensure_dirs()
    # 清理旧的分段产物，避免上一轮残留影响本轮串联/探测
    try:
        for p in list(TFL_DIR.glob('seg*_int8*.tflite')):
            p.unlink(missing_ok=True)
        for p in list(TPU_DIR.glob('seg*_int8*_edgetpu.tflite')):
            p.unlink(missing_ok=True)
    except Exception:
        pass
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write(f'PIPELINE greedy split time={datetime.now()}\n')
        f.write('=' * 70 + '\n\n')

    # 整模大小（用于估算目标；不要求编 EdgeTPU）
    h, w = input_size_hw
    rep_fn_full = rep_full_random(N=32, h=h, w=w, preprocess_fn=preprocess_fn)
    full_tfl, mode = convert_with_auto_channel_guard(model, rep_fn_full)
    full_tfl_path = TFL_DIR / 'full_int8.tflite'
    full_tfl_path.write_bytes(full_tfl)

    # 这里编译一次整模仅为拿到 EdgeTPU 后大小估计
    full_mb, ok = compile_tpu(full_tfl_path, f'full_model ({mode})')
    if not ok or full_mb is None:
        print('整模编译失败')
        sys.exit(1)
    print(f'FULL TPU: {full_mb:.2f}MB')
    log_write(f'FULL TPU SIZE: {full_mb:.2f}MB\n' + '-' * 60 + '\n')

    cands = pick_candidates(model, model_name)
    assert len(cands) >= SEGMENTS - 1, '候选数量不足以切成8段'

    tensors = [model.input] + [l.output for l in cands] + [model.output]

    def probe_from(prev_idx, seg_idx, target, segments_left):
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        results = []
        probe_attempts = 0
        probe_failures = 0
        early_stopped = False

        # 列出所有候选切点及与上一段的“距离”（层索引差），方便诊断
        try:
            print('  所有候选切点(含未探测):')
            for ci in range(prev_idx + 1, len(cands)):
                dist = ci - prev_idx
                print(f'    - {cands[ci].name}: dist={dist}')
            if segments_left == 1:
                dist_out = len(cands) - prev_idx
                print(f'    - OUTPUT: dist={dist_out}')
        except Exception:
            pass
        
        # 早停优化：探测到第一个 >HIGH 的候选点就停止
        for i in range(prev_idx + 1, len(cands)):
            # Xception 专用：第5段及之后禁用 Pool 作为切点
            if model_name == 'Xception' and seg_idx >= 5:
                l = cands[i]
                if isinstance(l, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)) or l.name == 'avg_pool':
                    continue
            # InceptionV3 专用：非最后一段禁用内部 pool；仅允许顶层 mixed*（不再允许 Concatenate，避免跨 concat 造成不连通）
            if model_name == 'InceptionV3' and (SEGMENTS - len(decisions)) > 1:
                l = cands[i]
                nm = getattr(l, 'name', '')
                is_pool = isinstance(l, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)) or nm == 'avg_pool'
                is_top_mixed = nm.startswith('mixed') and ('_' not in nm)
                if is_pool:
                    continue
                if not is_top_mixed:
                    continue
            name = cands[i].name
            probe_attempts += 1
            mb = probe_size(model, m_in, tensors[i + 1], f'seg{seg_idx}_{name}', h, w, preprocess_fn)
            if mb is not None:
                results.append((i, name, mb))
                # 如果当前候选点已经超过上限，后续候选点肯定更大，提前停止
                if mb > HIGH:
                    print(f'  早停：{name}={mb:.2f}MB >{HIGH}MB，跳过后续候选')
                    early_stopped = True
                    break
            else:
                probe_failures += 1
        
        # 仅当最后一段才探测 OUTPUT
        if segments_left == 1:
            i = len(cands)
            name = 'OUTPUT'
            probe_attempts += 1
            mb = probe_size(model, m_in, tensors[-1], f'seg{seg_idx}_OUTPUT', h, w, preprocess_fn)
            if mb is not None:
                results.append((i, name, mb))
            else:
                probe_failures += 1
        
        # 如果返回空列表，打印详细信息
        if not results:
            print(f'  probe_from 详细信息:')
            print(f'    探测尝试: {probe_attempts}')
            print(f'    探测失败: {probe_failures}')
            print(f'    早停: {early_stopped}')
            print(f'    可用候选: {len(cands) - (prev_idx + 1)}')
        else:
            # 打印已探测候选的估算大小及距离
            print('  已探测候选:')
            for (ii, nm, mbb) in results:
                dist2 = (ii - prev_idx) if ii < len(cands) else (len(cands) - prev_idx)
                print(f'    - {nm}: dist={dist2}, est={mbb:.2f}MB')
        
        return results

    # 可行性加强护栏：确保“最小 n-1 个可用节点”可达
    # - 当剩余2段时：当前段选定后，最后一段(OUTPUT)从新起点到输出需在窗口内
    # - 当剩余≥3段时：当前段选定后，下一段需存在至少一个窗口内候选，且从该候选到 OUTPUT 的最后一段大小也在窗口内
    def _estimate_last_mb(prev_idx_local: int) -> float | None:
        m_in_local = tensors[0] if prev_idx_local < 0 else tensors[prev_idx_local + 1]
        try:
            est = _estimate_tflite_size_mb(model, m_in_local, tensors[-1], 'avail_last', h, w, preprocess_fn)
            return est
        except Exception:
            return None

    def availability_ok_after_pick(i_pick: int, mb_pick: float, seg_idx_now: int, remaining_after_pick: float, segments_left_now: int) -> bool:
        # segments_left_now 是当前段选定后的剩余段数（不含当前段）
        if segments_left_now <= 1:
            # 剩余≤1段：直接接 OUTPUT，无需预留
            return True
        if segments_left_now == 2:
            # 剩余2段：seg7+seg8，seg7需要1个>LOW候选，seg8接OUTPUT
            prev_idx_next = i_pick
            target_next = remaining_after_pick / 2
            # 检查 seg7 是否有 >LOW 候选
            next_probes = probe_from(prev_idx_next, seg_idx_now + 1, target_next, 2)
            if not next_probes:
                return False
            seg7_valid = [(j, nm, mb) for (j, nm, mb) in next_probes if (nm != 'OUTPUT' and (mb > LOW) and (mb <= HIGH))]
            if not seg7_valid:
                return False
            # 检查 seg7 选定后，seg8 能接 OUTPUT 且 >LOW
            for (j, nm, mbj) in seg7_valid:
                seg8_mb = _estimate_last_mb(j)
                if (seg8_mb is not None) and (seg8_mb > LOW) and (seg8_mb <= HIGH):
                    return True
            return False
        # 剩余≥3段：需要为中间段预留候选，最后段接 OUTPUT
        prev_idx_next = i_pick
        target_next = remaining_after_pick / segments_left_now
        next_probes = probe_from(prev_idx_next, seg_idx_now + 1, target_next, segments_left_now - 1)
        if not next_probes:
            return False
        # 仅考虑窗口内候选（排除 OUTPUT）
        next_valid = [(j, nm, mb) for (j, nm, mb) in next_probes if (nm != 'OUTPUT' and (mb > LOW) and (mb <= HIGH))]
        for (j, nm, mbj) in next_valid:
            last_mb = _estimate_last_mb(j)
            if (last_mb is not None) and (last_mb > LOW) and (last_mb <= HIGH):
                return True
        return False

    # 在回退导致截断后，清理被截断段位之后的旧文件，避免旧文件被链检误用
    def _cleanup_after_cut(start_seg_idx_inclusive: int):
        try:
            for p in list(TFL_DIR.glob('seg*_int8*.tflite')):
                try:
                    num = int(re.findall(r'seg(\d+)_', p.name)[0])
                except Exception:
                    continue
                if num >= start_seg_idx_inclusive:
                    p.unlink(missing_ok=True)
            for p in list(TPU_DIR.glob('seg*_int8*_edgetpu.tflite')):
                try:
                    num = int(re.findall(r'seg(\d+)_', p.name)[0])
                except Exception:
                    continue
                if num >= start_seg_idx_inclusive:
                    p.unlink(missing_ok=True)
        except Exception:
            pass

    # Xception 尾段锚点优先（细化 block14 尾部）
    if model_name == 'Xception' and segments_left <= 3 and feasible_candidates:
        anchor_names = {'block14_sepconv1_act', 'block14_sepconv2_act'}
        anchor_candidates = [(i, name, mb) for (i, name, mb) in feasible_candidates if name in anchor_names]
        anchor_candidates = apply_side_bias(anchor_candidates)
        if anchor_candidates:
            pick = min(anchor_candidates, key=lambda x: abs(x[2] - target))
        else:
            cands2 = apply_side_bias(feasible_candidates)
            pick = min(cands2, key=lambda x: abs(x[2] - target))   
            
     elif not feasible_candidates:
        # 护栏触发：当前段无敌回退+级联重建
        print(f'第{seg_idx}段触发护栏，启动无敌回退')
        _print_failure_context(seg_idx, target, probes, decisions, remaining_mb, segments_left)
        
        # 当前段无敌回退：找"最早能预留且自己>LOW"的点
        m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        backtrack_pick = None
        print(f'  seg{seg_idx} 无敌回退：寻找最早可行点...')
        
        for j in range(prev_idx - 1, -1, -1):  # 向前搜索更早终点
            if j < 0:
                break
            try:
                mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'guard_backtrack_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                    # 检查能否为后续预留
                    rem_after = remaining_mb - mb_est
                    if availability_ok_after_pick(j, mb_est, seg_idx, rem_after, segments_left - 1):
                        # 固化验证
                        mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                        if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                            backtrack_pick = (j, cands[j].name, mb_real)
                            print(f'    找到可行回退点: {cands[j].name} = {mb_real:.2f}MB')
                            break
            except Exception:
                continue
        
        if backtrack_pick is None:
            print(f'  seg{seg_idx} 无敌回退失败，无可行点')
            sys.exit(7)
        
        j_back, name_back, mb_back = backtrack_pick
        # 截断被覆盖段并清理
        if j_back < len(decisions):
            print(f'    覆盖 seg{j_back+1}~seg{len(decisions)}，截断重建')
            del decisions[j_back:]
            _cleanup_after_cut(j_back + 1)
        
        # 固化当前段
        decisions.append((j_back, mb_back))
        prev_idx = j_back
        remaining_mb = full_mb - sum(mb for _, mb in decisions)
        print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (无敌回退)')
        log_write(f'Guard-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB
' + '-' * 60 + '
')
        if name_back != 'OUTPUT':
            cut_layer_names.append(name_back)
        seg_idx += 1
        continue
            
            if backtrack_pick is None:
                print(f'  seg{seg_idx} 无敌回退失败，无可行点')
                sys.exit(7)
            
            j_back, name_back, mb_back = backtrack_pick
            # 截断被覆盖段并清理
            if j_back < len(decisions):
                print(f'    覆盖 seg{j_back+1}~seg{len(decisions)}，截断重建')
                del decisions[j_back:]
                _cleanup_after_cut(j_back + 1)
            
            # 固化当前段
            decisions.append((j_back, mb_back))
            prev_idx = j_back
            remaining_mb = full_mb - sum(mb for _, mb in decisions)
            print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (无敌回退)')
            log_write(f'Guard-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
            if name_back != 'OUTPUT':
                cut_layer_names.append(name_back)
            seg_idx += 1
            continue

    def _estimate_tflite_size_mb(model, m_in, t_out, name_tag, h, w, preprocess_fn):
        try:
            sub = tf.keras.Model(m_in, t_out)
            rep_fn = make_rep_from_prev(model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
            tfl, _ = convert_with_auto_channel_guard(sub, rep_fn)
            return len(tfl) / 1024 / 1024
        except Exception:
            # 估算路径同样尝试回退到最近的 Merge 层，避免前向扫描全军覆没
            fb = _try_fallback_to_add_layer(model, m_in, t_out, f'est_{name_tag}', h, w, preprocess_fn)
            return fb

    # 诊断日志：打印当前已决策段与候选分类
    def _print_failure_context(seg_idx_local: int, target_local: float, probes_list, decisions_list, remaining_local: float, segments_left_local: int):
        used_total = sum(mb for (_idx, mb) in decisions_list)
        print(f'诊断: seg{seg_idx_local}, target={target_local:.2f}MB, remaining={remaining_local:.2f}MB, used={used_total:.2f}MB')
        if decisions_list:
            print('  已固化段:')
            for k, (ci, mb) in enumerate(decisions_list, start=1):
                name_k = ci if isinstance(ci, str) else cands[ci].name
                print(f'    seg{k}: {name_k} = {mb:.2f}MB')
        if not probes_list:
            return
        below = [(nm, mb) for (_i, nm, mb) in probes_list if mb < LOW]
        inwin = [(nm, mb) for (_i, nm, mb) in probes_list if LOW <= mb <= HIGH]
        above = [(nm, mb) for (_i, nm, mb) in probes_list if mb > HIGH]
        print('  候选统计:')
        if below:
            print('    <LOW:')
            for nm, mb in below:
                print(f'      - {nm}: {mb:.2f}MB')
        if inwin:
            print('    in[LOW,HIGH]:')
            for nm, mb in inwin:
                print(f'      - {nm}: {mb:.2f}MB')
        if above:
            print('    >HIGH:')
            for nm, mb in above:
                print(f'      - {nm}: {mb:.2f}MB')
        # 护栏检查：选中后余下平均是否合规
        if segments_left_local > 1 and inwin:
            print('    护栏不通过（选后余均值不在窗内）:')
            for nm, mb in inwin:
                avg_rest = (remaining_local - mb) / (segments_left_local - 1)
                if not (LOW <= avg_rest <= HIGH):
                    print(f'      - {nm}: 选后avg={avg_rest:.2f}MB')

    decisions = []
    prev_idx = -1
    seg_idx = 1
    remaining_mb = full_mb
    cut_layer_names = []
    # 动态平均趋势跟踪：用于选择侧偏（防止连续下滑后尾段超限）
    trend_dir = None  # 'down' | 'up' | None
    trend_streak = 0
    prev_avg = None

    initial_avg = full_mb / SEGMENTS
    while seg_idx <= SEGMENTS:
        segments_left = SEGMENTS - seg_idx + 1  # 包含当前段的剩余段数
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
        # 计算动态平均趋势
        cur_trend = None
        if prev_avg is not None:
            if target < prev_avg:
                cur_trend = 'down'
            elif target > prev_avg:
                cur_trend = 'up'
            else:
                cur_trend = 'flat'
            if cur_trend in ('down', 'up'):
                if cur_trend == trend_dir:
                    trend_streak += 1
                else:
                    trend_dir = cur_trend
                    trend_streak = 1
            # flat 不改变 streak
        probes = probe_from(prev_idx, seg_idx, target, segments_left)
        if not probes:
            print(f'❌ 第{seg_idx}段：probe_from 返回空列表')
            print(f'   起始位置: prev_idx={prev_idx}')
            print(f'   目标大小: {target:.2f}MB')
            print(f'   可用候选数: {len(cands) - (prev_idx + 1)}')
            _print_failure_context(seg_idx, target, [], decisions, remaining_mb, segments_left)
            
            # 空列表：当前段直接回退（一步到位预留）
            print(f'   当前段直接回退找可行点...')
            m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            earliest_pick = None
            for j in range(prev_idx - 1, -1, -1):  # 向前搜索更早终点
                if j < 0:
                    break
                try:
                    mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'empty_backtrack_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                    if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                        if availability_ok_after_pick(j, mb_est, seg_idx, remaining_mb - mb_est, segments_left - 1):
                            mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                            if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                                earliest_pick = (j, cands[j].name, mb_real)
                                break
                except Exception:
                    continue
            if earliest_pick is not None:
                j_back, name_back, mb_back = earliest_pick
                if j_back < len(decisions) - 1:
                    del decisions[j_back:]
                    _cleanup_after_cut(j_back + 1)
                prev_idx = j_back
                remaining_mb = full_mb - sum(mb for _, mb in decisions) - mb_back
                decisions.append((j_back, mb_back))
                print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (本段回退-空列表)')
                log_write(f'Empty-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
                if name_back != 'OUTPUT':
                    cut_layer_names.append(name_back)
                seg_idx += 1
                continue
            else:
                print('❌ 无可探测候选（含估算回退失败）')
                print(f'   根本原因: 所有候选点都无法转换为合规的 TFLite 模型')
                sys.exit(4)

        probes_sorted = sorted(probes, key=lambda x: abs(x[2] - target))
        ideal_i = probes.index(probes_sorted[0])

        # 改进：选择候选点（带可行性护栏）
        # 注意：非最后一段时，排除 OUTPUT 作为候选点
        if segments_left > 1:
            # 非最后一段：排除 OUTPUT，且严格大于 LOW
            valid_candidates = [(i, name, mb) for i, name, mb in probes if (mb > LOW) and (mb <= HIGH) and name != 'OUTPUT']
        else:
            # 最后一段：允许所有候选点（包括 OUTPUT），且严格大于 LOW
            valid_candidates = [(i, name, mb) for i, name, mb in probes if (mb > LOW) and (mb <= HIGH)]
        
        # 先按可行性过滤：选中后剩余平均必须 >= LOW
        def feasible_after_pick(mb_pick: float) -> bool:
            if segments_left <= 1:
                return True
            avg_after = (remaining_mb - mb_pick) / (segments_left - 1)
            result = avg_after >= LOW
            print(f'    护栏检查: mb_pick={mb_pick:.2f}, remaining={remaining_mb:.2f}, segments_left={segments_left}, avg_after={avg_after:.2f}, pass={result}')
            return result

        feasible_candidates = [(i, name, mb) for (i, name, mb) in valid_candidates if feasible_after_pick(mb)]

        # 注：可达性护栏仅在 guard-only/回退求解中作为“推/退幅度”的判据，不在基础候选过滤阶段生效

        # 侧偏过滤：改为仅参考上一次趋势
        def apply_side_bias(cands):
            if trend_dir == 'down':
                side = [(i, name, mb) for (i, name, mb) in cands if mb >= target]
                return side if side else cands
            if trend_dir == 'up':
                side = [(i, name, mb) for (i, name, mb) in cands if mb <= target]
                return side if side else cands
            return cands

        # Xception 尾段锚点优先（细化 block14 尾部）
        if model_name == 'Xception' and segments_left <= 3 and feasible_candidates:
            anchor_names = {'block14_sepconv1_act', 'block14_sepconv2_act'}
            anchor_candidates = [(i, name, mb) for (i, name, mb) in feasible_candidates if name in anchor_names]
            anchor_candidates = apply_side_bias(anchor_candidates)
            if anchor_candidates:
                pick = min(anchor_candidates, key=lambda x: abs(x[2] - target))
            else:
                cands2 = apply_side_bias(feasible_candidates)
                pick = min(cands2, key=lambda x: abs(x[2] - target))    elif not feasible_candidates:
        # 护栏触发：当前段无敌回退+级联重建
        print(f'第{seg_idx}段触发护栏，启动无敌回退')
        _print_failure_context(seg_idx, target, probes, decisions, remaining_mb, segments_left)
        
        # 当前段无敌回退：找"最早能预留且自己>LOW"的点
        m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        backtrack_pick = None
        print(f'  seg{seg_idx} 无敌回退：寻找最早可行点...')
        
        for j in range(prev_idx - 1, -1, -1):  # 向前搜索更早终点
            if j < 0:
                break
            try:
                mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'guard_backtrack_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                    # 检查能否为后续预留
                    rem_after = remaining_mb - mb_est
                    if availability_ok_after_pick(j, mb_est, seg_idx, rem_after, segments_left - 1):
                        # 固化验证
                        mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                        if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                            backtrack_pick = (j, cands[j].name, mb_real)
                            print(f'    找到可行回退点: {cands[j].name} = {mb_real:.2f}MB')
                            break
            except Exception:
                continue
        
        if backtrack_pick is None:
            print(f'  seg{seg_idx} 无敌回退失败，无可行点')
            sys.exit(7)
        
        j_back, name_back, mb_back = backtrack_pick
        # 截断被覆盖段并清理
        if j_back < len(decisions):
            print(f'    覆盖 seg{j_back+1}~seg{len(decisions)}，截断重建')
            del decisions[j_back:]
            _cleanup_after_cut(j_back + 1)
        
        # 固化当前段
        decisions.append((j_back, mb_back))
        prev_idx = j_back
        remaining_mb = full_mb - sum(mb for _, mb in decisions)
        print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (无敌回退)')
        log_write(f'Guard-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB
' + '-' * 60 + '
')
        if name_back != 'OUTPUT':
            cut_layer_names.append(name_back)
        seg_idx += 1
        continue
            
            if backtrack_pick is None:
                print(f'  seg{seg_idx} 无敌回退失败，无可行点')
                sys.exit(7)
            
            j_back, name_back, mb_back = backtrack_pick
            # 截断被覆盖段并清理
            if j_back < len(decisions):
                print(f'    覆盖 seg{j_back+1}~seg{len(decisions)}，截断重建')
                del decisions[j_back:]
                _cleanup_after_cut(j_back + 1)
            
            # 固化当前段
            decisions.append((j_back, mb_back))
            prev_idx = j_back
            remaining_mb = full_mb - sum(mb for _, mb in decisions)
            print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (无敌回退)')
            log_write(f'Guard-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
            if name_back != 'OUTPUT':
                cut_layer_names.append(name_back)
            seg_idx += 1
            continue
        else:
            # 统一策略：选择最接近动态 target 的候选（不做下一段前瞻约束）
            cands2 = apply_side_bias(feasible_candidates)
            pick = min(cands2, key=lambda x: abs(x[2] - target))

        # 若仍未选中，回退到“尽量可行且更大”的候选
        if pick is None and valid_candidates:
            # 找到所有满足剩余平均 >= LOW 的候选中最大者
            larger_feasible = sorted([(i, name, mb) for (i, name, mb) in valid_candidates if feasible_after_pick(mb)], key=lambda x: x[2], reverse=True)
            if larger_feasible:
                pick = larger_feasible[0]

        min_mb = min(x[2] for x in probes)
        if pick is None and min_mb > HIGH:
            print(f'第{seg_idx}段 min={min_mb:.2f}MB > HIGH，当前段直接回退')
            # 当前段优先：找最早可行点（能预留且 ≤HIGH）
            m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            earliest_pick = None
            for j in range(prev_idx - 1, -1, -1):  # 向前搜索更早终点
                if j < 0:
                    break
                try:
                    mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'large_backtrack_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                    if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                        if availability_ok_after_pick(j, mb_est, seg_idx, remaining_mb - mb_est, segments_left - 1):
                            mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                            if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                                earliest_pick = (j, cands[j].name, mb_real)
                                break
                except Exception:
                    continue
            if earliest_pick is not None:
                j_back, name_back, mb_back = earliest_pick
                if j_back < len(decisions) - 1:
                    del decisions[j_back:]
                    _cleanup_after_cut(j_back + 1)
                prev_idx = j_back
                remaining_mb = full_mb - sum(mb for _, mb in decisions) - mb_back
                decisions.append((j_back, mb_back))
                print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (本段回退-过大)')
                log_write(f'Large-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
                if name_back != 'OUTPUT':
                    cut_layer_names.append(name_back)
                seg_idx += 1
                continue
            else:
                print('当前段回退失败，全候选过大且无可回退点')
                sys.exit(5)

        if pick is None:
            # 判断是否整体过小/过大/夹在两侧但无窗口解
            min_mb = min(x[2] for x in probes)
            max_mb = max(x[2] for x in probes)
            if max_mb < LOW:
                # 过小：当前段优先回退（一步到位预留），不逐级动前段
                print(f'第{seg_idx}段全候选 < LOW，当前段直接回退')
                _print_failure_context(seg_idx, target, probes, decisions, remaining_mb, segments_left)
                m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
                earliest_pick = None
                for j in range(prev_idx - 1, -1, -1):  # 向前搜索更早终点
                    if j < 0:
                        break
                    try:
                        mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'small_backtrack_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                        if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                            if availability_ok_after_pick(j, mb_est, seg_idx, remaining_mb - mb_est, segments_left - 1):
                                mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                                if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                                    earliest_pick = (j, cands[j].name, mb_real)
                                    break
                    except Exception:
                        continue
                if earliest_pick is not None:
                    j_back, name_back, mb_back = earliest_pick
                    if j_back < len(decisions) - 1:
                        del decisions[j_back:]
                        _cleanup_after_cut(j_back + 1)
                    prev_idx = j_back
                    remaining_mb = full_mb - sum(mb for _, mb in decisions) - mb_back
                    decisions.append((j_back, mb_back))
                    print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (本段回退-过小)')
                    log_write(f'Small-backtrack seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
                    if name_back != 'OUTPUT':
                        cut_layer_names.append(name_back)
                    seg_idx += 1
                    continue
                else:
                    print(f'当前段回退失败，无法为第{seg_idx}段找到可行回退点')
                    sys.exit(6)
            else:
                # 夹缝：根据当前平均 vs 初始平均选择策略
                print(f'第{seg_idx}段无窗口解（min<{LOW} < {HIGH}<max），按趋势选择策略')
                _print_failure_context(seg_idx, target, probes, decisions, remaining_mb, segments_left)
                
                if target < initial_avg:
                    # 前面段偏小 → 优先让上一段前移吸收（吃掉 low），当前段去吃 large
                    print(f'  target({target:.2f}) < initial_avg({initial_avg:.2f})，优先上一段吸收 low')
                    # 先尝试上一段前移
                    if len(decisions) > 0:
                        moved = try_shift_previous_to_absorb(len(decisions) - 1, decisions)
                        if moved:
                            prev_idx = decisions[-1][0]
                            remaining_mb = full_mb - sum(mb for _, mb in decisions)
                            probes = probe_from(prev_idx, seg_idx, target, segments_left)
                            if probes:
                                # 选择 large 候选中最早可行的
                                large_cands = [(i, name, mb) for (i, name, mb) in probes if mb > target and (mb > LOW) and (mb <= HIGH)]
                                if large_cands and availability_ok_after_pick(large_cands[0][0], large_cands[0][2], seg_idx, remaining_mb - large_cands[0][2], segments_left - 1):
                                    pick = large_cands[0]
                                else:
                                    # 无 large 可选，回到当前段自回退
                                    pick = None
                    if pick is None:
                        # 上一段无法前移或无 large 可选 → 当前段自回退
                        print('  上一段前移失败，当前段自回退')
                        m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
                        earliest_pick = None
                        for j in range(prev_idx - 1, -1, -1):
                            if j < 0:
                                break
                            try:
                                mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'sandwich_self_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                                if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                                    if availability_ok_after_pick(j, mb_est, seg_idx, remaining_mb - mb_est, segments_left - 1):
                                        mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                                        if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                                            earliest_pick = (j, cands[j].name, mb_real)
                                            break
                            except Exception:
                                continue
                        if earliest_pick is not None:
                            j_back, name_back, mb_back = earliest_pick
                            if j_back < len(decisions) - 1:
                                del decisions[j_back:]
                                _cleanup_after_cut(j_back + 1)
                            prev_idx = j_back
                            remaining_mb = full_mb - sum(mb for _, mb in decisions) - mb_back
                            decisions.append((j_back, mb_back))
                            print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (本段回退-夹缝小)')
                            log_write(f'Sandwich-self-small seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
                            if name_back != 'OUTPUT':
                                cut_layer_names.append(name_back)
                            seg_idx += 1
                            continue
                        else:
                            print(f'当前段自回退失败')
                            sys.exit(6)
                else:
                    # 前面段偏大 → 优先当前段回退覆盖"上一段+low"，把 large 留给下一段
                    print(f'  target({target:.2f}) >= initial_avg({initial_avg:.2f})，当前段回退覆盖上一段+low')
                    m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
                    earliest_pick = None
                    for j in range(prev_idx - 1, -1, -1):
                        if j < 0:
                            break
                        try:
                            mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'sandwich_back_seg{seg_idx}_{cands[j].name}', h, w, preprocess_fn)
                            if (mb_est is not None) and (mb_est > LOW) and (mb_est <= HIGH):
                                if availability_ok_after_pick(j, mb_est, seg_idx, remaining_mb - mb_est, segments_left - 1):
                                    mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
                                    if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                                        earliest_pick = (j, cands[j].name, mb_real)
                                        break
                        except Exception:
                            continue
                    if earliest_pick is not None:
                        j_back, name_back, mb_back = earliest_pick
                        if j_back < len(decisions) - 1:
                            del decisions[j_back:]
                            _cleanup_after_cut(j_back + 1)
                        prev_idx = j_back
                        remaining_mb = full_mb - sum(mb for _, mb in decisions) - mb_back
                        decisions.append((j_back, mb_back))
                        print(f'seg{seg_idx}: {name_back} = {mb_back:.2f}MB ✓ (本段回退-夹缝大)')
                        log_write(f'Sandwich-back-large seg{seg_idx}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
                        if name_back != 'OUTPUT':
                            cut_layer_names.append(name_back)
                        seg_idx += 1
                        continue
                    else:
                        print(f'当前段回退失败')
                        sys.exit(6)

        i, name, mb = pick
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        mb_fix, ok2 = finalize_segment(model, m_in, tensors[i + 1], seg_idx, h, w, preprocess_fn)
        if (not ok2) or (not ((mb_fix > LOW) and (mb_fix <= HIGH))):
            print('固化后大小异常，尝试替代候选...')
            # 在同批候选中尝试“更早 i”的其它可行点（满足预留），直到成功
            alts = []
            for (ii, nm, mbb) in valid_candidates:
                if ii == i:
                    continue
                if (mbb > LOW) and (mbb <= HIGH):
                    if (segments_left <= 1) or (((remaining_mb - mbb) / (segments_left - 1)) >= LOW):
                        if availability_ok_after_pick(ii, mbb, seg_idx, remaining_mb - mbb, segments_left - 1):
                            alts.append((ii, nm, mbb))
            alts.sort(key=lambda x: x[0])
            picked_alt = None
            for (ii, nm, mbb) in alts:
                mb_try, ok_try = finalize_segment(model, m_in, tensors[ii + 1], seg_idx, h, w, preprocess_fn)
                if ok_try and (mb_try > LOW) and (mb_try <= HIGH):
                    picked_alt = (ii, nm, mb_try)
                    break
            if picked_alt is None:
                # 触发回退：标记为 guard-only 失败，交由回退流程处理（本段优先）
                print('替代候选仍失败，进入回退流程')
                pick = None
                # 人为制造无可探测状态以走回退分支
                probes = []
                continue
            else:
                i, name, mb_fix = picked_alt

        prev_idx = i
        remaining_mb -= mb_fix
        print(f'seg{seg_idx}: {name} = {mb_fix:.2f}MB ✓')
        log_write(f'Finalize seg{seg_idx}: {name} = {mb_fix:.2f}MB\n' + '-' * 60 + '\n')
        if name != 'OUTPUT':
            cut_layer_names.append(name)
        # 记录本段决策，确保 segments_left 正确减少
        decisions.append((i, mb_fix))
        seg_idx += 1
        # 更新动态平均参考
        prev_avg = remaining_mb / (SEGMENTS - len(decisions)) if (SEGMENTS - len(decisions)) > 0 else None

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
