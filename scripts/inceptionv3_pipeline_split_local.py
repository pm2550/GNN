#!/usr/bin/env python3

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import contextlib

# 全局C++层stdout/stderr硬静音，供任意位置调用
@contextlib.contextmanager
def _suppress_cpp_io(silence_stdout=True, silence_stderr=True):
    null = os.open(os.devnull, os.O_RDWR)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        if silence_stdout:
            os.dup2(null, 1)
        if silence_stderr:
            os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(null)

# 提前静音，抑制TF C++ WARNING输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 禁用 oneDNN 和 XNNPACK 优化，避免 double free 错误
os.environ['TF_DISABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_XNNPACK'] = '1'

# 统一静音（减少无关日志）
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import tensorflow as tf
import numpy as np

# 全局配置
SEGMENTS = 8
LOW = 1.0
HIGH = 6.91

# 全局路径变量（在 main 中初始化）
OUT = None
TFL_DIR = None
TPU_DIR = None
LOG_DIR = None
LOG = None
BASE = None
MODEL_NAME = None

# 探测缓存
PROBE_CACHE = {}

# 过滤噪声日志的正则
DROP_PATTERNS = [
    r'Estimated count of arithmetic ops', r' MACs$',
    r'Ignored output_format', r'Ignored drop_control_dependency',
    r'NUMA node', r'Your kernel may have been built without NUMA support',
    r'Please consider providing the trackable_obj'
]
DROP_RE = re.compile('|'.join(DROP_PATTERNS))

def sanitize_log_lines(s: str) -> str:
    try:
        return '\n'.join(line for line in s.splitlines() if not DROP_RE.search(line))
    except Exception:
        return s

def sanitize_log(s: str) -> str:
    return re.sub(r'[^\x20-\x7E]', '?', s)

# 替换 PROBE_CACHE 键为稳定名称
# 代表性：在使用处直接构造稳定键

# pick_candidates 去重
def pick_candidates(model, model_name):
    candidates = []
    seen = set()
    for layer in model.layers:
        try:
            # 跳过输入层
            if type(layer).__name__ in ['InputLayer']:
                continue
            if id(layer) in seen:
                continue
            layer_name = layer.name
            layer_type = type(layer).__name__

            ok = False
            if model_name == 'InceptionV3':
                # InceptionV3：非最后段仅允许顶层 mixed* 作为切点；avg_pool 仅留给最后一段
                if layer_name.startswith('mixed'):
                    ok = True
                elif layer_name in ['avg_pool', 'global_average_pooling2d']:
                    ok = True
                else:
                    ok = False
            else:
                # 其它模型：Merge/Pool/Activation/BatchNorm等作一般候选（保持原策略，但去重）
                if any(k in layer_type for k in ['Pool','Pooling']):
                    ok = True
                elif layer_type in ['Add','Concatenate','Multiply','Average','Maximum','Minimum']:
                    ok = True
                else:
                    # 具有 3D/4D 输出的特征图也可考虑
                    if hasattr(layer, 'output') and getattr(layer, 'output', None) is not None:
                        shp = getattr(layer.output, 'shape', None)
                        if shp is not None and len(shp) >= 3:
                            ok = True
            if ok:
                candidates.append(layer)
                seen.add(id(layer))
        except Exception:
            continue
    print(f'找到 {len(candidates)} 个候选切点')
    return candidates

# chain_inference_check 优先CPU TFLite，再备选TPU
def chain_inference_check(base_model, preprocess_fn, input_size_hw, samples=5):
    h, w = input_size_hw
    test_inputs = []
    for _ in range(samples):
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        if preprocess_fn:
            img = preprocess_fn(img)
        test_inputs.append(img)
    test_batch = np.stack(test_inputs)
    original_outputs = base_model(test_batch).numpy()
    import re as _re
    def _collect(dir_path, pattern):
        files = list(dir_path.glob(pattern))
        mapping = {}
        for p in files:
            m = _re.search(r'seg(\d+)_', p.name)
            if not m:
                continue
            idx = int(m.group(1))
            mapping[idx] = p
        return mapping
    tfl_map = _collect(TFL_DIR, 'seg*_int8.tflite')
    tpu_map = _collect(TPU_DIR, 'seg*_int8_edgetpu.tflite')
    need_idxs = list(range(1, SEGMENTS + 1))
    seg_list = []
    if all(i in tfl_map for i in need_idxs):
        seg_list = [tfl_map[i] for i in need_idxs]
    elif all(i in tpu_map for i in need_idxs):
        seg_list = [tpu_map[i] for i in need_idxs]
    else:
        # 选择覆盖更多段的集合；若仍不完整则报错
        if len(tfl_map) >= len(tpu_map) and len(tfl_map) > 0:
            seg_list = [tfl_map[i] for i in sorted(tfl_map.keys())]
        elif len(tpu_map) > 0:
            seg_list = [tpu_map[i] for i in sorted(tpu_map.keys())]
        else:
            return {'error':'No segments found','num_samples':samples}
    current = test_batch
    for p in seg_list:
        intr = tf.lite.Interpreter(model_path=str(p))
        intr.allocate_tensors()
        inp = intr.get_input_details()[0]
        out = intr.get_output_details()[0]
        # 不做任何动态调整，严格要求shape/dtype一致
        if tuple(current.shape) != tuple(inp['shape']):
            return {'error': f'Input shape mismatch for {p.name}: need {tuple(inp["shape"])}, got {tuple(current.shape)}', 'num_samples': samples}
        if current.dtype != inp['dtype']:
            try:
                current = current.astype(inp['dtype'])
            except Exception:
                return {'error': f'Input dtype mismatch for {p.name}: need {inp["dtype"]}, got {current.dtype}', 'num_samples': samples}
        intr.set_tensor(inp['index'], current)
        intr.invoke()
        current = intr.get_tensor(out['index'])
    segmented_outputs = current
    if original_outputs.shape != segmented_outputs.shape:
        return {'error': f'Shape mismatch: original {original_outputs.shape} vs segmented {segmented_outputs.shape}', 'num_samples': samples}
    original_top1 = np.argmax(original_outputs, axis=1)
    segmented_top1 = np.argmax(segmented_outputs, axis=1)
    top1_match = int(np.sum(original_top1 == segmented_top1))
    avg_abs_diff = float(np.mean(np.abs(original_outputs - segmented_outputs)))
    return {'top1_match': top1_match, 'num_samples': samples, 'avg_abs_diff': avg_abs_diff, 'original_shape': list(original_outputs.shape), 'segmented_shape': list(segmented_outputs.shape)}

def ensure_dirs():
    """确保输出目录存在"""
    for d in [OUT, TFL_DIR, TPU_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def log_write(s: str):
    """写入日志文件"""
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(sanitize_log(s))

def _gen_random_imgs(N=128, h=299, w=299, preprocess_fn=None):
    """生成随机图像用于校准"""
    for _ in range(N):
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        if preprocess_fn:
            img = preprocess_fn(img)
        yield img

def rep_full_random(N=128, h=299, w=299, preprocess_fn=None):
    """生成随机代表性数据集（按TFLite要求：yield [array] 且含batch维）"""
    def gen():
        for img in _gen_random_imgs(N, h, w, preprocess_fn):
            # 添加batch维并包装为列表
            yield [np.expand_dims(img.astype(np.float32), axis=0)]
    return gen


def make_rep_from_prev(base_model, prev_tensor, N=128, h=299, w=299, preprocess_fn=None):
    """从前一个tensor生成代表性数据集（yield [array] 且含batch维）"""
    def gen():
        # 创建从输入到prev_tensor的子模型
        try:
            sub_model = tf.keras.Model(base_model.input, prev_tensor)
            for img in _gen_random_imgs(N, h, w, preprocess_fn):
                img_batch = np.expand_dims(img, axis=0).astype(np.float32)
                intermediate = sub_model(img_batch).numpy()
                # intermediate 形如 [1, ...]，直接作为列表元素返回
                yield [intermediate]
        except Exception as e:
            print(f'Warning: make_rep_from_prev failed: {e}')
            # 回退到随机数据（保留batch维）
            try:
                shape_wo_batch = prev_tensor.shape.as_list()[1:]
            except AttributeError:
                shape_wo_batch = list(prev_tensor.shape)[1:]
            for _ in range(N):
                rand_batch = np.random.randn(1, *shape_wo_batch).astype(np.float32)
                yield [rand_batch]
    return gen

def _build_converter_from_cf(model_sub, rep_fn, legacy=False):
    """构建TFLite转换器"""
    converter = None
    
    # 方法1：tf.function concrete_function（Keras 3 可靠路径）
    try:
        # 明确构建 tf.function 并获取 concrete function
        input_shape = tuple(int(x) if x is not None else 1 for x in model_sub.input.shape)
        fn = tf.function(model_sub)
        concrete_func = fn.get_concrete_function(tf.TensorSpec(input_shape, dtype=tf.float32))
        # 关键：传入 trackable_obj 以绑定变量到 Keras 3 Functional，避免 _get_save_spec 相关问题
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=model_sub)
    except Exception as e:
        print(f"  concrete function转换失败: {e}")
        
        # 方法2：SavedModel（Keras 3: 使用 export 导出 SavedModel）
        try:
            print(f"  尝试SavedModel(export)转换...")
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Keras 3 正确导出 SavedModel 的API
                if hasattr(model_sub, "export"):
                    model_sub.export(temp_dir)
                else:
                    # 兼容旧版本
                    model_sub.save(temp_dir)
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
            print(f"  SavedModel(export)转换成功")
        except Exception as e2:
            print(f"  SavedModel(export)转换失败: {e2}")
            
            # 方法3：Keras 模型直接转换（在某些版本下可用）
            try:
                print(f"  尝试Keras模型转换...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model_sub)
                print(f"  Keras模型转换成功")
            except Exception as e3:
                print(f"  Keras模型转换失败: {e3}")
                raise RuntimeError(f"所有TFLite转换方法都失败了: concrete_func={e}, saved_model={e2}, keras={e3}")
    
    if converter is None:
        raise RuntimeError("无法创建TFLite转换器")
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # 兼容性：避免部分算子放回 float，强制量化细化
    try:
        converter._experimental_disable_per_channel = False
    except Exception:
        pass
    
    if legacy:
        # 使用传统量化器
        try:
            converter._experimental_new_quantizer = False
        except AttributeError:
            pass  # 在某些版本中可能不支持
    
    return converter

def _analyze_int8_and_channels(tflite_bytes, expected_c):
    """分析TFLite模型的量化状态和通道数"""
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        has_float = False
        got_channels = None
        
        for detail in input_details:
            if detail['dtype'] != np.int8:
                has_float = True
            if len(detail['shape']) >= 4:  # NHWC格式
                got_channels = detail['shape'][-1]
        
        return True, has_float, got_channels
    except Exception:
        return False, True, None

def expected_in_channels(model_sub):
    """获取模型期望的输入通道数"""
    return model_sub.input.shape[-1] if len(model_sub.input.shape) >= 4 else None

def convert_with_auto_channel_guard(model_sub, rep_fn):
    """带自动通道守护的TFLite转换"""
    exp_c = expected_in_channels(model_sub)
    import contextlib, io, sys
    import warnings

    def _build_converter_by(method: str, legacy_flag: bool):
        # method: 'concrete' | 'saved' | 'keras'
        if method == 'concrete':
            # 与 _build_converter_from_cf 的方法1一致
            input_shape = tuple(int(x) if x is not None else 1 for x in model_sub.input.shape)
            fn = tf.function(model_sub, autograph=False)
            concrete_func = fn.get_concrete_function(tf.TensorSpec(input_shape, dtype=tf.float32))
            conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=model_sub)
        elif method == 'saved':
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                if hasattr(model_sub, 'export'):
                    model_sub.export(temp_dir)
                else:
                    model_sub.save(temp_dir)
                conv = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
        elif method == 'keras':
            conv = tf.lite.TFLiteConverter.from_keras_model(model_sub)
        else:
            raise ValueError('unknown method')
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_fn
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        try:
            conv._experimental_disable_per_channel = False
        except Exception:
            pass
        if legacy_flag:
            try:
                conv._experimental_new_quantizer = False
            except Exception:
                pass
        return conv

    def _convert_safely(legacy_flag: bool):
        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with contextlib.redirect_stderr(buf):
                with _suppress_cpp_io(True, True):
                    # 依次尝试 concrete → saved → keras，若 convert() 抛出 _get_save_spec 或其他异常则自动换路
                    methods = ['concrete', 'saved', 'keras']
                    last_err = None
                    for m in methods:
                        try:
                            conv = _build_converter_by(m, legacy_flag)
                            tfl = conv.convert()
                            break
                        except Exception as e:
                            last_err = e
                            # 针对 _get_save_spec 直接尝试下一路
                            continue
                    else:
                        raise last_err if last_err is not None else RuntimeError('convert failed')
        noise = buf.getvalue()
        if noise:
            log_write(sanitize_log_lines(noise) + '\n')
        return tfl

    # 首先尝试per-channel量化
    try:
        t0 = _convert_safely(False)
        ok, any_float, got_c = _analyze_int8_and_channels(t0, exp_c)
        if ok and not any_float:
            return t0, 'per_channel'
    except Exception as e:
        print(f"  per-channel量化失败: {e}")
        t0, any_float = None, True

    # 尝试传统量化器
    try:
        t1 = _convert_safely(True)
        ok2, any_float2, got_c2 = _analyze_int8_and_channels(t1, exp_c)
        if ok2 and not any_float2:
            return t1, 'legacy'
    except Exception as e:
        print(f"  legacy量化失败: {e}")
        t1, any_float2 = None, True

    if t0 is not None and not any_float:
        return t0, 'per_channel_partial'
    if t1 is not None and not any_float2:
        return t1, 'legacy_partial'
    if t0 is not None:
        return t0, 'fallback'
    if t1 is not None:
        return t1, 'fallback'
    raise RuntimeError("所有TFLite转换方法都失败了")

def compile_tpu(tfl_path: Path, tag: str):
    """编译EdgeTPU模型"""
    tpu_path = TPU_DIR / f'{tfl_path.stem}_edgetpu.tflite'
    
    cmd = f'edgetpu_compiler -s {tfl_path} -o {TPU_DIR}'
    log_write(f'Compiling {tag}: {cmd}\n')
    
    ret = os.system(cmd)
    if ret != 0:
        log_write(f'EdgeTPU compilation failed for {tag}\n')
        return None, False
    
    if not tpu_path.exists():
        log_write(f'EdgeTPU file not found: {tpu_path}\n')
        return None, False
    
    mb = tpu_path.stat().st_size / 1024 / 1024
    log_write(f'EdgeTPU compiled: {tpu_path.name} = {mb:.2f}MB\n')
    # 日志过滤写入
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            if os.path.exists(str(tfl_path)):
                f.write(f'[compile] {tag}: {tfl_path.name}\n')
    except Exception:
        pass
    return mb, True

def pick_candidates(model, model_name):
    """基于 legacy 规则的候选筛选（带去重）"""
    cands = []
    seen = set()
    import re as _re
    for l in model.layers:
        name = l.name
        out = getattr(l, 'output', None)
        if out is None:
            continue
        rank = len(getattr(out, 'shape', ()))
        if rank not in (2, 3, 4):
            continue
        # 通用候选：Add / Concatenate / 各类 Pool（含 GAP）/ keras 命名的 avg_pool
        is_add = isinstance(l, tf.keras.layers.Add)
        is_concat = isinstance(l, tf.keras.layers.Concatenate)
        is_pool = isinstance(l, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D, tf.keras.layers.GlobalAveragePooling2D)) or name == 'avg_pool'
        if is_add or is_concat or is_pool:
            if id(l) not in seen:
                cands.append(l)
                seen.add(id(l))
        # 模型特定锚点
        if model_name == 'InceptionV3':
            is_mixed = name.startswith('mixed') and ('_' not in name)
            is_avg_pool = (name == 'avg_pool')
            if is_mixed or is_avg_pool:
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name.startswith('ResNet'):
            if _re.match(r'conv[2-5]_block\d+_out$', name) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name == 'DenseNet201':
            if _re.match(r'pool[2-4]_pool$', name) or _re.match(r'conv\d+_block\d+_concat$', name) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
        elif model_name == 'Xception':
            # 1) 残差汇合/池化层 + avg_pool
            if isinstance(l, (tf.keras.layers.Add, tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)) or name == 'avg_pool':
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
                continue
            # 2) 尾段细分：block14_*_act
            if name.startswith('block14_') and name.endswith('_act'):
                if id(l) not in seen:
                    cands.append(l)
                    seen.add(id(l))
    print(f'找到 {len(cands)} 个候选切点')
    return cands

def probe_size(base_model, m_in, t_out, name_tag, h, w, preprocess_fn):
    global PROBE_CACHE
    key = (id(m_in), id(t_out))
    if key in PROBE_CACHE:
        return PROBE_CACHE[key]
    try:
        sub = tf.keras.Model(m_in, t_out)
        rep_fn = make_rep_from_prev(base_model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl, _ = convert_with_auto_channel_guard(sub, rep_fn)
        tfl_mb = len(tfl) / 1024 / 1024
        if tfl_mb > 6.0:
            temp_path = TFL_DIR / f'probe_{name_tag}_temp.tflite'
            temp_path.write_bytes(tfl)
            with _suppress_cpp_io(True, True):
                mb_tpu, ok_tpu = compile_tpu(temp_path, f'probe_{name_tag}')
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if ok_tpu and mb_tpu is not None:
                tfl_mb = mb_tpu
        PROBE_CACHE[key] = tfl_mb
        return tfl_mb
    except ValueError as e:
        if "`inputs` not connected to `outputs`" in str(e):
            result = _try_fallback_to_add_layer(base_model, m_in, t_out, name_tag, h, w, preprocess_fn)
            PROBE_CACHE[key] = result
            return result
        else:
            PROBE_CACHE[key] = None
            return None
    except Exception:
        PROBE_CACHE[key] = None
        return None

def _try_fallback_to_add_layer(base_model, m_in, t_out, name_tag, h, w, preprocess_fn):
    """当遇到跳跃/多输入合流问题时，尝试回退到最近的 Merge 层"""
    # 找到t_out对应的层
    t_out_layer = None
    for layer in base_model.layers:
        if hasattr(layer, 'output') and layer.output is t_out:
            t_out_layer = layer
            break
    
    if not t_out_layer:
        print(f'  {name_tag}: 无法找到输出层，回退失败')
        return None
    
    # 找到该层在模型中的索引
    t_out_layer_idx = None
    for i, layer in enumerate(base_model.layers):
        if layer == t_out_layer:
            t_out_layer_idx = i
            break
    
    if t_out_layer_idx is None:
        print(f'  {name_tag}: 无法找到层索引，回退失败')
        return None
    
    # 先尝试回退到"模型自身的合法切点"
    def is_model_anchor(layer_name: str) -> bool:
        try:
            if MODEL_NAME == 'Xception':
                return any(anchor in layer_name for anchor in ['block14_sepconv1_act', 'block14_sepconv2_act'])
            elif MODEL_NAME == 'InceptionV3':
                return layer_name.startswith('mixed')
            return False
        except Exception:
            return False
    
    # 优先尝试模型锚点
    for i in range(t_out_layer_idx - 1, -1, -1):
        layer = base_model.layers[i]
        if is_model_anchor(layer.name):
            try:
                fallback_sub = tf.keras.Model(m_in, layer.output)
                rep_fn = make_rep_from_prev(base_model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
                tfl, mode = convert_with_auto_channel_guard(fallback_sub, rep_fn)
                tfl_mb = len(tfl) / 1024 / 1024
                print(f'  {name_tag}: 回退到锚点 {layer.name}，大小={tfl_mb:.2f}MB')
                return tfl_mb
            except Exception:
                continue
    
    # 模型特定的回退策略
    merge_types = (
        ['Add'] if MODEL_NAME == 'Xception' 
        else ['Add', 'Concatenate', 'Multiply', 'Average', 'Maximum', 'Minimum']
    )
    
    # 向前寻找合适的回退点
    for i in range(t_out_layer_idx - 1, -1, -1):
        layer = base_model.layers[i]
        if type(layer).__name__ in merge_types:
            try:
                fallback_sub = tf.keras.Model(m_in, layer.output)
                rep_fn = make_rep_from_prev(base_model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
                tfl, mode = convert_with_auto_channel_guard(fallback_sub, rep_fn)
                tfl_mb = len(tfl) / 1024 / 1024
                print(f'  {name_tag}: 回退到 {layer.name}，大小={tfl_mb:.2f}MB')
                return tfl_mb
            except Exception as e:
                continue
    
    print(f'  {name_tag}: 无法找到合适的 Merge 回退点')
    return None

def finalize_segment(base_model, m_in, t_out, seg_idx, h, w, preprocess_fn):
    """固化模型段"""
    try:
        sub = tf.keras.Model(m_in, t_out)
        rep_fn = make_rep_from_prev(base_model, m_in, N=32, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl, _ = convert_with_auto_channel_guard(sub, rep_fn)
        
        tfl_path = TFL_DIR / f'seg{seg_idx}_int8.tflite'
        tfl_path.write_bytes(tfl)
        
        mb, ok = compile_tpu(tfl_path, f'seg{seg_idx}')
        return mb, ok
    except Exception:
        # 如果出错，也尝试回退到最近的 Merge 层
        return _try_fallback_to_add_layer(base_model, m_in, t_out, f'seg{seg_idx}', h, w, preprocess_fn), False

def _estimate_tflite_size_mb(model, m_in, t_out, name_tag, h, w, preprocess_fn):
    """估算TFLite模型大小（MB）"""
    try:
        sub = tf.keras.Model(m_in, t_out)
        rep_fn = make_rep_from_prev(model, m_in, N=16, h=h, w=w, preprocess_fn=preprocess_fn)
        tfl, _ = convert_with_auto_channel_guard(sub, rep_fn)
        return len(tfl) / 1024 / 1024
    except Exception:
        # 估算路径同样尝试回退到最近的 Merge 层，避免前向扫描全军覆没
        fb = _try_fallback_to_add_layer(model, m_in, t_out, f'est_{name_tag}', h, w, preprocess_fn)
        return fb

def _print_failure_context(seg_idx_local: int, target_local: float, probes_list, decisions_list, remaining_local: float, segments_left_local: int, cands):
    """诊断日志：打印当前已决策段与候选分类"""
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

def _estimate_last_mb(prev_idx_local: int, tensors, model, h, w, preprocess_fn) -> float | None:
    """估算从指定点到OUTPUT的最后一段大小"""
    m_in_local = tensors[0] if prev_idx_local < 0 else tensors[prev_idx_local + 1]
    try:
        est = _estimate_tflite_size_mb(model, m_in_local, tensors[-1], 'avail_last', h, w, preprocess_fn)
        return est
    except Exception:
        return None

    

def invincible_backtrack_current_segment(seg_idx, prev_idx, tensors, cands, model, h, w, preprocess_fn, remaining_mb, segments_left):
    """无敌回退：当前段寻找最早可行点"""
    m_in_cur = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
    backtrack_pick = None
    print(f'  seg{seg_idx} 无敌回退：寻找最早可行点...')
    # 从上一段切点之后向前扫描，寻找最早可行终点
    for j in range(prev_idx + 1, len(cands)):
        try:
            name_j = cands[j].name
            mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'invincible_backtrack_seg{seg_idx}_{name_j}', h, w, preprocess_fn)
            if mb_est is None:
                continue
            print(f'    回退试探: {name_j} est={mb_est:.2f}MB')
            if mb_est > HIGH:
                print(f'    回退早停: {name_j} = {mb_est:.2f}MB > {HIGH}MB')
                break
            if mb_est <= LOW:
                continue
            # 回退阶段仅要求窗内（>LOW 且 <=HIGH），预留由随后段重建承担
            mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], seg_idx, h, w, preprocess_fn)
            if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                backtrack_pick = (j, name_j, mb_real)
                print(f'    找到可行回退点: {name_j} = {mb_real:.2f}MB')
                break
        except Exception:
            continue
    
    return backtrack_pick

def _cleanup_after_cut(start_seg_idx_inclusive: int):
    """在回退导致截断后，清理被截断段位之后的旧文件"""
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

def greedy_with_backtrack(model, model_name, preprocess_fn, input_size_hw):
    """带无敌回退的贪心分段算法"""
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

    # 这里不再编译整模 TPU，仅用 TFLite 大小估算
    full_mb = len(full_tfl) / 1024 / 1024
    ok = True
    print(f'FULL TFLite: {full_mb:.2f}MB')
    log_write(f'FULL TFLITE SIZE: {full_mb:.2f}MB\n' + '-' * 60 + '\n')

    # 若整模 TFLite > 6MB，则补编一次 TPU 获取更贴近部署的大小
    if full_mb > 6.0:
        mb_tpu, ok_tpu = compile_tpu(full_tfl_path, 'full_model')
        if ok_tpu and mb_tpu is not None:
            full_mb = mb_tpu
            print(f'FULL TPU: {full_mb:.2f}MB')
            log_write(f'FULL TPU SIZE: {full_mb:.2f}MB\n' + '-' * 60 + '\n')

    cands = pick_candidates(model, model_name)
    assert len(cands) >= SEGMENTS - 1, '候选数量不足以切成8段'

    tensors = [model.input] + [l.output for l in cands] + [model.output]

    def probe_from(prev_idx, seg_idx, target, segments_left, cands, tensors, model, h, w, preprocess_fn):
        """逐探逐判（收集窗内）：
        - 仅在当前段内顺序探测；
        - 打印每个候选的大小；
        - 收集所有 LOW<mb<=HIGH 的候选；
        - 遇到首个 >HIGH 立即早停；
        - 返回收集到的窗内候选列表，供主循环按动态平均与侧偏挑选。
        """
        m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
        in_window = []
        for i in range(prev_idx + 1, len(cands)):
            name = cands[i].name
            # InceptionV3 非最后段只探顶层 mixed*（不含下划线），避免跨分支导致多输入构图问题
            if (model_name == 'InceptionV3') and (segments_left > 1):
                if not (name.startswith('mixed') and ('_' not in name)):
                    continue
            _flush_print(f'  探测: seg{seg_idx} -> {name}')
            t_out = tensors[i + 1]
            mb = probe_size(model, m_in, t_out, f'seg{seg_idx}_{name}', h, w, preprocess_fn)
            if mb is None:
                continue
            _flush_print(f'    大小: {name} = {mb:.2f}MB')
            if mb > HIGH:
                _flush_print(f'  早停: {name} = {mb:.2f}MB > {HIGH}MB')
                break
            if mb <= LOW:
                continue
            # 窗内：收集
            in_window.append((i, name, mb))
        return in_window

    def availability_ok_after_pick(i_pick: int, mb_pick: float, seg_idx_now: int, remaining_after_pick: float, segments_left_now: int, cands, tensors, model, h, w, preprocess_fn) -> bool:
        """可用性护栏（局部版）：依赖本地 probe_from，确保能形成剩余 n-1 段且末段接 OUTPUT。
        segments_left_now: 当前段选定后的剩余段数。
        """
        # 剩余≤1：无需预留
        if segments_left_now <= 1:
            return True
        # 剩余2段：seg7+seg8，seg7需>LOW且 seg8 接 OUTPUT>LOW
        if segments_left_now == 2:
            prev_idx_next = i_pick
            target_next = remaining_after_pick / 2
            next_probes = probe_from(prev_idx_next, seg_idx_now + 1, target_next, 2, cands, tensors, model, h, w, preprocess_fn)
            if not next_probes:
                return False
            seg7_valid = [(j, nm, mb) for (j, nm, mb) in next_probes if (nm != 'OUTPUT' and (mb > LOW) and (mb <= HIGH))]
            if not seg7_valid:
                return False
            for (j, nm, mbj) in seg7_valid:
                # seg8 直接 OUTPUT 的大小（估算）
                m_in_local = tensors[0] if j < 0 else tensors[j + 1]
                try:
                    mb8 = _estimate_tflite_size_mb(model, m_in_local, tensors[-1], 'avail_last', h, w, preprocess_fn)
                    if mb8 and mb8 > LOW:
                        return True
                except Exception:
                    continue
            return False
        # 剩余≥3段：递归检查
        prev_idx_next = i_pick
        target_next = remaining_after_pick / segments_left_now
        next_probes = probe_from(prev_idx_next, seg_idx_now + 1, target_next, segments_left_now, cands, tensors, model, h, w, preprocess_fn)
        if not next_probes:
            return False
        next_valid = [(j, nm, mb) for (j, nm, mb) in next_probes if (nm != 'OUTPUT' and (mb > LOW) and (mb <= HIGH))]
        if not next_valid:
            return False
        for (j, nm, mbj) in next_valid:
            rem_after_next = remaining_after_pick - mbj
            if availability_ok_after_pick(j, mbj, seg_idx_now + 1, rem_after_next, segments_left_now - 1, cands, tensors, model, h, w, preprocess_fn):
                return True
        return False

    # 动态平均趋势跟踪：用于选择侧偏（防止连续下滑后尾段超限）
    def apply_side_bias(cands):
        """应用动态侧偏策略"""
        if not cands or prev_avg is None:
            return cands
        
        current_avg = remaining_mb / (SEGMENTS - len(decisions)) if (SEGMENTS - len(decisions)) > 0 else target
        
        # 只考虑最后一次趋势
        if current_avg < prev_avg:  # 下降趋势
            trend_dir = 'down'
        elif current_avg > prev_avg:  # 上升趋势
            trend_dir = 'up'
        else:
            trend_dir = None
        
        if trend_dir == 'down':
            # 下降趋势：优先选择较大的候选
            side = [(i, name, mb) for (i, name, mb) in cands if mb >= target]
            return side if side else cands
        elif trend_dir == 'up':
            # 上升趋势：优先选择较小的候选
            side = [(i, name, mb) for (i, name, mb) in cands if mb <= target]
            return side if side else cands
        return cands

    # 主循环变量初始化
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
    
    # ===== 主循环：无敌回退贪心算法 =====
    while seg_idx <= SEGMENTS:
        segments_left = SEGMENTS - seg_idx + 1  # 包含当前段的剩余段数
        target = remaining_mb / segments_left
        
        _flush_print(f'\n=== seg{seg_idx}: target={target:.2f}MB, remaining={remaining_mb:.2f}MB, segments_left={segments_left} ===')
        
        # 最后一段：直接连接到OUTPUT
        if segments_left == 1:
            m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            mb, ok2 = finalize_segment(model, m_in, tensors[-1], seg_idx, h, w, preprocess_fn)
            if not ok2:
                print('尾段编译失败')
                sys.exit(2)
            if not (LOW <= mb <= HIGH):
                print(f'尾段 {mb:.2f}MB 不合规，需要回退调整前面的段')
                sys.exit(3)
            decisions.append(('OUTPUT', mb))
            print(f'seg{seg_idx}: OUTPUT = {mb:.2f}MB ✓')
            log_write(f'Finalize seg{seg_idx}: OUTPUT = {mb:.2f}MB\n' + '-' * 60 + '\n')
            break

        # 探测候选切点
        probes = probe_from(prev_idx, seg_idx, target, segments_left, cands, tensors, model, h, w, preprocess_fn)
        
        # 过滤可行候选（仅判窗内，不做护栏前置筛选）
        feasible_candidates = []
        for (i, name, mb) in probes:
            if (mb > LOW) and (mb <= HIGH):
                feasible_candidates.append((i, name, mb))

        # 分类诊断（避免 NameError）
        any_below = any(mb < LOW for (_i, _nm, mb) in probes) if probes else False
        any_above = any(mb > HIGH for (_i, _nm, mb) in probes) if probes else False
        all_small = bool(probes) and all(mb <= LOW for (_i, _nm, mb) in probes)
        all_large = bool(probes) and all(mb > HIGH for (_i, _nm, mb) in probes)
        sandwich = bool(probes) and (not feasible_candidates) and (not all_small) and (not all_large)

        # 诊断：窗内候选简表（最多10个，按与target的接近程度）
        try:
            if feasible_candidates:
                _flush_print(f'  窗内候选数量: {len(feasible_candidates)}')
                tops = sorted(feasible_candidates, key=lambda x: abs(x[2] - target))[:10]
                print('  窗内候选(前10按|mb-target|):')
                for (ii, nm, mb) in tops:
                    dist = (ii - prev_idx) if ii < len(cands) else (len(cands) - prev_idx)
                    avg_rest_tmp = ((remaining_mb - mb) / (segments_left - 1)) if segments_left > 1 else 0.0
                    print(f'    - {nm}: mb={mb:.2f}MB, dist={dist}, avg_rest={avg_rest_tmp:.2f}')
        except Exception:
            pass

        # 处理各种失败情况，启用无敌回退
        pick = None

        # 情况1：有窗内候选，先正常选取
        if feasible_candidates:
            # 模型特定的锚点优先策略
            if model_name == 'Xception' and segments_left <= 3:
                anchor_names = {'block14_sepconv1_act', 'block14_sepconv2_act'}
                anchor_candidates = [(i, name, mb) for (i, name, mb) in feasible_candidates if name in anchor_names]
                anchor_candidates = apply_side_bias(anchor_candidates)
                if anchor_candidates:
                    pick = min(anchor_candidates, key=lambda x: abs(x[2] - target))
                else:
                    cands2 = apply_side_bias(feasible_candidates)
                    pick = min(cands2, key=lambda x: abs(x[2] - target))
            else:
                cands2 = apply_side_bias(feasible_candidates)
                pick = min(cands2, key=lambda x: abs(x[2] - target))
            # 选中后仅检查余均值护栏；可用性护栏仅用于回退阶段
            if pick is not None:
                i_chk, nm_chk, mb_chk = pick
                if segments_left > 1:
                    avg_rest = (remaining_mb - mb_chk) / (segments_left - 1)
                    if avg_rest < LOW:
                        _flush_print(f'  余均值护栏触发: avg_rest={avg_rest:.2f} < LOW({LOW})')
                        pick = None
                if pick is not None:
                    print(f'  选中候选: {nm_chk} est={mb_chk:.2f}MB, target={target:.2f}MB')

        # 情况2：无候选/全小/夹缝/过大 或护栏不通过 → 无敌回退
        if not pick:
            failure_reason = '护栏触发'
            print(f'第{seg_idx}段{failure_reason}，启动无敌回退')
            _print_failure_context(seg_idx, target, probes, decisions, remaining_mb, segments_left, cands)
            # 避免“原地回退”：当前段若已有窗内候选但护栏不通过，则回退时跳过这些候选
            skip_names = set(nm for (_ii, nm, _mb) in feasible_candidates)

            def cascade_backtrack(cur_seg: int) -> tuple | None:
                """从当前段向前级联无敌回退；返回 (seg_idx_fixed, j_back, name_back, mb_back) 或 None"""
                nonlocal decisions, remaining_mb
                # 计算当前段的 prev_idx_k 与剩余预算
                if cur_seg == 1:
                    prev_idx_k = -1
                    used_mb = 0.0
                else:
                    prev_idx_k = decisions[cur_seg - 2][0]
                    used_mb = sum(mb for (_ci, mb) in decisions[:cur_seg-1])
                remaining_k = full_mb - used_mb
                segments_left_k = SEGMENTS - cur_seg + 1
                # 当前段先尝试“当前段优先”的无敌回退
                m_in_cur = tensors[0] if prev_idx_k < 0 else tensors[prev_idx_k + 1]
                back = None
                # 决定扫描方向：已固化段从当前终点向前回扫；未固化段从上一段末端向后试探
                if cur_seg <= len(decisions):
                    end_idx_cur = decisions[cur_seg - 1][0]
                    j_iter = range(end_idx_cur - 1, prev_idx_k, -1)
                else:
                    j_iter = range(prev_idx_k + 1, len(cands))
                for j in j_iter:
                    try:
                        l_j = cands[j]
                        name_j = l_j.name
                        # 跳过导致护栏触发但未通过的当前段窗内候选，避免“原地不动”
                        if cur_seg == seg_idx and name_j in skip_names:
                            continue
                        # 模型特定过滤：与探测阶段一致
                        if model_name == 'InceptionV3' and (segments_left_k > 1):
                            is_top_mixed = name_j.startswith('mixed') and ('_' not in name_j)
                            if not is_top_mixed:
                                continue
                        if model_name == 'Xception' and (cur_seg >= 5):
                            if isinstance(l_j, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)) or name_j == 'avg_pool':
                                continue
                        mb_est = _estimate_tflite_size_mb(model, m_in_cur, tensors[j + 1], f'bt_seg{cur_seg}_{name_j}', h, w, preprocess_fn)
                        if mb_est is None:
                            continue
                        print(f'  回退试探(seg{cur_seg}): {name_j} est={mb_est:.2f}MB')
                        if mb_est > HIGH:
                            print(f'  回退早停(seg{cur_seg}): {name_j} = {mb_est:.2f}MB > {HIGH}MB')
                            break
                        if mb_est <= LOW:
                            continue
                        # 仅对“触发回退的最靠尾段”执行可用性护栏；其余更靠前的级联回退不再校验尾部可行性
                        if cur_seg == seg_idx:
                            rem_after = remaining_k - mb_est
                            seg_left_after = segments_left_k - 1
                            guard_ok = availability_ok_after_pick(
                                j, mb_est, cur_seg, rem_after, seg_left_after,
                                cands, tensors, model, h, w, preprocess_fn
                            )
                            if not guard_ok:
                                print(f'  回退候选不满足后续可用性: {name_j} est={mb_est:.2f}MB')
                                continue
                        # 通过（或非触发段）后直接固化验证
                        mb_real, ok_real = finalize_segment(model, m_in_cur, tensors[j + 1], cur_seg, h, w, preprocess_fn)
                        if ok_real and (mb_real > LOW) and (mb_real <= HIGH):
                            back = (cur_seg, j, name_j, mb_real)
                            break
                    except Exception:
                        continue
                if back is not None:
                    return back
                # 当前段无法回退，则推进到上一段
                if cur_seg <= 1:
                    return None
                return cascade_backtrack(cur_seg - 1)

            # 记录原始触发段位（最靠尾的触发者）
            orig_fail_seg = seg_idx
            res = cascade_backtrack(seg_idx)
            if res is None:
                # 只有 seg1 无法保持 >LOW 才宣告失败
                _flush_print('无敌回退失败：seg1 无法满足 >LOW')
                sys.exit(7)
            seg_fixed, j_back, name_back, mb_back = res
            # 截断被覆盖段并清理，从 seg_fixed 起全部重建
            cut_from = j_back + 1
            # 删除 seg_fixed 及之后的决策
            del decisions[seg_fixed-1:]
            # 注意：不要删除刚刚回退固化成功的 seg_fixed 文件，只清理其后的段
            _cleanup_after_cut(seg_fixed + 1)
            # 记录当前固化
            decisions.append((j_back, mb_back))
            print(f'seg{seg_fixed}: {name_back} = {mb_back:.2f}MB ✓ (无敌回退)')
            log_write(f'Invincible-backtrack seg{seg_fixed}: {name_back} = {mb_back:.2f}MB\n' + '-' * 60 + '\n')
            if name_back != 'OUTPUT':
                cut_layer_names.append(name_back)
            # 重新计算预算
            remaining_mb = full_mb - sum(mb for (_ci, mb) in decisions)
            # 预留计划：为 seg_fixed+1..orig_fail_seg 选定“最早可行”的窗内候选索引，避免回到回退前的选择
            reserved_plan = {}
            k_plan = seg_fixed + 1
            while k_plan <= orig_fail_seg:
                segs_left_kp = SEGMENTS - k_plan + 1
                # 安全取得上一段切点：优先使用 decisions[k_plan-2]，不足则用最后一个已决策切点
                if (k_plan - 2) >= 0 and len(decisions) >= (k_plan - 1):
                    prev_idx_kp = decisions[k_plan - 2][0]
                else:
                    prev_idx_kp = decisions[-1][0] if decisions else -1
                m_in_kp = tensors[0] if prev_idx_kp < 0 else tensors[prev_idx_kp + 1]
                inwin_kp = []
                for ii in range(prev_idx_kp + 1, len(cands)):
                    name2 = cands[ii].name
                    if (model_name == 'InceptionV3') and (segs_left_kp > 1):
                        if not (name2.startswith('mixed') and ('_' not in name2)):
                            continue
                    mb2 = probe_size(model, m_in_kp, tensors[ii + 1], f'reserve_seg{k_plan}_{name2}', h, w, preprocess_fn)
                    if mb2 is None:
                        continue
                    if mb2 > HIGH:
                        break
                    if mb2 > LOW:
                        inwin_kp.append((ii, name2, mb2))
                if inwin_kp:
                    # 选择“最早”窗内候选作为预留
                    ii_pick_kp, _nm_kp, _mb_kp = min(inwin_kp, key=lambda x: x[0])
                    reserved_plan[k_plan] = ii_pick_kp
                k_plan += 1
            # 立即“全固化”：顺次重建 seg_fixed+1 .. orig_fail_seg（优先使用预留计划；仅窗内，不再做尾部可用性护栏）
            k = seg_fixed + 1
            while k <= orig_fail_seg:
                segs_left_k = SEGMENTS - k + 1
                target_k = remaining_mb / segs_left_k
                # 同步取得上一段切点（若此时 decisions 尚未填满到 k-1，回退到当前最后一个）
                if (k - 2) >= 0 and len(decisions) >= (k - 1):
                    prev_idx_k = decisions[k - 2][0]
                else:
                    prev_idx_k = decisions[-1][0] if decisions else -1
                m_in_k = tensors[0] if prev_idx_k < 0 else tensors[prev_idx_k + 1]
                # 探测窗内候选，遇>HIGH早停
                inwin_k = []
                for ii in range(prev_idx_k + 1, len(cands)):
                    name2 = cands[ii].name
                    if (model_name == 'InceptionV3') and (segs_left_k > 1):
                        if not (name2.startswith('mixed') and ('_' not in name2)):
                            continue
                    mb2 = probe_size(model, m_in_k, tensors[ii + 1], f'cascade_seg{k}_{name2}', h, w, preprocess_fn)
                    if mb2 is None:
                        continue
                    if mb2 > HIGH:
                        break
                    if mb2 > LOW:
                        inwin_k.append((ii, name2, mb2))
                if not inwin_k:
                    # 若无窗内候选，宣告失败（理论上不应发生）
                    _flush_print(f'  级联重建 seg{k} 无窗内候选')
                    sys.exit(9)
                # 优先使用预留索引；否则选择“最早”窗内候选（移动头部，避免回到旧切点）
                if k in reserved_plan:
                    match = [x for x in inwin_k if x[0] == reserved_plan[k]]
                    if match:
                        ii_pick, name_pick, mb_pick = match[0]
                    else:
                        ii_pick, name_pick, mb_pick = min(inwin_k, key=lambda x: x[0])
                else:
                    ii_pick, name_pick, mb_pick = min(inwin_k, key=lambda x: x[0])
                mb_fix_k, ok_fix_k = finalize_segment(model, m_in_k, tensors[ii_pick + 1], k, h, w, preprocess_fn)
                if not ok_fix_k or not (LOW <= mb_fix_k <= HIGH):
                    _flush_print(f'  级联重建 seg{k} 固化失败或越界')
                    sys.exit(10)
                decisions.append((ii_pick, mb_fix_k))
                if name_pick != 'OUTPUT':
                    cut_layer_names.append(name_pick)
                remaining_mb = full_mb - sum(mb for (_ci, mb) in decisions)
                print(f'seg{k}: {name_pick} = {mb_fix_k:.2f}MB ✓ (级联)')
                log_write(f'Cascade finalize seg{k}: {name_pick} = {mb_fix_k:.2f}MB\n' + '-' * 60 + '\n')
                k += 1
            # 回退后盈余平分（提示）：相邻段可尝试单向均衡大小（未自动执行）
            print('  回退后盈余平分（提示）：相邻段可尝试单向均衡大小（未自动执行）')
            # 从原始触发段之后继续
            seg_idx = orig_fail_seg + 1
            prev_idx = decisions[-1][0]
            continue
        
        # 正常情况：固化选中的候选
        if pick:
            i, name, mb_est = pick
            print(f'  固化前估算: {name} est={mb_est:.2f}MB')
            m_in = tensors[0] if prev_idx < 0 else tensors[prev_idx + 1]
            mb_fix, ok_fix = finalize_segment(model, m_in, tensors[i + 1], seg_idx, h, w, preprocess_fn)
            if ok_fix:
                _flush_print(f'  固化完成: {name} real={mb_fix:.2f}MB')
            else:
                _flush_print(f'  固化失败: {name}')
        
        # 如果固化失败，尝试其他候选
        if not ok_fix:
            valid_alts = [(ii, nn, mm) for (ii, nn, mm) in feasible_candidates if (ii, nn, mm) != pick]
            valid_alts.sort(key=lambda x: x[0])  # 按索引排序
            picked_alt = None
            for alt in valid_alts:
                ii, nn, mm = alt
                mb_alt, ok_alt = finalize_segment(model, m_in, tensors[ii + 1], seg_idx, h, w, preprocess_fn)
                if ok_alt:
                    picked_alt = (ii, nn, mb_alt)
                    break
            if picked_alt:
                i, name, mb_fix = picked_alt
            else:
                print(f'seg{seg_idx}: 所有候选固化失败，启动无敌回退')
                # 触发无敌回退
                continue
        
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

    # 结束主循环，生成总结
    tpu_files = sorted(TPU_DIR.glob('seg*_int8_edgetpu.tflite'))
    ok_cnt = 0
    for f in tpu_files:
        mb = f.stat().st_size / 1024 / 1024
        ok = LOW <= mb <= HIGH
        ok_cnt += ok
        print(f'  {f.name}: {mb:.2f}MB {"✓" if ok else "✗"}')
    print(f'✓ {ok_cnt}/{len(tpu_files)} 段合规')
    
    summary = {
        'model_name': model_name,
        'segments': SEGMENTS,
        'size_range': [LOW, HIGH],
        'total_mb': full_mb,
        'cut_points': cut_layer_names,
        'decisions': [(name if isinstance(idx, str) else cands[idx].name, mb) for (idx, mb) in decisions],
        'compliant_segments': ok_cnt,
        'total_segments': len(tpu_files)
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    try:
        check = chain_inference_check(model, preprocess_fn, (h, w), samples=1)
        (OUT / 'chain_check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
        if 'error' in check:
            print('chain_check error:', check['error'])
        else:
            print(f"chain_check: top1_match={check['top1_match']}/{check['num_samples']}, avg_abs_diff={check['avg_abs_diff']:.6f}")
    except Exception as e:
        print('chain_check failed:', e)

# 轻量实时进度打印
import sys as _sys

def _flush_print(msg: str):
    try:
        print(msg)
        _sys.stdout.flush()
    except Exception:
        pass

def main():
    # 先声明 global，避免在本函数中"声明前使用"
    global SEGMENTS, LOW, HIGH, OUT, TFL_DIR, TPU_DIR, LOG_DIR, LOG, BASE, MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', type=int, default=SEGMENTS)
    parser.add_argument('--low', type=float, default=LOW)
    parser.add_argument('--high', type=float, default=HIGH)
    parser.add_argument('--model', type=str, choices=['InceptionV3', 'ResNet101', 'ResNet50', 'DenseNet201', 'Xception'], required=True)
    
    args = parser.parse_args()
    SEGMENTS = args.segments
    LOW = args.low
    HIGH = args.high
    MODEL_NAME = args.model
    
    # Model constructors
    constructors = {
        'InceptionV3': tf.keras.applications.InceptionV3,
        'ResNet101': tf.keras.applications.ResNet101,
        'ResNet50': tf.keras.applications.ResNet50,
        'DenseNet201': tf.keras.applications.DenseNet201,
        'Xception': tf.keras.applications.Xception
    }
    
    preprocess_fns = {
        'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
        'ResNet101': tf.keras.applications.resnet.preprocess_input,
        'ResNet50': tf.keras.applications.resnet.preprocess_input,
        'DenseNet201': tf.keras.applications.densenet.preprocess_input,
        'Xception': tf.keras.applications.xception.preprocess_input
    }
    
    input_sizes = {
        'InceptionV3': (299, 299),
        'ResNet101': (224, 224),
        'ResNet50': (224, 224),
        'DenseNet201': (224, 224),
        'Xception': (299, 299)
    }
    
    ctor = constructors[MODEL_NAME]
    preprocess_fn = preprocess_fns[MODEL_NAME]
    h, w = input_sizes[MODEL_NAME]
    
    ROOT = Path(__file__).parent.parent
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
