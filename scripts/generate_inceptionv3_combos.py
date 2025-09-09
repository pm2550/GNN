#!/usr/bin/env python3
# 依据固定的 8 段切点，按 pipeline.md 生成 k=2..7 的“前缀组合”，仅相邻合并，不重切。

import argparse
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time
import os

import tensorflow as tf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'models_local' / 'public' / 'inceptionv3_8seg_uniform_local'
COMPILER = str(ROOT / 'edgetpu_compiler')

def load_cut_names_from_summary(base_dir: Path) -> list[str]:
    """从最新分段结果 summary.json 读取切点名称列表。
    兼容 'cut_points' 或基于 'decisions' 推导。
    失败则返回空列表。
    """
    try:
        summary_path = base_dir / 'full_split_pipeline_local' / 'summary.json'
        if not summary_path.exists():
            return []
        data = json.loads(summary_path.read_text(encoding='utf-8'))
        if 'cut_points' in data and isinstance(data['cut_points'], list) and data['cut_points']:
            return list(data['cut_points'])
        if 'decisions' in data and isinstance(data['decisions'], list):
            names = []
            for item in data['decisions']:
                # item 形如 [name, mb] 或 [idx->name, mb]
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    nm = item[0]
                    if isinstance(nm, str) and nm != 'OUTPUT':
                        names.append(nm)
            return names
    except Exception:
        return []
    return []

def ensure_dirs(out_root: Path):
    (out_root / 'tflite').mkdir(parents=True, exist_ok=True)
    (out_root / 'tpu').mkdir(parents=True, exist_ok=True)
    (out_root / 'logs').mkdir(parents=True, exist_ok=True)

def build_segments(model, cut_names):
    segs = []
    prev = model.input
    for name in cut_names:
        out = model.get_layer(name).output
        segs.append((prev, out))
        prev = out
    segs.append((prev, model.output))
    return segs

def to_bytes_mib(n):
    return f"{n/(1024*1024):.2f}MB"

def convert_tflite_int8(model_sub, rep_fn):
    # 使用 concrete function，避免 Keras3 _get_save_spec 问题
    ish = model_sub.input.shape[1:]
    spec = tf.TensorSpec([1, *ish], tf.float32, name='input')
    @tf.function
    def f(x):
        return model_sub(x)
    cf = f.get_concrete_function(spec)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], trackable_obj=model_sub)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv._experimental_disable_per_channel = True
    if hasattr(conv, "_experimental_new_quantizer"):
        conv._experimental_new_quantizer = False
    conv.representative_dataset = rep_fn
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()

def compile_tpu(tfl_path: Path, out_dir: Path, tag: str):
    logs_dir = out_dir / 'logs'
    tpu_dir = out_dir / 'tpu'
    tpu_dir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([COMPILER, '-o', str(tpu_dir), str(tfl_path)], capture_output=True, text=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    with open(logs_dir / f'{ts}_{tag}.log', 'w', encoding='utf-8') as f:
        if res.stdout:
            f.write('STDOUT\n'); f.write(res.stdout); f.write('\n')
        if res.stderr:
            f.write('STDERR\n'); f.write(res.stderr); f.write('\n')
    edgetpu_path = tpu_dir / (tfl_path.stem + '_edgetpu.tflite')
    ok = edgetpu_path.exists()
    size_mb = os.path.getsize(edgetpu_path)/(1024*1024) if ok else None
    return ok, size_mb

def make_rep_from_prev(base_model, prev_tensor, samples=32):
    """使用原模型子图产生与 prev_tensor 形状一致的中间激活，供代表性数据使用。"""
    h = int(base_model.input.shape[1])
    w = int(base_model.input.shape[2])
    preprocess = tf.keras.applications.inception_v3.preprocess_input
    # 从输入到 prev_tensor 的子模型
    sub_inp_to_prev = tf.keras.Model(base_model.input, prev_tensor)
    def gen():
        for _ in range(samples):
            img = (np.random.randint(0, 256, (h, w, 3)).astype(np.uint8)).astype(np.float32)
            img = preprocess(img)
            batch = np.expand_dims(img, axis=0).astype(np.float32)
            acts = sub_inp_to_prev(batch).numpy()  # 形如 [1, ...]
            yield [acts.astype(np.float32)]
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, choices=[2,3,4,5,6,7])
    parser.add_argument('--suffix', type=str, default=None)
    args = parser.parse_args()

    m = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)
    # 动态读取切点（若失败则回退到默认）
    cut_names = load_cut_names_from_summary(BASE)
    if not cut_names:
        cut_names = ['mixed3', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10']
    segments = build_segments(m, cut_names)

    k = args.k
    # 前缀组合：前 k-1 段各自独立，最后一段为“尾巴全部合并”
    out_dir = BASE / f'combos_K{k}' if not args.suffix else BASE / f'combos_K{k}_{args.suffix}'
    ensure_dirs(out_dir)

    # 生成前 k-1 段
    for i in range(k-1):
        s_in, s_out = segments[i]
        sub = tf.keras.Model(s_in, s_out, name=f'seg{i+1}')
        rep = make_rep_from_prev(m, s_in, samples=32)
        tfl = convert_tflite_int8(sub, rep)
        tfl_path = out_dir / 'tflite' / f'seg{i+1}_int8.tflite'
        tfl_path.write_bytes(tfl)
        compile_tpu(tfl_path, out_dir, f'seg{i+1}')

    # 生成尾段（从 seg_k 到 seg_8）
    tail_in, _ = segments[k-1]
    _, tail_out = segments[-1]
    tail = tf.keras.Model(tail_in, tail_out, name=f'tail_seg{k}_to_8')
    rep_tail = make_rep_from_prev(m, tail_in, samples=64)
    tfl_tail = convert_tflite_int8(tail, rep_tail)
    t_tail = out_dir / 'tflite' / f'tail_seg{k}_to_8_int8.tflite'
    t_tail.write_bytes(tfl_tail)
    compile_tpu(t_tail, out_dir, f'tail_seg{k}_to_8')

    meta = {
        'created_at': datetime.now().isoformat(),
        'k': k,
        'cut_names': CUT_NAMES,
        'scheme': 'prefix + merged tail',
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'done: {out_dir}')

if __name__ == '__main__':
    main()


