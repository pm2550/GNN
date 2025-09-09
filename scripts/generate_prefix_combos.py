#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
COMPILER = str(ROOT / 'edgetpu_compiler')

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


def ensure_dirs(out_root: Path):
    (out_root / 'tflite').mkdir(parents=True, exist_ok=True)
    (out_root / 'tpu').mkdir(parents=True, exist_ok=True)
    (out_root / 'logs').mkdir(parents=True, exist_ok=True)


def build_segments_from_cuts(model: tf.keras.Model, cut_names: list[str]):
    segs = []
    prev = model.input
    for name in cut_names:
        out = model.get_layer(name).output
        segs.append((prev, out))
        prev = out
    segs.append((prev, model.output))
    return segs


def make_rep_from_input(preprocess_fn, input_size_hw: tuple[int, int], samples: int = 32):
    h, w = input_size_hw
    def gen():
        for _ in range(samples):
            img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8).astype(np.float32)
            x = preprocess_fn(img)[None, ...]
            yield [x.astype(np.float32)]
    return gen


def make_rep_from_tensor(base_model: tf.keras.Model, tensor, preprocess_fn, input_size_hw, samples: int = 32):
    feeder = tf.keras.Model(base_model.input, tensor)
    h, w = input_size_hw
    def gen():
        for _ in range(samples):
            img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8).astype(np.float32)
            x = preprocess_fn(img)[None, ...]
            y = feeder(x, training=False).numpy()
            yield [y.astype(np.float32)]
    return gen


def tflite_from_concrete(model_sub: tf.keras.Model, rep_fn):
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
    res = tf.compat.v1.gfile.GFile
    ts = time.strftime('%Y%m%d-%H%M%S')
    proc = tf.compat.v1
    import subprocess
    r = subprocess.run([COMPILER, '-o', str(tpu_dir), str(tfl_path)], capture_output=True, text=True)
    with open(logs_dir / f'{ts}_{tag}.log', 'w', encoding='utf-8') as f:
        if r.stdout:
            f.write('STDOUT\n'); f.write(r.stdout); f.write('\n')
        if r.stderr:
            f.write('STDERR\n'); f.write(r.stderr); f.write('\n')
    edgetpu_path = tpu_dir / (tfl_path.stem + '_edgetpu.tflite')
    ok = edgetpu_path.exists()
    size_mb = os.path.getsize(edgetpu_path)/(1024*1024) if ok else None
    return ok, size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--k', required=True, type=int, choices=[2,3,4,5,6,7])
    parser.add_argument('--suffix', type=str, default=None)
    args = parser.parse_args()

    reg = MODEL_REGISTRY[args.model]
    ctor = reg['ctor']
    preprocess_fn = reg['preprocess']
    input_hw = reg['input_size']

    base_dir = ROOT / 'models_local' / 'public' / f"{args.model.lower()}_8seg_uniform_local"
    summary_path = base_dir / 'full_split_pipeline_local' / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'summary.json 不存在: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    # 兼容不同键名：优先 cut_points，其次 cut_names，再次从 decisions 推导
    cut_names = summary.get('cut_points') or summary.get('cut_names')
    if (not cut_names) and isinstance(summary.get('decisions'), list):
        names = []
        for item in summary['decisions']:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                nm = item[0]
                if isinstance(nm, str) and nm != 'OUTPUT':
                    names.append(nm)
        cut_names = names
    if not cut_names or len(cut_names) != 7:
        raise ValueError('cut_names/cut_points 无效或不为 7 个（8 段需要 7 个切点）')

    out_dir = base_dir / (f"combos_K{args.k}_{args.suffix}" if args.suffix else f"combos_K{args.k}")
    ensure_dirs(out_dir)
    tfl_dir = out_dir / 'tflite'

    # 构建模型与段
    m = ctor(weights='imagenet', include_top=True)
    segments = build_segments_from_cuts(m, cut_names)

    # 生成前 k-1 段（代表性数据：原图经 preprocess，直接喂第一段；第2段起使用中间激活）
    for i in range(args.k - 1):
        s_in, s_out = segments[i]
        sub = tf.keras.Model(s_in, s_out, name=f'seg{i+1}')
        if i == 0:
            rep = make_rep_from_input(preprocess_fn, input_hw, samples=32)
        else:
            rep = make_rep_from_tensor(m, s_in, preprocess_fn, input_hw, samples=32)
        tfl = tflite_from_concrete(sub, rep)
        tfl_path = tfl_dir / f'seg{i+1}_int8.tflite'
        tfl_path.write_bytes(tfl)
        compile_tpu(tfl_path, out_dir, f'seg{i+1}')

    # 生成尾部组合段（seg_k 到 seg_8）
    tail_in, _ = segments[args.k - 1]
    _, tail_out = segments[-1]
    tail = tf.keras.Model(tail_in, tail_out, name=f'tail_seg{args.k}_to_8')
    rep_tail = make_rep_from_tensor(m, tail_in, preprocess_fn, input_hw, samples=32)
    tfl_tail = tflite_from_concrete(tail, rep_tail)
    t_tail = tfl_dir / f'tail_seg{args.k}_to_8_int8.tflite'
    t_tail.write_bytes(tfl_tail)
    compile_tpu(t_tail, out_dir, f'tail_seg{args.k}_to_8')

    meta = {
        'model': args.model,
        'k': args.k,
        'cut_names': cut_names,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'scheme': 'prefix + merged tail'
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'done: {out_dir}')


if __name__ == '__main__':
    main()


