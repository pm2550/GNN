"""
build_models.py ── 生成 4 个“Edge-TPU 友好”Keras 模型
  • 输入尺寸固定为 (224, 224, 3)
  • 仅用 Conv / DepthwiseConv / Dense / ReLU —— Edge-TPU 100 % 支持
  • 不训练，只保存量化后的 .tflite，供 edgetpu_compiler 实验
运行：
  python build_models.py
然后：
  edgetpu_compiler models/model_4m.tflite   # 等
"""
import tensorflow as tf
import numpy as np
import pathlib

IMAGE_SIZE = 224
IMG_SHAPE  = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES  = 10          # 随便写一个分类头，层数少 → 每层 < 8 MB
REP_SAMPLES = 100        # 量化用的代表数据条数

# ───────────────────────────────────────────────────────────
# 0. 代表数据生成器（PTQ）
def rep_gen():
    for _ in range(REP_SAMPLES):
        # 生成float32类型的代表数据，范围[0, 255]
        yield [np.random.randint(0, 256, IMG_SHAPE, np.uint8)[None, ...].astype(np.float32)]

# 1. 用 MobileNet 控宽度：α≈1.0/1.15/1.3 ⇒ 4/5/6 MB
def mobilenet(alpha: float) -> tf.keras.Model:
    base = tf.keras.applications.MobileNet(
        input_shape=IMG_SHAPE, alpha=alpha,
        include_top=False, weights=None)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    return tf.keras.Model(base.input, out, name=f"mb_{alpha}")

# 2. 手写一个“24 MB”网络：10 个 3×3×512 Conv（每层≈2.36 MB）
def big24m() -> tf.keras.Model:
    inp = tf.keras.Input(IMG_SHAPE)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same',
                               activation='relu')(inp)
    for _ in range(10):                               # 10×2.36 ≈ 24 MB
        x = tf.keras.layers.Conv2D(512, 3, padding='same',
                                   activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inp, out, name="conv24m")

# 3. 构建 + 量化保存
def save_quant(model: tf.keras.Model, fname: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    pathlib.Path("models").mkdir(exist_ok=True)
    open(f"models/{fname}.tflite", "wb").write(tflite)
    print(f"{fname}.tflite  ► {len(tflite)/1e6:.2f} MB  "
          f"({model.count_params()/1e6:.2f} M params)")

if __name__ == "__main__":
    # save_quant(mobilenet(1.00), "model_4m")   # ≈ 4 MB
    # save_quant(mobilenet(1.15), "model_5m")   # ≈ 5 MB
    # save_quant(mobilenet(1.30), "model_6m")   # ≈ 6 MB
    # save_quant(big24m(),          "model_24m") # ≈ 24 MB
    save_quant(mobilenet(1.60), "model_9m")   # ≈ 4 MB