"""
生成“加重量级”版本的典型层 TFLite 模型（浮点 + 全 INT8）
依赖：pip install tensorflow==2.15 numpy
Edge TPU 每片片上约 8 MiB 权重缓存，请勿无限增大；以下配置已验证可正常编译。
"""

import os, numpy as np, tensorflow as tf

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 代表性数据集（量化校准用） =====================
def rep_dataset(shape):
    def gen():
        for _ in range(100):
            yield [np.random.rand(*shape).astype("float32")]
    return gen

# ===================== 各层“重载”配置 ===============================
configs = {
    # Conv2D：3 × 重复，输入通道 32 → 64 → 128，卷积核 3×3
    "conv2d_heavy": {
        "input_shape": (1, 224, 224, 32),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding="same", activation=None),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation=None),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation=None),
        ])(inp)
    },
    # DepthwiseConv2D：5 × 重复，通道 64
    "depthwise_conv2d_heavy": {
        "input_shape": (1, 224, 224, 64),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, padding="same") for _ in range(5)
        ])(inp)
    },
    # MaxPool：在高分辨率输入上连做 5 次 2×2 pool
    "max_pool_heavy": {
        "input_shape": (1, 512, 512, 16),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(2) for _ in range(5)
        ])(inp)
    },
    # AvgPool：同上
    "avg_pool_heavy": {
        "input_shape": (1, 512, 512, 16),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(2) for _ in range(5)
        ])(inp)
    },
    # Dense：输入 1×1024，两个全连接 2048→2048
    "dense_heavy": {
        "input_shape": (1, 1024),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation=None),
            tf.keras.layers.Dense(2048, activation=None),
        ])(inp)
    },
    # ReLU：对 1×256×256×128 激活张量做 10 次 ReLU
    "relu_heavy": {
        "input_shape": (1, 256, 256, 128),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.ReLU() for _ in range(10)
        ])(inp)
    },
}

# ===================== TFLite 导出函数 ===============================
def export(model, name, shape, quantize=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_dataset(shape)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
    tflm = conv.convert()
    suffix = "_int8.tflite" if quantize else ".tflite"
    path = os.path.join(SAVE_DIR, name + suffix)
    with open(path, "wb") as f:
        f.write(tflm)
    print("Saved", path)

# ===================== 主流程 ===============================
def main():
    for name, cfg in configs.items():
        inp = tf.keras.Input(shape=cfg["input_shape"][1:])
        out = cfg["build"](inp)
        model = tf.keras.Model(inp, out)
        export(model, name, cfg["input_shape"], quantize=False)
        export(model, name, cfg["input_shape"], quantize=True)

if __name__ == "__main__":
    main()
