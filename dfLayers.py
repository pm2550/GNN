"""
生成适合Edge TPU的轻量级层模型（感知检测类工作负载）
依赖：pip install tensorflow==2.15 numpy
Edge TPU 优化：小模型、低通道数、适合移动端推理
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

# ===================== Edge TPU友好的轻量级配置 ===============================
configs = {
    # Conv2D：MobileNet风格，轻量级卷积
    "conv2d_light": {
        "input_shape": (1, 224, 224, 3),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu6"),
            tf.keras.layers.Conv2D(64, 1, padding="same", activation="relu6"),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu6"),
        ])(inp)
    },
    # DepthwiseConv2D：MobileNet核心层
    "depthwise_conv2d_light": {
        "input_shape": (1, 112, 112, 32),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu6"),
            tf.keras.layers.Conv2D(64, 1, padding="same", activation="relu6"),
            tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu6"),
        ])(inp)
    },
    # MaxPool：典型特征提取
    "max_pool_light": {
        "input_shape": (1, 56, 56, 64),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(2, strides=2),
            tf.keras.layers.MaxPooling2D(2, strides=2),
        ])(inp)
    },
    # AvgPool：固定大小的平均池化（Edge TPU兼容）
    "avg_pool_light": {
        "input_shape": (1, 56, 56, 64),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(2, strides=2),
            tf.keras.layers.AveragePooling2D(2, strides=2),
            tf.keras.layers.AveragePooling2D(2, strides=2),
        ])(inp)
    },
    # Dense：轻量级分类头
    "dense_light": {
        "input_shape": (1, 128),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu6"),
            tf.keras.layers.Dense(1000, activation=None),  # ImageNet分类数
        ])(inp)
    },
    # SeparableConv：Xception风格可分离卷积
    "separable_conv_light": {
        "input_shape": (1, 112, 112, 32),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu6"),
            tf.keras.layers.SeparableConv2D(128, 3, padding="same", activation="relu6"),
        ])(inp)
    },
    # Detection head：目标检测常用的预测头
    "detection_head": {
        "input_shape": (1, 19, 19, 512),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu6"),
            tf.keras.layers.Conv2D(84, 1, padding="same", activation=None),  # 21类×4坐标
        ])(inp)
    },
    # Feature pyramid：多尺度特征融合
    "feature_pyramid": {
        "input_shape": (1, 38, 38, 256),
        "build": lambda inp: tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu6"),
            tf.keras.layers.UpSampling2D(2),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu6"),
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
    print("Saved", path, f"({len(tflm)/1024:.1f} KB)")

# ===================== 主流程 ===============================
def main():
    print("生成Edge TPU友好的轻量级模型...")
    for name, cfg in configs.items():
        print(f"\n生成 {name}:")
        inp = tf.keras.Input(shape=cfg["input_shape"][1:])
        out = cfg["build"](inp)
        model = tf.keras.Model(inp, out)
        
        # 显示模型信息
        total_params = model.count_params()
        print(f"  参数量: {total_params:,}")
        
        export(model, name, cfg["input_shape"], quantize=False)
        export(model, name, cfg["input_shape"], quantize=True)

if __name__ == "__main__":
    main()
