"""
make_two_models.py
生成用于 SRAM 竞争实验的两个 TFLite 模型
A  large_dense_int8.tflite   仅含 Dense 权重 8.4 MB
B  many_conv_int8.tflite     含多层卷积 权重约 4 MB
Edge TPU 共编译时总权重超 8 MB 可观察优先级与访问频率策略
"""
import os, numpy as np, tensorflow as tf

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- 代表性数据集函数 ----------------
def rep_data_dense():
    for _ in range(100):
        yield [np.random.rand(1, 1024).astype("float32")]

def rep_data_conv():
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype("float32")]

# ---------------- A 大 Dense 模型 ----------------
def build_large_dense():
    inputs = tf.keras.Input(shape=(1024,))
    x = tf.keras.layers.Dense(8192, use_bias=True)(inputs)   # 1024×8192 ≈ 8.39 MB (INT8)
    return tf.keras.Model(inputs, x, name="large_dense")

# ---------------- B 多卷积模型 ----------------
def build_many_conv():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    # conv 1 3×3 3→64
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation=None)(x)
    # conv 2 3×3 64→128
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation=None)(x)
    # conv 3 3×3 128→256
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation=None)(x)
    # conv 4 3×3 256→512
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation=None)(x)
    # conv 5 3×3 512→512
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation=None)(x)
    # conv 6 1×1 512→1024
    x = tf.keras.layers.Conv2D(1024, 1, padding="same", activation=None)(x)
    return tf.keras.Model(inputs, x, name="many_conv")

# ---------------- TFLite 量化导出 ----------------
def to_tflite_int8(model, file_name, rep_fn):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_fn
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    tflite_model = conv.convert()
    path = os.path.join(SAVE_DIR, file_name)
    with open(path, "wb") as f:
        f.write(tflite_model)
    print("Saved", path)

def main():
    # 构建并量化 A
    model_a = build_large_dense()
    to_tflite_int8(model_a, "large_dense_int8.tflite", rep_data_dense)

    # 构建并量化 B
    model_b = build_many_conv()
    to_tflite_int8(model_b, "many_conv_int8.tflite", rep_data_conv)

if __name__ == "__main__":
    main()
