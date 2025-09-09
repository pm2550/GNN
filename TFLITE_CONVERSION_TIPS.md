# TFLite转换器问题解决方案

## 问题描述

在使用`tf.keras.Model(middle_tensor, output_tensor)`创建从中间张量开始的组合模型时，TFLite转换器会出现以下错误：

```
RuntimeError: tensorflow/lite/kernels/conv.cc:351 input_channel % filter_input_channel != 0 (3 != 0)
Node number 0 (CONV_2D) failed to prepare.
```

**错误原因**：TFLite转换器错误地将中间张量的输入理解为RGB图像输入(3通道)，而不是实际的中间特征图(如1024通道)。

## ✅ 解决方案

### 关键设置组合

```python
converter = tf.lite.TFLiteConverter.from_keras_model(combo_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 🔑 关键设置1：禁用per-channel量化
converter._experimental_disable_per_channel = True

# 🔑 关键设置2：使用旧量化器（避免新量化器的bug）
if hasattr(converter, "_experimental_new_quantizer"):
    converter._experimental_new_quantizer = False

# 🔑 关键设置3：确保校准数据形状完全匹配
def representative_dataset():
    for i in range(calibration_samples):
        sample = calibration_data[i:i+1]  # 确保batch_size=1
        yield [sample]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 转换
tflite_model = converter.convert()
```

### 完整工作示例

```python
# 创建组合模型（从seg2到seg7）
tail_in = segments[1][0]   # seg2的输入张量
tail_out = segments[-1][1] # seg7的输出张量
combo_model = tf.keras.Model(inputs=tail_in, outputs=tail_out, name="combo_model")

# 生成校准数据（seg1的输出激活）
seg1_model = tf.keras.Model(segments[0][0], segments[0][1])
calibration_acts = seg1_model.predict(input_images, batch_size=2, verbose=0)

# 应用解决方案
converter = tf.lite.TFLiteConverter.from_keras_model(combo_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_disable_per_channel = True
converter._experimental_new_quantizer = False

def rep_gen():
    for i in range(len(calibration_acts)):
        yield [calibration_acts[i:i+1]]  # (1, H, W, C)

converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
```

## 验证成功

- ✅ 转换成功，无错误
- ✅ 生成的TFLite模型可正常推理
- ✅ 模型大小合理（多段组合约36-40MB）
- ✅ 支持EdgeTPU编译

## 适用场景

此解决方案适用于：
- ResNet、MobileNet等CNN模型的中间切分
- 任何需要从中间张量开始的子模型转换
- INT8量化的分段模型部署

## 注意事项

1. **校准数据必须匹配**：确保校准数据的形状与中间张量完全一致
2. **batch_size=1**：校准时使用单样本批次
3. **旧量化器更稳定**：新量化器在处理中间张量时有bug
4. **per-tensor量化**：禁用per-channel避免通道不匹配问题

---
*记录时间：2025-08-29*  
*验证模型：ResNet101*  
*TensorFlow版本：2.12.0*

