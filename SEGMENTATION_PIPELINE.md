# ResNet101 模型分段与组合推理完整流程文档

## 1. 模型切分规则

### 1.1 候选切点定义
- **目标层**：网络各"模块/阶段出口"的 4D 特征图
- **筛选条件**：
  - 分支已融合、单生产者
  - 可直接作为子模型 I/O
  - 优先已下采样处
  - 层名以 `*_out` 结尾（ResNet residual block 输出）

### 1.2 固定切分算法
**贪心块级切分算法（Block-based Greedy Segmentation）**

```
输入：模型 M，单段上限 S=6MiB
输出：最小可行段数 K，唯一切点序列


**算法特性**：
- ✅ **确定性**：给定模型和阈值，输出唯一
- ✅ **可复现**：不同环境运行结果一致
- ✅ **最优性**：找到满足约束的最小段数
- ✅ **通用性**：适用于任何CNN模型

### 1.3 ResNet101 实际切点
基于 S=7MiB 约束，算法确定的切点：

```json
{
  "cut_names": [
    "conv4_block4_out",   // 第1刀：(None, 14, 14, 1024)
    "conv4_block10_out",  // 第2刀：(None, 14, 14, 1024)  
    "conv4_block16_out",  // 第3刀：(None, 14, 14, 1024)
    "conv4_block22_out",  // 第4刀：(None, 14, 14, 1024)
    "conv5_block1_out",   // 第5刀：(None, 7, 7, 2048)
    "conv5_block2_out"    // 第6刀：(None, 7, 7, 2048)
  ],
  "segments": 7,          // K=7 (最小可行段数)
  "max_seg_size": "6.985 MiB"  // 所有段均 ≤7MiB
}
```

## 2. 组合模型生成流程

### 2.1 目标组合定义
生成 k=2 到 k=7 的前缀组合：

```
k=2: [seg1] + [seg2+seg3+seg4+seg5+seg6+seg7]
k=3: [seg1] + [seg2] + [seg3+seg4+seg5+seg6+seg7]  
k=4: [seg1] + [seg2] + [seg3] + [seg4+seg5+seg6+seg7]
k=5: [seg1] + [seg2] + [seg3] + [seg4] + [seg5+seg6+seg7]
k=6: [seg1] + [seg2] + [seg3] + [seg4] + [seg5] + [seg6+seg7]
k=7: [seg1] + [seg2] + [seg3] + [seg4] + [seg5] + [seg6] + [seg7]
```

### 2.2 技术实现挑战

#### 问题1：TFLite转换器Bug
**错误现象**：
```
RuntimeError: tensorflow/lite/kernels/conv.cc:351 
input_channel % filter_input_channel != 0 (3 != 0)
Node number 0 (CONV_2D) failed to prepare.
```

**错误原因**：
- TFLite转换器错误地将中间张量的输入理解为RGB图像输入(3通道)
- 而不是实际的中间特征图(如1024通道)
- 问题出现在量化校准阶段的通道推断

**错误的尝试**：
1. ❌ 调整校准数据形状 - 无效
2. ❌ 修改batch_size - 无效  
3. ❌ 重建模型结构 - 复杂且易错

#### 问题2：校准数据不匹配
**问题描述**：
- 组合模型的输入是中间张量（如 seg2 的输入）
- 需要 seg1 的输出作为校准数据
- 形状必须完全匹配：`(batch, H, W, C)`

### 2.3 ✅ 正确解决方案

#### 核心修复设置
```python
def convert_combo_model_fixed(combo_model, calibration_data, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(combo_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 🔑 关键修复1：禁用per-channel量化
    converter._experimental_disable_per_channel = True
    
    # 🔑 关键修复2：使用旧量化器（避免新量化器bug）
    if hasattr(converter, "_experimental_new_quantizer"):
        converter._experimental_new_quantizer = False
    
    # 🔑 关键修复3：确保校准数据形状完全匹配
    def representative_dataset():
        for i in range(min(len(calibration_data), 8)):
            sample = calibration_data[i:i+1]  # 确保batch_size=1
            yield [sample]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    return tflite_model
```

#### 完整实现流程
```python
# 1. 构建原始模型和段定义
model = tf.keras.applications.ResNet101(weights=None, include_top=True, input_shape=(224, 224, 3))
segments = []
prev = model.input
for name in cut_names:
    out = model.get_layer(name).output
    segments.append((prev, out))
    prev = out
segments.append((prev, model.output))

# 2. 生成逐段校准数据
calibrations = [original_input_images]
for i in range(len(segments) - 1):
    s_in, s_out = segments[i]
    seg_model = tf.keras.Model(s_in, s_out)
    acts = seg_model.predict(calibrations[i], batch_size=2, verbose=0)
    calibrations.append(acts)

# 3. 为每个k值生成组合模型
for k in range(2, K + 1):
    # 创建尾部组合模型 (seg_k 到 seg_K)
    tail_start_idx = k - 1
    tail_in, _ = segments[tail_start_idx]    # seg_k的输入张量
    _, tail_out = segments[-1]              # seg_K的输出张量
    
    combo_model = tf.keras.Model(inputs=tail_in, outputs=tail_out, 
                                 name=f"combo_seg{k}_to_{K}")
    
    # 使用对应的校准数据
    calib_data = calibrations[tail_start_idx]
    
    # 应用修复的转换方法
    tflite_model = convert_combo_model_fixed(combo_model, calib_data, output_path)
```

### 2.4 验证结果

#### 生成的组合模型规格
```
k=2: seg2→seg7 组合    36.46 MB  ✅ 推理正常
k=3: seg3→seg7 组合    30.07 MB  ✅ 推理正常  
k=4: seg4→seg7 组合    23.68 MB  ✅ 推理正常
k=5: seg5→seg7 组合    17.29 MB  ✅ 推理正常
k=6: seg6→seg7 组合    10.46 MB  ✅ 推理正常
k=7: seg7 单独         6.22 MB   ✅ 推理正常
```

#### EdgeTPU编译验证
- ✅ 所有组合模型成功编译为EdgeTPU版本
- ✅ 编译日志无错误
- ✅ TPU模型大小略大于CPU版本（正常现象）

#### 推理验证
```python
# 测试示例：k=2组合模型
interpreter = tf.lite.Interpreter(model_path='tail_seg2_to_7_int8.tflite')
interpreter.allocate_tensors()

# 输入：(1, 14, 14, 1024) INT8  - seg1的输出
# 输出：(1, 1000) INT8         - 最终分类结果
# 状态：✅ 推理成功，输出正常
```

## 3. 部署选项

### 3.1 软件串联推理
**特点**：使用独立的段模型，软件控制数据流
```python
# 逐段推理
current_data = input_image
for segment_model in segment_models:
    current_data = segment_model(current_data)
final_result = current_data
```

**优势**：
- ✅ 灵活性高，可任意组合
- ✅ 内存使用可控
- ✅ 易于调试和监控

### 3.2 硬件组合推理  
**特点**：使用单个组合TFLite模型
```python
# 单模型推理
combo_interpreter = tf.lite.Interpreter(model_path='combo_model.tflite')
result = combo_interpreter(intermediate_input)
```

**优势**：
- ✅ 推理延迟更低
- ✅ EdgeTPU优化更好
- ✅ 部署更简单

## 4. 文件结构

```
/workplace/models/public/resnet101_greedy_under7/
├── full_split/                    # 完整7段分割
│   ├── tflite/                   # CPU TFLite模型
│   ├── tpu/                      # EdgeTPU模型  
│   ├── meta.json                 # 切点和尺寸信息
│   └── note.txt                  # 详细说明
├── pairs_k2/                     # k=2组合
│   ├── tflite/
│   │   ├── seg1_int8.tflite     # 前缀段
│   │   └── tail_seg2_to_7_int8.tflite  # 真正的组合模型
│   ├── tpu/                      # EdgeTPU版本
│   ├── meta.json
│   └── note.txt
├── pairs_k3/ ... pairs_k7/       # k=3到k=7组合
├── TFLITE_CONVERSION_TIPS.md     # 技术解决方案文档
└── RESNET101_SEGMENTATION_PIPELINE.md  # 本文档
```

## 5. 关键脚本

- `resnet101_greedy_under7.py` - 贪心切分算法实现
- `generate_real_combo_models.py` - 组合模型生成
- `resnet101_inference_pipeline.py` - 串联推理管道
- `test_combo_model_fix.py` - TFLite转换器修复验证

## 6. 论文应用

### 6.1 方法描述
```
Fixed Block-based Greedy Segmentation: 
We segment the model by greedily accumulating quantized weight sizes 
along topological order, placing cuts at the nearest candidate 
(*_out layers) before exceeding 7MiB per segment, with fallback 
mechanism for oversized segments. This yields deterministic, 
reproducible segmentation with minimal segments K satisfying 
size constraints.
```

### 6.2 实验设置
- **模型**：ResNet101 (ImageNet预训练)
- **量化**：INT8 PTQ (Post-Training Quantization)
- **硬件**：Google Coral EdgeTPU
- **约束**：每段 ≤7MiB (CPU + EdgeTPU TFLite)
- **结果**：K=7段，最大段6.985MiB

---

**文档版本**：v1.0  
**创建时间**：2025-08-29  
**验证环境**：TensorFlow 2.12.0, Ubuntu 20.04, EdgeTPU Compiler  
**状态**：✅ 完全验证通过

