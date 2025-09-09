# YOLOv8 to EdgeTPU Docker 环境配置

## 🐳 方法1: 使用Dockerfile（推荐）

### 构建镜像
```bash
# 在包含Dockerfile.yolov8的目录中执行
docker build -f Dockerfile.yolov8 -t yolov8-edgetpu:latest .
```

### 运行容器
```bash
docker run --rm -it --gpus all \
  -v /mnt/c/Users/pm/Desktop/TPU/GNN_new/GNN:/workplace \
  -w /workplace \
  --name yolov8-edgetpu \
  yolov8-edgetpu:latest
```

## 🔧 方法2: 在现有容器中安装（快速）

### 启动你的现有容器
```bash
docker run --rm -it --gpus all \
  -v /mnt/c/Users/pm/Desktop/TPU/GNN_new/GNN:/workplace \
  -w /workplace \
  --name tf2.10-gpu \
  tensorflow/tensorflow:2.10.0-gpu \
  bash
```

### 在容器中运行安装脚本
```bash
# 复制安装脚本到容器（在宿主机上执行）
docker cp setup_yolov8_environment.sh tf2.10-gpu:/tmp/

# 在容器中执行（在容器内执行）
bash /tmp/setup_yolov8_environment.sh
source ~/.bashrc
```

## 🎯 使用转换脚本

### 1. 复制YOLOv8模型
```bash
cd /workplace/models
cp ../AART/yolov8n.pt .
```

### 2. 运行完整转换流水线
```bash
# 先生成ONNX文件
python3 raw_pytorch_export.py

# 然后转换为TFLite和EdgeTPU
python3 final_convert.py
```

## 📋 转换流程说明

1. **YOLOv8.pt → ONNX** (`raw_pytorch_export.py`)
   - 使用PyTorch直接导出，避免ultralytics内存问题
   - 生成 `yolov8n_raw.onnx` (12MB)

2. **ONNX → TFLite** (`final_convert.py`)
   - 使用onnx-tf转换为TensorFlow SavedModel
   - 生成float32和INT8量化版本
   
3. **TFLite → EdgeTPU**
   - 使用edgetpu_compiler编译
   - 支持segment分割编译（如果需要）

## 🔍 环境配置详情

### 系统依赖
- `libgl1-mesa-glx` - OpenCV图形库支持
- `libglib2.0-0` - GLib库
- `git` - Git版本控制
- `libjemalloc` - 内存分配器（解决double free问题）

### Python库版本
- `tensorflow==2.10.0` - 基础TensorFlow
- `ultralytics==8.2.0` - 降级的稳定版本
- `protobuf==3.20.3` - 解决版本冲突
- `onnx-tf` - ONNX到TensorFlow转换
- `tf2onnx` - TensorFlow到ONNX转换
- `onnx2tf` - 备用转换方法

### 环境变量
- `CUDA_VISIBLE_DEVICES=""` - 强制CPU模式避免冲突
- `MALLOC_CHECK_=0` - 禁用内存检查避免double free
- `LD_PRELOAD=libjemalloc.so.2` - 使用jemalloc内存分配器

## 🎉 预期结果

成功运行后，你将得到：
- ✅ `yolov8n_raw.onnx` (12MB) - ONNX格式
- ✅ `yolov8n_float32.tflite` (5.1MB) - TFLite float32版本
- ✅ `yolov8n_int8.tflite` (如果量化成功) - INT8量化版本
- ⚠️ EdgeTPU编译（可能需要segment分割或存在兼容性限制）

## 🔧 故障排除

### 如果遇到"double free"错误
```bash
export MALLOC_CHECK_=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

### 如果遇到CUDA冲突
```bash
export CUDA_VISIBLE_DEVICES=""
```

### 如果EdgeTPU编译失败
- YOLOv8使用的Cast操作不被EdgeTPU原生支持
- TFLite模型仍可在支持TensorFlow Lite的设备上运行
- 考虑使用segment分割编译获得部分加速

## 📝 注意事项

1. EdgeTPU对操作支持有限制，YOLOv8的某些操作可能无法完全在TPU上运行
2. 生成的TFLite模型完全可用，即使EdgeTPU编译失败
3. 环境变量设置对避免库冲突很重要
4. 如果需要完全的EdgeTPU兼容性，可能需要重新训练模型或使用专门的EdgeTPU优化版本 