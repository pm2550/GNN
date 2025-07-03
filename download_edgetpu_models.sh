#!/bin/bash
# EdgeTPU模型下载脚本
# 自动下载推荐的模型文件

set -e

MODEL_DIR="models"
mkdir -p $MODEL_DIR

echo "🔄 开始下载EdgeTPU优化模型..."


# 检查下载工具
if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl -L -o"
    echo "📦 使用 curl 下载..."
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -O"
    echo "📦 使用 wget 下载..."
else
    echo "❌ 未找到 curl 或 wget，请手动下载模型文件"
    echo "📂 推荐从 Google Coral 官方获取真实的 EdgeTPU 模型："
    echo "   https://coral.ai/models/"
    exit 1
fi

# 实际可用的 EdgeTPU 模型下载
echo ""
echo "📥 下载官方 EdgeTPU 模型..."

# 1. MobileNet SSD v2 (目标检测)
echo "  📥 MobileNet SSD v2 Object Detection..."
$DOWNLOAD_CMD $MODEL_DIR/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" || \
    echo "    ⚠️  MobileNet SSD 下载失败"

# 2. DeepLab v3 (语义分割)
echo "  📥 DeepLab v3 Semantic Segmentation..."
$DOWNLOAD_CMD $MODEL_DIR/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/deeplabv3_mnv2_pascal_quant_edgetpu.tflite" || \
    echo "    ⚠️  DeepLab v3 下载失败"

# 3. MobileNet v2 分类 (可用于特征提取)
echo "  📥 MobileNet v2 Classification..."
$DOWNLOAD_CMD $MODEL_DIR/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite" || \
    echo "    ⚠️  MobileNet v2 下载失败"

# 4. EfficientNet EdgeTPU (如果可用)
echo "  📥 尝试下载 EfficientNet EdgeTPU..."
$DOWNLOAD_CMD $MODEL_DIR/efficientnet_edgetpu_s_quant_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/efficientnet_edgetpu_s_quant_edgetpu.tflite" || \
    echo "    ⚠️  EfficientNet EdgeTPU 下载失败"


echo "✅ 模型下载完成！"
echo "📁 模型保存在: $MODEL_DIR/"
ls -la $MODEL_DIR/

echo ""
echo "🚀 下一步："
echo "1. 验证模型文件完整性: ls -la $MODEL_DIR/*.tflite"
echo "2. 运行模型测试: python test_tpu_models.py"
echo "3. 集成到感知流水线"
echo ""
echo "📋 已下载的 EdgeTPU 模型用途："
echo "  ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite - 目标检测"
echo "  deeplabv3_mnv2_pascal_quant_edgetpu.tflite - 语义分割"
echo "  mobilenet_v2_1.0_224_quant_edgetpu.tflite - 特征提取/分类"
echo "  efficientnet_edgetpu_s_quant_edgetpu.tflite - 高效分类"
echo ""
echo "💡 注意：深度估计模型需要自行训练或寻找第三方实现"
