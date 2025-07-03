#!/usr/bin/env python3
"""
基于经典架构的EdgeTPU深度估计模型
1. MobileNetV2 + 深度估计头 (类似FastDepth)
2. 基于已发表论文的成熟架构
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import subprocess
import os

class FastDepthEdgeTPU:
    """
    基于FastDepth论文的EdgeTPU优化版本
    参考: FastDepth: Fast Monocular Depth Estimation on Embedded Systems (ICRA 2019)
    """
    
    def __init__(self, input_height=480, input_width=640):
        self.input_height = input_height
        self.input_width = input_width
        
    def build_fastdepth_model(self):
        """构建基于MobileNetV2的FastDepth模型"""
        
        # 输入
        input_tensor = Input(shape=(self.input_height, self.input_width, 3), name="rgb_input")
        
        # 使用预训练的MobileNetV2作为encoder (去掉顶层)
        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False,
            alpha=1.0  # 使用完整版本以获得最佳特征
        )
        
        # 提取多尺度特征 (FastDepth论文中的skip connections)
        # MobileNetV2的关键层
        skip1 = backbone.get_layer('block_1_expand_relu').output    # 112x112
        skip2 = backbone.get_layer('block_3_expand_relu').output    # 56x56  
        skip3 = backbone.get_layer('block_6_expand_relu').output    # 28x28
        skip4 = backbone.get_layer('block_13_expand_relu').output   # 14x14
        skip5 = backbone.output                                     # 7x7
        
        # FastDepth的上采样解码器
        # 第一层上采样
        x = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsample_1')(skip5)
        x = Concatenate(name='concat_1')([x, skip4])
        x = Conv2D(512, 3, padding='same', activation='relu', name='decode_conv_1')(x)
        
        # 第二层上采样  
        x = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsample_2')(x)
        x = Concatenate(name='concat_2')([x, skip3])
        x = Conv2D(256, 3, padding='same', activation='relu', name='decode_conv_2')(x)
        
        # 第三层上采样
        x = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsample_3')(x)
        x = Concatenate(name='concat_3')([x, skip2])
        x = Conv2D(128, 3, padding='same', activation='relu', name='decode_conv_3')(x)
        
        # 第四层上采样
        x = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsample_4')(x)
        x = Concatenate(name='concat_4')([x, skip1])
        x = Conv2D(64, 3, padding='same', activation='relu', name='decode_conv_4')(x)
        
        # 最终上采样到原始分辨率
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_final')(x)
        
        # 深度输出层
        depth_output = Conv2D(1, 3, padding='same', activation='sigmoid', name='depth_output')(x)
        
        model = Model(inputs=input_tensor, outputs=depth_output, name='FastDepth_EdgeTPU')
        return model

class MobileNetV2StereoDepth:
    """
    基于MobileNetV2的立体深度估计模型
    参考经典的立体匹配架构
    """
    
    def __init__(self, input_height=480, input_width=640, max_disparity=64):
        self.input_height = input_height
        self.input_width = input_width
        self.max_disparity = max_disparity
        
    def create_shared_encoder(self):
        """创建共享的MobileNetV2编码器"""
        
        input_tensor = Input(shape=(self.input_height, self.input_width, 3))
        
        # 使用MobileNetV2作为特征提取器
        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False,
            alpha=0.75  # 使用0.75倍宽度以减少计算量
        )
        
        # 选择合适的特征层 (1/4分辨率)
        features = backbone.get_layer('block_6_expand_relu').output  # 60x80x144
        
        return Model(inputs=input_tensor, outputs=features, name='stereo_encoder')
    
    def correlation_layer(self, left_features, right_features):
        """计算左右特征的相关性 - 简化版本"""
        
        batch_size = tf.shape(left_features)[0]
        height = tf.shape(left_features)[1] 
        width = tf.shape(left_features)[2]
        channels = tf.shape(left_features)[3]
        
        # 简化的相关性计算
        max_disp = self.max_disparity // 4  # 适应feature map分辨率
        cost_volume = []
        
        for d in range(0, max_disp, 2):  # 每隔2个像素计算一次
            if d == 0:
                shifted_right = right_features
            else:
                # 向左平移
                padding = tf.zeros([batch_size, height, d, channels])
                shifted_right = tf.concat([
                    right_features[:, :, d:, :], 
                    padding
                ], axis=2)
            
            # 计算绝对差值
            correlation = tf.reduce_mean(tf.abs(left_features - shifted_right), axis=-1, keepdims=True)
            cost_volume.append(correlation)
        
        return tf.concat(cost_volume, axis=-1)
    
    def build_stereo_model(self):
        """构建完整的立体深度估计模型"""
        
        # 输入
        left_input = Input(shape=(self.input_height, self.input_width, 3), name='left_image')
        right_input = Input(shape=(self.input_height, self.input_width, 3), name='right_image')
        
        # 共享编码器
        encoder = self.create_shared_encoder()
        
        left_features = encoder(left_input)
        right_features = encoder(right_input)
        
        # 计算cost volume
        cost_volume = self.correlation_layer(left_features, right_features)
        
        # 深度回归网络
        x = Conv2D(64, 3, padding='same', activation='relu', name='regress_conv1')(cost_volume)
        x = Conv2D(32, 3, padding='same', activation='relu', name='regress_conv2')(x)
        x = Conv2D(16, 3, padding='same', activation='relu', name='regress_conv3')(x)
        
        # 深度预测
        depth_low = Conv2D(1, 3, padding='same', activation='sigmoid', name='depth_low')(x)
        
        # 上采样到原始分辨率
        depth_output = UpSampling2D(size=(4, 4), interpolation='bilinear', name='depth_final')(depth_low)
        
        model = Model(
            inputs=[left_input, right_input], 
            outputs=depth_output, 
            name='MobileNetV2_StereoDepth'
        )
        
        return model

class MiDaSMobileNet:
    """
    基于MiDaS思想的MobileNet深度估计模型
    参考: Towards Robust Monocular Depth Estimation (MiDaS论文)
    """
    
    def __init__(self, input_height=480, input_width=640):
        self.input_height = input_height
        self.input_width = input_width
        
    def build_midas_mobilenet(self):
        """构建MiDaS风格的MobileNet模型"""
        
        input_tensor = Input(shape=(self.input_height, self.input_width, 3), name='image_input')
        
        # MobileNetV2 backbone
        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False,
            alpha=1.0
        )
        
        # 多尺度特征融合 (MiDaS的核心思想)
        feature_16 = backbone.get_layer('block_1_expand_relu').output   # 1/2 - 128x128
        feature_8 = backbone.get_layer('block_3_expand_relu').output    # 1/4 - 64x64
        feature_4 = backbone.get_layer('block_6_expand_relu').output    # 1/8 - 32x32
        feature_2 = backbone.get_layer('block_13_expand_relu').output   # 1/16 - 16x16
        feature_1 = backbone.output                                     # 1/32 - 8x8
        
        # Feature Fusion Module (类似MiDaS)
        def feature_fusion_block(features, target_size, name_prefix):
            """特征融合块 - 使用固定的缩放策略"""
            target_h, target_w = target_size
            
            # 根据已知的特征尺寸进行调整
            # feature_16: 128x128 -> 32x32 (需要1/4下采样)
            # feature_8:  64x64  -> 32x32 (需要1/2下采样) 
            # feature_4:  32x32  -> 32x32 (不需要调整)
            # feature_2:  16x16  -> 32x32 (需要2x上采样)
            # feature_1:  8x8    -> 32x32 (需要4x上采样)
            
            if 'fuse_16' in name_prefix:
                # 128x128 -> 32x32
                resized = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), 
                                         name=f'{name_prefix}_downsample')(features)
            elif 'fuse_8' in name_prefix:
                # 64x64 -> 32x32  
                resized = AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         name=f'{name_prefix}_downsample')(features)
            elif 'fuse_4' in name_prefix:
                # 32x32 -> 32x32 (already correct size)
                resized = features
            elif 'fuse_2' in name_prefix:
                # 16x16 -> 32x32
                resized = UpSampling2D(size=(2, 2), interpolation='bilinear',
                                     name=f'{name_prefix}_upsample')(features)
            elif 'fuse_1' in name_prefix:
                # 8x8 -> 32x32
                resized = UpSampling2D(size=(4, 4), interpolation='bilinear',
                                     name=f'{name_prefix}_upsample')(features)
            else:
                resized = features
            
            # 通道调整到256
            adjusted = Conv2D(256, 1, padding='same', name=f'{name_prefix}_adjust')(resized)
            return adjusted
        
        target_size = [self.input_height // 8, self.input_width // 8]  # 1/8分辨率作为融合目标 (32x32)
        
        # 将所有特征调整到相同尺寸 (32x32x256)
        fused_1 = feature_fusion_block(feature_1, target_size, 'fuse_1')     # 8x8 -> 32x32
        fused_2 = feature_fusion_block(feature_2, target_size, 'fuse_2')     # 16x16 -> 32x32
        fused_4 = feature_fusion_block(feature_4, target_size, 'fuse_4')     # 32x32 -> 32x32 (already correct size)
        fused_8 = feature_fusion_block(feature_8, target_size, 'fuse_8')     # 64x64 -> 32x32
        fused_16 = feature_fusion_block(feature_16, target_size, 'fuse_16')  # 128x128 -> 32x32
        
        # 特征融合 - 现在所有特征都是32x32x256
        fused_features = Add(name='feature_fusion')([fused_1, fused_2, fused_4, fused_8, fused_16])
        fused_features = ReLU(name='fusion_relu')(fused_features)
        
        # 深度解码器
        x = Conv2D(128, 3, padding='same', activation='relu', name='decode_1')(fused_features)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up_1')(x)
        
        x = Conv2D(64, 3, padding='same', activation='relu', name='decode_2')(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up_2')(x)
        
        x = Conv2D(32, 3, padding='same', activation='relu', name='decode_3')(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up_3')(x)
        
        # 最终深度输出
        depth_output = Conv2D(1, 3, padding='same', activation='sigmoid', name='depth_final')(x)
        
        model = Model(inputs=input_tensor, outputs=depth_output, name='MiDaS_MobileNet')
        return model

def optimize_for_edgetpu(model, model_name):
    """为EdgeTPU优化模型"""
    
    print(f"🔧 为EdgeTPU优化 {model_name}...")
    
    # 量化感知训练配置
    import tensorflow_model_optimization as tfmot
    
    # 定义量化配置 - 针对EdgeTPU优化
    def apply_quantization(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    
    # 应用量化注解
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization,
    )
    
    # 量化模型
    quantized_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    return quantized_model

def convert_to_tflite(model, model_name):
    """步骤1: 转换为TensorFlow Lite (基础转换)"""
    
    print(f"📦 步骤1: 转换 {model_name} 为TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 基础优化，不量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 转换
    tflite_model = converter.convert()
    
    # 保存基础TFLite模型
    output_path = f"{model_name.lower()}.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite模型已保存: {output_path}")
    print(f"   模型大小: {len(tflite_model) / (1024*1024):.2f} MB")
    return output_path

def quantize_keras_to_tflite(keras_model, model_name, representative_data_gen):
    """步骤2: 从Keras模型直接量化为INT8 TFLite"""
    
    print(f"⚙️  步骤2: 量化 {model_name} 为INT8...")
    
    # 生成量化版本文件名
    quantized_path = f"{model_name.lower()}_quantized.tflite"
    
    try:
        print(f"🔄 正在进行INT8量化...")
        
        # 从Keras模型创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        
        # 设置量化优化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 设置代表性数据集用于量化校准
        converter.representative_dataset = representative_data_gen
        
        # 强制INT8量化
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # 设置输入输出类型为INT8（EdgeTPU要求）
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        print(f"📊 使用代表性数据进行量化校准...")
        
        # 执行量化转换
        quantized_tflite_model = converter.convert()
        
        # 保存量化模型
        with open(quantized_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        print(f"✅ INT8量化模型已保存: {quantized_path}")
        
        # 显示文件大小
        quantized_size = len(quantized_tflite_model) / (1024 * 1024)
        print(f"📊 量化模型大小: {quantized_size:.2f}MB")
        
        return quantized_path
        
    except Exception as e:
        print(f"❌ INT8量化失败: {e}")
        print(f"💡 尝试创建默认量化版本...")
        
        # 如果INT8量化失败，创建默认量化版本
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            fallback_model = converter.convert()
            
            with open(quantized_path, 'wb') as f:
                f.write(fallback_model)
            
            print(f"✅ 默认量化模型已保存: {quantized_path}")
            print(f"⚠️  注意: 这不是INT8量化，EdgeTPU兼容性可能有限")
            
            return quantized_path
        except Exception as e2:
            print(f"❌ 默认量化也失败: {e2}")
            return None

def compile_to_edgetpu(quantized_tflite_path, model_name):
    """步骤3: 编译量化模型为EdgeTPU格式"""
    
    print(f"🚀 步骤3: 编译 {model_name} 为EdgeTPU...")
    
    if not os.path.exists(quantized_tflite_path):
        print(f"❌ 量化文件不存在: {quantized_tflite_path}")
        return None
    
    # 生成EdgeTPU文件名和日志文件名
    edgetpu_path = quantized_tflite_path.replace('_quantized.tflite', '_edgetpu.tflite')
    log_path = quantized_tflite_path.replace('_quantized.tflite', '_edgetpu.log')
    
    print(f"🔄 编译 {quantized_tflite_path} -> {edgetpu_path}")
    print(f"� 编译日志将保存到: {log_path}")
    
    try:
        # 运行EdgeTPU编译器并保存日志
        result = subprocess.run([
            'edgetpu_compiler', 
            quantized_tflite_path,
            '-o', '.',
            '--out_dir', '.'
        ], capture_output=True, text=True, timeout=300)
        
        # 保存详细日志
        log_content = f"EdgeTPU编译日志 - {model_name}\n"
        log_content += f"="*50 + "\n"
        log_content += f"输入文件: {quantized_tflite_path}\n"
        log_content += f"输出文件: {edgetpu_path}\n"
        log_content += f"编译时间: {subprocess.time.time() if hasattr(subprocess, 'time') else 'N/A'}\n"
        log_content += f"返回码: {result.returncode}\n\n"
        
        if result.stdout:
            log_content += "标准输出:\n"
            log_content += result.stdout + "\n\n"
        
        if result.stderr:
            log_content += "错误输出:\n"
            log_content += result.stderr + "\n\n"
        
        # 写入日志文件
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        if result.returncode == 0:
            print(f"✅ EdgeTPU编译成功: {edgetpu_path}")
            print(f"📊 编译信息:")
            
            # 显示关键信息
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if ('Model successfully compiled' in line or 
                        'Ops mapped to Edge TPU' in line or
                        'Compilation succeeded' in line):
                        print(f"   {line}")
            
            # 检查文件大小
            if os.path.exists(edgetpu_path):
                size_mb = os.path.getsize(edgetpu_path) / (1024 * 1024)
                print(f"   EdgeTPU模型大小: {size_mb:.2f} MB")
            
            print(f"📝 详细日志: {log_path}")
            return edgetpu_path
        else:
            print(f"❌ EdgeTPU编译失败")
            print(f"📝 详细错误日志: {log_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ 编译超时: {quantized_tflite_path}")
        return None
    except Exception as e:
        print(f"❌ 编译异常: {e}")
        return None

def main():
    """主函数 - 创建三种经典架构的EdgeTPU版本"""
    
    print("🏗️  创建基于经典架构的EdgeTPU深度估计模型")
    print("="*60)
    print(f"📐 使用640x480分辨率 (与感知管道一致)")
    
    models_created = []
    
    # 1. FastDepth + MobileNetV2
    print("\n1️⃣  FastDepth + MobileNetV2 (单目深度估计)")
    fastdepth = FastDepthEdgeTPU(input_height=480, input_width=640)
    fastdepth_model = fastdepth.build_fastdepth_model()
    
    print(f"📊 FastDepth模型参数: {fastdepth_model.count_params():,}")
    
    # 2. MobileNetV2 立体深度估计  
    print("\n2️⃣  MobileNetV2 + 立体匹配")
    stereo_depth = MobileNetV2StereoDepth(input_height=480, input_width=640)
    stereo_model = stereo_depth.build_stereo_model()
    
    print(f"📊 立体深度模型参数: {stereo_model.count_params():,}")
    
    # 3. MiDaS风格的MobileNet
    print("\n3️⃣  MiDaS + MobileNet (多尺度融合)")
    midas_mobilenet = MiDaSMobileNet(input_height=480, input_width=640)
    midas_model = midas_mobilenet.build_midas_mobilenet()
    
    print(f"📊 MiDaS-MobileNet参数: {midas_model.count_params():,}")
    
    # 保存模型架构图
    print(f"\n📋 保存模型架构图...")
    try:
        for model, name in [(fastdepth_model, 'FastDepth'), 
                           (stereo_model, 'StereoDepth'), 
                           (midas_model, 'MiDaS')]:
            
            tf.keras.utils.plot_model(
                model, 
                to_file=f'{name.lower()}_architecture.png',
                show_shapes=True,
                show_layer_names=True
            )
            print(f"✅ {name} 架构图已保存")
            
            models_created.append((model, name))
    except Exception as e:
        print(f"⚠️  架构图保存失败 (正常，继续): {e}")
        models_created = [(fastdepth_model, 'FastDepth'), 
                         (stereo_model, 'StereoDepth'), 
                         (midas_model, 'MiDaS')]
    
    # 转换和编译流程
    print(f"\n🔧 开始三步骤转换流程...")
    print(f"  步骤1: Keras -> TFLite")
    print(f"  步骤2: TFLite -> 量化TFLite") 
    print(f"  步骤3: 量化TFLite -> EdgeTPU")
    
    # 生成代表性数据集用于量化
    def representative_data_gen_mono():
        """单目模型的代表性数据"""
        for i in range(50):  # 简化为50个样本
            data = np.random.rand(1, 480, 640, 3).astype(np.float32)
            data = data * 0.8 + 0.1  # 调整到[0.1, 0.9]范围
            yield [data]
    
    def representative_data_gen_stereo():
        """立体模型的代表性数据"""
        for i in range(50):  # 简化为50个样本
            left_data = np.random.rand(1, 480, 640, 3).astype(np.float32)
            right_data = np.random.rand(1, 480, 640, 3).astype(np.float32)
            left_data = left_data * 0.8 + 0.1
            right_data = right_data * 0.8 + 0.1
            yield [left_data, right_data]
    
    # 处理每个模型
    edgetpu_models = []
    
    for model, name in models_created:
        print(f"\n🔄 处理 {name}...")
        
        try:
            # 步骤1: 转换为TFLite
            tflite_path = convert_to_tflite(model, name)
            
            # 步骤2: 直接从Keras模型量化为INT8 TFLite
            rep_data_gen = representative_data_gen_stereo if name == 'StereoDepth' else representative_data_gen_mono
            quantized_path = quantize_keras_to_tflite(model, name, rep_data_gen)
            
            # 步骤3: 编译为EdgeTPU
            if quantized_path:
                edgetpu_path = compile_to_edgetpu(quantized_path, name)
                
                if edgetpu_path:
                    edgetpu_models.append(edgetpu_path)
                    print(f"✅ {name} 完整流程成功")
                else:
                    print(f"❌ {name} EdgeTPU编译失败")
            else:
                print(f"❌ {name} 量化失败")
                
        except Exception as e:
            print(f"❌ {name} 处理失败: {e}")
            continue
    
    print(f"\n🎯 推荐使用顺序:")
    print(f"  1. FastDepth (最简单，论文验证) - 单目深度估计")
    print(f"  2. MobileNetV2-Stereo (立体匹配，更准确) - 立体深度估计") 
    print(f"  3. MiDaS-MobileNet (最先进，多尺度) - 单目深度估计")
    
    print(f"\n🏁 完成情况:")
    print(f"  • 模型创建: {len(models_created)}/3")
    print(f"  • TFLite转换: 检查 *.tflite 文件")
    print(f"  • 量化TFLite: 检查 *_quantized.tflite 文件")
    print(f"  • EdgeTPU编译: {len(edgetpu_models)}/3")
    
    print(f"\n📐 分辨率说明:")
    print(f"  • 输入: 640x480 (与感知管道一致)")
    print(f"  • FastDepth输出: 640x480x1 (深度图)")
    print(f"  • StereoDepth输出: 640x480x1 (视差图)")
    print(f"  • MiDaS输出: 640x480x1 (相对深度)")
    
    if edgetpu_models:
        print(f"\n✅ 编译成功的EdgeTPU模型:")
        for model in edgetpu_models:
            print(f"  • {model}")
        print(f"\n📝 编译日志文件:")
        for model in edgetpu_models:
            log_file = model.replace('_edgetpu.tflite', '_edgetpu.log')
            if os.path.exists(log_file):
                print(f"  • {log_file}")
    
    print(f"\n🎯 手动EdgeTPU编译命令（如需要）:")
    print(f"edgetpu_compiler fastdepth_quantized.tflite")
    print(f"edgetpu_compiler stereodepth_quantized.tflite") 
    print(f"edgetpu_compiler midas_quantized.tflite")
    
    return models_created, edgetpu_models

if __name__ == "__main__":
    models = main()
