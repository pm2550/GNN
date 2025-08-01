#!/usr/bin/env python3
"""
增强分层模型 - Conv-Stack-n 和 DW-Stack-n
基于LaTeX规格创建的模型架构
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import subprocess

def create_conv_stack_model(n_repeats, input_shape=(224, 224, 3)):
    """
    创建 Conv-Stack-n 模型
    
    参数:
        n_repeats: 重复次数 (1, 3, 5, 7)
        input_shape: 输入形状，默认 (224, 224, 3)
    
    返回:
        Keras模型
    """
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # 重复 n 次 Conv2D 3x3, 32 filters, stride 1, ReLU6
    for i in range(n_repeats):
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu6',
            name=f'conv2d_{i+1}'
        )(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name=f'conv_stack_{n_repeats}')
    return model

def create_dw_stack_model(n_repeats, input_shape=(224, 224, 32)):
    """
    创建 DW-Stack-n 模型
    
    参数:
        n_repeats: 重复次数 (1, 3, 5, 7)
        input_shape: 输入形状，默认 (224, 224, 32)
    
    返回:
        Keras模型
    """
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # 重复 n 次 DepthwiseConv2D 3x3, depth=1, stride 1, ReLU6
    for i in range(n_repeats):
        x = layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            depth_multiplier=1,
            padding='same',
            activation='relu6',
            name=f'depthwise_conv2d_{i+1}'
        )(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name=f'dw_stack_{n_repeats}')
    return model

def quantize_model(model, representative_dataset_gen):
    """
    量化模型为int8
    
    参数:
        model: Keras模型
        representative_dataset_gen: 代表性数据集生成器
    
    返回:
        量化后的TFLite模型字节
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

def create_representative_dataset(input_shape, num_samples=100):
    """
    创建代表性数据集用于量化
    
    参数:
        input_shape: 输入形状
        num_samples: 样本数量
    
    返回:
        数据生成器函数
    """
    def representative_dataset_gen():
        for _ in range(num_samples):
            # 创建随机数据，范围[0, 255]然后归一化到[0, 1]
            data = np.random.randint(0, 256, size=(1,) + input_shape, dtype=np.uint8)
            data = data.astype(np.float32) / 255.0
            yield [data]
    
    return representative_dataset_gen

def check_edgetpu_compiler():
    """
    检查Edge TPU编译器是否可用
    
    返回:
        bool: 编译器是否可用
    """
    try:
        # 尝试多个可能的编译器路径
        possible_paths = [
            'edgetpu_compiler',
            './edgetpu_compiler',
            '/usr/bin/edgetpu_compiler',
            './edgetpu_compiler_bin/edgetpu_compiler'
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0 or 'edgetpu_compiler' in result.stderr.lower():
                    print(f"找到Edge TPU编译器: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        print("警告: 未找到Edge TPU编译器，将跳过Edge TPU编译步骤")
        return None
        
    except Exception as e:
        print(f"检查Edge TPU编译器时出错: {e}")
        return None

def compile_to_edgetpu(tflite_model_path, compiler_path=None):
    """
    编译TFLite模型为Edge TPU格式
    
    参数:
        tflite_model_path: TFLite模型文件路径
        compiler_path: 编译器路径
    
    返回:
        Edge TPU模型文件路径或None
    """
    if compiler_path is None:
        print(f"跳过Edge TPU编译: {tflite_model_path}")
        return None
        
    edgetpu_model_path = tflite_model_path.replace('.tflite', '_edgetpu.tflite')
    
    try:
        # 使用Edge TPU编译器
        compile_cmd = [compiler_path, '-s', tflite_model_path]
        
        print(f"正在编译Edge TPU模型: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        if result.returncode == 0:
            print(f"Edge TPU编译成功: {edgetpu_model_path}")
            return edgetpu_model_path
        else:
            print(f"Edge TPU编译失败:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Edge TPU编译超时")
        return None
    except Exception as e:
        print(f"Edge TPU编译时出错: {e}")
        return None

def create_and_quantize_all_models():
    """
    创建所有模型变体并进行量化和Edge TPU编译
    """
    n_values = [1, 3, 5, 7]
    
    print("增强分层模型创建器")
    print("====================")
    print("开始创建和量化模型...")
    
    # 检查Edge TPU编译器
    compiler_path = check_edgetpu_compiler()
    
    for n in n_values:
        print(f"\n处理 n={n} 的模型...")
        
        # 创建Conv-Stack-n模型
        print(f"创建 Conv-Stack-{n} 模型...")
        conv_model = create_conv_stack_model(n)
        print(f"模型参数数量: {conv_model.count_params():,}")
        
        # 为Conv-Stack模型创建代表性数据集
        conv_dataset_gen = create_representative_dataset((224, 224, 3))
        
        # 量化Conv-Stack模型
        print(f"量化 Conv-Stack-{n} 模型...")
        try:
            conv_quantized = quantize_model(conv_model, conv_dataset_gen)
            
            # 保存量化的Conv-Stack模型
            conv_tflite_path = f"conv_stack_{n}_int8.tflite"
            with open(conv_tflite_path, 'wb') as f:
                f.write(conv_quantized)
            print(f"✓ 保存量化模型: {conv_tflite_path} ({len(conv_quantized):,} bytes)")
            
            # 编译为Edge TPU
            conv_edgetpu_path = compile_to_edgetpu(conv_tflite_path, compiler_path)
            if conv_edgetpu_path:
                print(f"✓ Edge TPU模型: {conv_edgetpu_path}")
                
        except Exception as e:
            print(f"✗ Conv-Stack-{n} 量化失败: {e}")
        
        # 创建DW-Stack-n模型
        print(f"创建 DW-Stack-{n} 模型...")
        dw_model = create_dw_stack_model(n)
        print(f"模型参数数量: {dw_model.count_params():,}")
        
        # 为DW-Stack模型创建代表性数据集
        dw_dataset_gen = create_representative_dataset((224, 224, 32))
        
        # 量化DW-Stack模型
        print(f"量化 DW-Stack-{n} 模型...")
        try:
            dw_quantized = quantize_model(dw_model, dw_dataset_gen)
            
            # 保存量化的DW-Stack模型
            dw_tflite_path = f"dw_stack_{n}_int8.tflite"
            with open(dw_tflite_path, 'wb') as f:
                f.write(dw_quantized)
            print(f"✓ 保存量化模型: {dw_tflite_path} ({len(dw_quantized):,} bytes)")
            
            # 编译为Edge TPU
            dw_edgetpu_path = compile_to_edgetpu(dw_tflite_path, compiler_path)
            if dw_edgetpu_path:
                print(f"✓ Edge TPU模型: {dw_edgetpu_path}")
                
        except Exception as e:
            print(f"✗ DW-Stack-{n} 量化失败: {e}")
        
        print(f"完成 n={n} 的所有模型处理")

def test_model_inference(model_path, input_shape):
    """
    测试模型推理
    
    参数:
        model_path: 模型文件路径
        input_shape: 输入形状
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n模型测试: {model_path}")
        print(f"输入形状: {input_details[0]['shape']}")
        print(f"输入类型: {input_details[0]['dtype']}")
        print(f"输出形状: {output_details[0]['shape']}")
        print(f"输出类型: {output_details[0]['dtype']}")
        
        # 创建测试输入
        if input_details[0]['dtype'] == np.uint8:
            test_input = np.random.randint(0, 256, size=input_details[0]['shape'], dtype=np.uint8)
        else:
            test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"推理成功 - 输出值范围: [{output_data.min()}, {output_data.max()}]")
        
    except Exception as e:
        print(f"模型测试失败: {e}")

if __name__ == "__main__":
    # 创建并量化所有模型
    create_and_quantize_all_models()
    
    print("\n" + "="*50)
    print("模型创建总结")
    print("="*50)
    
    # 列出生成的文件
    tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite') and ('conv_stack_' in f or 'dw_stack_' in f)]
    tflite_files.sort()
    
    print("生成的模型文件:")
    for filename in tflite_files:
        file_size = os.path.getsize(filename)
        print(f"  ✓ {filename} ({file_size:,} bytes)")
    
    # 测试一个模型的推理
    if tflite_files:
        print(f"\n测试第一个模型的推理:")
        test_model_path = tflite_files[0]
        if 'conv_stack_' in test_model_path:
            test_model_inference(test_model_path, (224, 224, 3))
        else:
            test_model_inference(test_model_path, (224, 224, 32))