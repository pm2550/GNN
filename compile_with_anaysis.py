#!/usr/bin/env python3
"""
改进的编译脚本，提供详细的编译分析
"""

import os
import shutil
import subprocess
import re

# 创建目标文件夹
os.makedirs("tpu", exist_ok=True)
os.makedirs("cpu", exist_ok=True)

# 定义层名和对应的量化模型文件
layer_models = {
    "conv2d": "conv2d_heavy_int8.tflite",
    "depthwise_conv2d": "depthwise_conv2d_heavy_int8.tflite", 
    "max_pool": "max_pool_heavy_int8.tflite",
    "avg_pool": "avg_pool_heavy_int8.tflite",
    "dense": "dense_heavy_int8.tflite",
    "relu": "relu_heavy_int8.tflite"
}

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def analyze_compilation_log(log_content):
    """分析编译日志"""
    lines = log_content.split('\n')
    mapped_ops = []
    unmapped_ops = []
    
    for line in lines:
        if 'Mapped to Edge TPU' in line:
            op_info = line.strip().split()
            if len(op_info) >= 2:
                mapped_ops.append(f"{op_info[0]} ({op_info[1]})")
        elif 'Operation will run on CPU' in line:
            op_info = line.strip().split()
            if len(op_info) >= 2:
                unmapped_ops.append(f"{op_info[0]} ({op_info[1]})")
    
    return mapped_ops, unmapped_ops

def compile_and_organize():
    """编译量化模型并进行详细分析"""
    
    print("🔬 EdgeTPU 编译详细分析")
    print("="*50)
    
    for layer_name, model_file in layer_models.items():
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path):
            print(f"❌ {model_path} 不存在，跳过")
            continue
            
        print(f"\n📊 正在分析 {layer_name} 层模型...")
        
        # 获取原始文件大小
        orig_size = os.path.getsize(model_path)
        
        # 1. 复制原量化模型到 cpu 文件夹
        cpu_name = f"{layer_name}_cpu.tflite"
        cpu_path = os.path.join("cpu", cpu_name)
        shutil.copy2(model_path, cpu_path)
        
        # 2. 用 EdgeTPU 编译器编译并捕获输出
        try:
            result = subprocess.run([
                "edgetpu_compiler", "-s", model_path
            ], capture_output=True, text=True, check=True)
            
            # 分析编译日志
            mapped_ops, unmapped_ops = analyze_compilation_log(result.stdout)
            
            # 3. 查找编译后的文件
            base_name = model_file.replace(".tflite", "")
            edgetpu_file = f"{base_name}_edgetpu.tflite"
            edgetpu_path = edgetpu_file
            
            if os.path.exists(edgetpu_path):
                # 获取编译后文件大小
                tpu_size = os.path.getsize(edgetpu_path)
                size_ratio = tpu_size / orig_size
                
                # 4. 移动到 tpu 文件夹并重命名
                tpu_name = f"{layer_name}_tpu.tflite"
                tpu_path = os.path.join("tpu", tpu_name)
                shutil.move(edgetpu_path, tpu_path)
                
                # 打印详细分析
                print(f"  📏 文件大小:")
                print(f"    原始: {format_size(orig_size)}")
                print(f"    TPU:  {format_size(tpu_size)} ({size_ratio:.1f}x)")
                
                print(f"  ✅ 成功映射到 EdgeTPU:")
                for op in mapped_ops:
                    print(f"    - {op}")
                
                if unmapped_ops:
                    print(f"  ⚠️  仍在 CPU 运行:")
                    for op in unmapped_ops:
                        print(f"    - {op}")
                else:
                    print(f"  🎯 100% 算子映射到 EdgeTPU")
                    
                # 给出性能建议
                if size_ratio > 10:
                    print(f"  💡 建议: 文件增大 {size_ratio:.1f}x，主要是编译器开销")
                    print(f"      对于小模型，考虑与其他算子组合以提高效率")
                
            else:
                print(f"  ❌ 找不到编译后的文件: {edgetpu_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 编译失败: {e}")
            print(f"     错误输出: {e.stderr}")
        except FileNotFoundError:
            print("  ❌ 找不到 edgetpu_compiler")
            break
            
    print(f"\n🎯 编译完成！性能优化建议:")
    print(f"  1. Conv2D 和 DepthwiseConv2D 通常有最好的 EdgeTPU 加速效果")
    print(f"  2. 小型 Pooling 模型建议与其他算子组合使用")
    print(f"  3. Dense 层在大权重时效果较好")

if __name__ == "__main__":
    compile_and_organize()
