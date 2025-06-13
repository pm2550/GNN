#!/usr/bin/env python3
"""
对 models 目录下的各层量化模型进行 EdgeTPU 编译，并组织到 tpu 和 cpu 文件夹
"""

import os
import shutil
import subprocess

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

def compile_and_organize():
    """编译量化模型并组织到相应文件夹"""
    
    for layer_name, model_file in layer_models.items():
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path):
            print(f"警告：{model_path} 不存在，跳过")
            continue
            
        print(f"正在编译 {layer_name} 层模型...")
        
        # 1. 复制原量化模型到 cpu 文件夹
        cpu_name = f"{layer_name}_cpu.tflite"
        cpu_path = os.path.join("cpu", cpu_name)
        shutil.copy2(model_path, cpu_path)
        print(f"  ✓ 复制到 CPU: {cpu_path}")
        
        # 2. 用 EdgeTPU 编译器编译
        try:
            subprocess.run([
                "edgetpu_compiler", "-s", model_path
            ], capture_output=True, text=True, check=True)
            
            # 3. 查找编译后的文件（通常是原文件名加 _edgetpu）
            base_name = model_file.replace(".tflite", "")
            edgetpu_file = f"{base_name}_edgetpu.tflite"
            edgetpu_path = edgetpu_file  # 编译后的文件在当前目录
            
            if os.path.exists(edgetpu_path):
                # 4. 移动到 tpu 文件夹并重命名
                tpu_name = f"{layer_name}_tpu.tflite"
                tpu_path = os.path.join("tpu", tpu_name)
                shutil.move(edgetpu_path, tpu_path)
                print(f"  ✓ 编译并移动到 TPU: {tpu_path}")
            else:
                print(f"  ✗ 找不到编译后的文件: {edgetpu_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 编译失败: {e}")
            print(f"     stdout: {e.stdout}")
            print(f"     stderr: {e.stderr}")
        except FileNotFoundError:
            print("  ✗ 找不到 edgetpu_compiler，请确保已安装 Edge TPU Compiler")
            break
            
    print("\n完成！文件组织结果：")
    print("CPU 文件夹:")
    if os.path.exists("cpu"):
        for f in sorted(os.listdir("cpu")):
            print(f"  {f}")
    print("TPU 文件夹:")
    if os.path.exists("tpu"):
        for f in sorted(os.listdir("tpu")):
            print(f"  {f}")

if __name__ == "__main__":
    compile_and_organize()
