#!/usr/bin/env python3
"""
轻量级模型的EdgeTPU编译脚本，提供详细的编译分析
"""

import os
import shutil
import subprocess
import re

# 创建目标文件夹
os.makedirs("tpu", exist_ok=True)
os.makedirs("cpu", exist_ok=True)

# 定义轻量级层模型
layer_models = {
    "conv2d": "conv2d_light_int8.tflite",
    "depthwise_conv2d": "depthwise_conv2d_light_int8.tflite", 
    "max_pool": "max_pool_light_int8.tflite",
    "avg_pool": "avg_pool_light_int8.tflite",
    "dense": "dense_light_int8.tflite",
    "separable_conv": "separable_conv_light_int8.tflite",
    "detection_head": "detection_head_int8.tflite",
    "feature_pyramid": "feature_pyramid_int8.tflite"
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
    mapped_count = 0
    total_count = 0
    
    for line in lines:
        if 'Mapped to Edge TPU' in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                op_name = parts[0]
                count = parts[1] if parts[1].isdigit() else "1"
                mapped_ops.append(f"{op_name} ({count})")
                mapped_count += int(count) if count.isdigit() else 1
        elif any(keyword in line for keyword in ['will run on CPU', 'Operation is working on an unsupported']):
            parts = line.strip().split()
            if len(parts) >= 2:
                op_name = parts[0]
                count = parts[1] if parts[1].isdigit() else "1"
                unmapped_ops.append(f"{op_name} ({count})")
        
        # 统计总操作数
        if 'Total number of operations:' in line:
            total_count = int(line.split(':')[1].strip())
    
    return mapped_ops, unmapped_ops, mapped_count, total_count

def compile_and_organize():
    """编译量化模型并进行详细分析"""
    
    print("🚀 EdgeTPU 轻量级模型编译分析")
    print("="*60)
    
    success_count = 0
    total_models = 0
    
    for layer_name, model_file in layer_models.items():
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path):
            print(f"⏭️  {model_path} 不存在，跳过")
            continue
            
        total_models += 1
        print(f"\n🔍 正在分析 {layer_name} 层模型...")
        
        # 获取原始文件大小
        orig_size = os.path.getsize(model_path)
        
        # 1. 复制原量化模型到 cpu 文件夹
        cpu_name = f"{layer_name}_cpu.tflite"
        cpu_path = os.path.join("cpu", cpu_name)
        shutil.copy2(model_path, cpu_path)
        print(f"  📁 原始模型已复制到: {cpu_path}")
        
        # 2. 用 EdgeTPU 编译器编译
        try:
            # 切换到models目录进行编译
            result = subprocess.run([
                "edgetpu_compiler", "-s", "--timeout", "300", model_file
            ], cwd="models", capture_output=True, text=True, check=True)
            
            # 分析编译日志
            mapped_ops, unmapped_ops, mapped_count, total_count = analyze_compilation_log(result.stdout)
            
            # 3. 查找编译后的文件
            base_name = model_file.replace(".tflite", "")
            edgetpu_file = f"{base_name}_edgetpu.tflite"
            edgetpu_path = os.path.join("models", edgetpu_file)
            
            if os.path.exists(edgetpu_path):
                # 获取编译后文件大小
                tpu_size = os.path.getsize(edgetpu_path)
                size_ratio = tpu_size / orig_size
                
                # 4. 移动到 tpu 文件夹并重命名
                tpu_name = f"{layer_name}_tpu.tflite"
                tpu_path = os.path.join("tpu", tpu_name)
                shutil.move(edgetpu_path, tpu_path)
                
                success_count += 1
                
                # 打印详细分析
                print(f"  📏 文件大小:")
                print(f"    原始: {format_size(orig_size)}")
                print(f"    TPU:  {format_size(tpu_size)} (变化: {size_ratio:.1f}x)")
                
                # 显示操作映射情况
                if total_count > 0:
                    success_rate = (mapped_count / total_count) * 100
                    print(f"  🎯 EdgeTPU 映射率: {success_rate:.1f}% ({mapped_count}/{total_count})")
                
                if mapped_ops:
                    print(f"  ✅ 成功映射到 EdgeTPU:")
                    for op in mapped_ops[:5]:  # 只显示前5个
                        print(f"    - {op}")
                    if len(mapped_ops) > 5:
                        print(f"    ... 还有 {len(mapped_ops)-5} 个操作")
                
                if unmapped_ops:
                    print(f"  ⚠️  仍在 CPU 运行:")
                    for op in unmapped_ops[:3]:  # 只显示前3个
                        print(f"    - {op}")
                    if len(unmapped_ops) > 3:
                        print(f"    ... 还有 {len(unmapped_ops)-3} 个操作")
                else:
                    print(f"  🏆 100% 算子映射到 EdgeTPU！")
                    
                print(f"  🎉 编译成功: {tpu_path}")
                
            else:
                print(f"  ❌ 找不到编译后的文件: {edgetpu_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 编译失败: {e}")
            if e.stderr:
                print(f"     错误输出: {e.stderr}")
        except FileNotFoundError:
            print("  ❌ 找不到 edgetpu_compiler，请先安装 Edge TPU 编译器")
            break
            
    print(f"\n🎯 编译总结:")
    print(f"  成功编译: {success_count}/{total_models} 个模型")
    print(f"  CPU 模型位置: ./cpu/")
    print(f"  TPU 模型位置: ./tpu/")
    print(f"\n💡 性能优化建议:")
    print(f"  1. Conv2D 和 DepthwiseConv2D 通常有最好的 EdgeTPU 加速效果")
    print(f"  2. SeparableConv 在移动端推理中效率很高")
    print(f"  3. 检测头和特征金字塔模块适合多任务推理")
    print(f"  4. 100% 映射率的模型将获得最佳性能")

if __name__ == "__main__":
    compile_and_organize()
