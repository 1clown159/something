#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 Visualizer - 测试数据生成器
生成多种类型的测试文件用于验证算法
"""

import numpy as np
import os
from pathlib import Path

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent / "uploads" / "test_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据形状配置 (模拟小规模地震数据)
DEFAULT_SHAPE = (2, 100, 200)  # 2个profile, 100条trace, 200个sample

def save_float_data(data, filename):
    """保存为float32二进制文件"""
    filepath = OUTPUT_DIR / filename
    data.astype(np.float32).tofile(filepath)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  [OK] Generated: {filename}")
    print(f"       Size: {size_mb:.2f} MB")
    print(f"       Shape: {data.shape}")
    print(f"       Value range: [{data.min():.2f}, {data.max():.2f}]")
    print()
    return filepath

def generate_random_data():
    """
    测试文件1: 完全随机数据
    
    特点:
    - 高熵，无规律性
    - 预期压缩效果: 差 (可能膨胀)
    - 可视化: 6通道特征图呈现噪声模式
    
    压缩原理:
    - 神经网络难以预测随机数据
    - 概率分布接近均匀分布
    - 熵接近8 bits/voxel
    """
    print("="*60)
    print("测试文件1: 完全随机数据 (random_data.bin)")
    print("-"*60)
    print("特点: 高熵，无规律性")
    print("预期压缩效果: 差，可能膨胀")
    print("="*60)
    
    np.random.seed(42)
    data = np.random.randn(*DEFAULT_SHAPE).astype(np.float32)
    
    # 添加一些大值增加多样性
    mask = np.random.random(DEFAULT_SHAPE) > 0.95
    data[mask] *= 10
    
    return save_float_data(data, "random_data.bin")

def generate_constant_data():
    """
    测试文件2: 常量数据
    
    特点:
    - 低熵，完全可预测
    - 预期压缩效果: 极好 (压缩比>50x)
    - 可视化: 6通道特征图呈现统一的预测模式
    
    压缩原理:
    - 神经网络能完美预测常量
    - 概率分布集中在单一值
    - 熵接近0 bits/voxel
    """
    print("="*60)
    print("测试文件2: 常量数据 (constant_data.bin)")
    print("-"*60)
    print("特点: 所有值相同，完全可预测")
    print("预期压缩效果: 极好，压缩比 > 50x")
    print("="*60)
    
    # 创建常量数据
    value = 3.14159
    data = np.full(DEFAULT_SHAPE, value, dtype=np.float32)
    
    return save_float_data(data, "constant_data.bin")

def generate_gradient_data():
    """
    测试文件3: 渐变数据
    
    特点:
    - 中等熵，强规律性
    - 预期压缩效果: 好 (压缩比5-10x)
    - 可视化: 6通道特征图呈现梯度模式
    
    压缩原理:
    - LOCO-I预测器能很好地预测线性渐变
    - 残差值很小且集中
    - 熵中等，约3-4 bits/voxel
    """
    print("="*60)
    print("测试文件3: 线性渐变数据 (gradient_data.bin)")
    print("-"*60)
    print("特点: 线性渐变，有规律性")
    print("预期压缩效果: 好，压缩比 5-10x")
    print("="*60)
    
    p, t, s = DEFAULT_SHAPE
    
    # 创建3D渐变
    data = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    
    for i in range(p):
        for j in range(t):
            for k in range(s):
                # 线性渐变公式
                data[i, j, k] = (i * 10 + j * 0.1 + k * 0.01)
    
    # 归一化到合理范围
    data = data / data.max() * 100
    
    return save_float_data(data, "gradient_data.bin")

def generate_pulse_data():
    """
    测试文件4: 脉冲/方波数据
    
    特点:
    - 低熵，重复模式
    - 预期压缩效果: 好 (压缩比10-20x)
    - 可视化: 6通道特征图呈现边缘检测模式
    
    压缩原理:
    - 周期性模式可被预测
    - 边缘处残差较大，其余位置残差小
    - 适合展示因果掩码的效果
    """
    print("="*60)
    print("测试文件4: 脉冲/方波数据 (pulse_data.bin)")
    print("-"*60)
    print("特点: 周期性方波，重复模式")
    print("预期压缩效果: 好，压缩比 10-20x")
    print("="*60)
    
    p, t, s = DEFAULT_SHAPE
    data = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    
    # 创建周期性的脉冲
    period = 20
    for i in range(p):
        for j in range(t):
            for k in range(s):
                if (k // period) % 2 == 0:
                    data[i, j, k] = 10.0  # 高电平
                else:
                    data[i, j, k] = -10.0  # 低电平
    
    # 添加一些噪声
    noise = np.random.randn(*DEFAULT_SHAPE) * 0.1
    data += noise.astype(np.float32)
    
    return save_float_data(data, "pulse_data.bin")

def generate_seismic_like_data():
    """
    测试文件5: 模拟地震数据
    
    特点:
    - 中等熵，类似真实地震数据
    - 有反射层、噪声等特征
    - 预期压缩效果: 中等偏好 (压缩比3-5x)
    - 可视化: 6通道特征图呈现真实特征提取过程
    
    压缩原理:
    - 地震数据有局部相关性
    - 反射层产生强信号
    - CNN可以学习地震数据的统计特性
    """
    print("="*60)
    print("测试文件5: 模拟地震数据 (seismic_like_data.bin)")
    print("-"*60)
    print("特点: 类似真实地震数据，有反射层和噪声")
    print("预期压缩效果: 中等偏好，压缩比 3-5x")
    print("="*60)
    
    np.random.seed(2024)
    p, t, s = DEFAULT_SHAPE
    data = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    
    for profile_idx in range(p):
        # 基础噪声 (模拟背景噪声)
        noise = np.random.randn(t, s) * 0.5
        data[profile_idx] += noise
        
        # 添加反射层 (模拟地下界面)
        num_layers = 5
        for layer in range(num_layers):
            # 反射层位置
            layer_pos = int(s * (0.2 + 0.6 * layer / num_layers))
            layer_thickness = np.random.randint(3, 8)
            
            # 反射层强度随trace变化
            for trace_idx in range(t):
                amplitude = np.random.randn() * 5 + 10  # 反射强度
                phase_shift = np.random.randn() * 2  # 相位变化
                
                # 创建波形
                for sample_offset in range(-layer_thickness, layer_thickness+1):
                    sample_idx = layer_pos + sample_offset + int(phase_shift)
                    if 0 <= sample_idx < s:
                        # 使用Ricker小波形状
                        x = sample_offset / (layer_thickness / 2.0)
                        wavelet = amplitude * (1 - 2*x*x) * np.exp(-x*x)
                        data[profile_idx, trace_idx, sample_idx] += wavelet
        
        # 添加直达波 (模拟直达信号)
        for trace_idx in range(t):
            delay = int(abs(trace_idx - t/2) * 0.5)  # 随trace延迟
            if delay < s:
                data[profile_idx, trace_idx, delay:delay+5] += 20.0 * np.exp(-np.arange(5))
    
    return save_float_data(data, "seismic_like_data.bin")

def generate_correlated_data():
    """
    测试文件6: 高度相关数据 (强空间相关性)
    
    特点:
    - 低熵，强空间相关性
    - 相邻像素值非常接近
    - 预期压缩效果: 极好 (压缩比20x+)
    - 可视化: 展示6通道如何利用空间相关性
    
    压缩原理:
    - 空间冗余度高
    - 预测器可以准确预测
    - 残差值非常小
    """
    print("="*60)
    print("测试文件6: 高度相关数据 (correlated_data.bin)")
    print("-"*60)
    print("特点: 强空间相关性，相邻像素值接近")
    print("预期压缩效果: 极好，压缩比 > 20x")
    print("="*60)
    
    np.random.seed(123)
    p, t, s = DEFAULT_SHAPE
    data = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    
    # 随机起点
    data[0, 0, 0] = np.random.randn() * 10
    
    # 按扫描顺序生成高度相关的数据
    for i in range(p):
        for j in range(t):
            for k in range(s):
                if i == 0 and j == 0 and k == 0:
                    continue
                
                # 基于前一个值生成
                prev_val = 0
                count = 0
                
                if k > 0:
                    prev_val += data[i, j, k-1]
                    count += 1
                if j > 0:
                    prev_val += data[i, j-1, k]
                    count += 1
                if i > 0:
                    prev_val += data[i-1, j, k]
                    count += 1
                
                if count > 0:
                    base_val = prev_val / count
                    # 添加微小变化
                    data[i, j, k] = base_val + np.random.randn() * 0.01
    
    return save_float_data(data, "correlated_data.bin")

def generate_mixed_pattern_data():
    """
    测试文件7: 混合模式数据
    
    特点:
    - 不同区域有不同的数据特征
    - 有随机区域、渐变区域、常量区域
    - 预期压缩效果: 中等
    - 可视化: 展示算法如何处理不同特征的数据
    
    压缩原理:
    - 不同区域需要不同的预测策略
    - CNN可以学习区域特征
    """
    print("="*60)
    print("测试文件7: 混合模式数据 (mixed_pattern_data.bin)")
    print("-"*60)
    print("特点: 不同区域有不同特征")
    print("预期压缩效果: 中等，压缩比 2-4x")
    print("="*60)
    
    np.random.seed(999)
    p, t, s = DEFAULT_SHAPE
    data = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    
    for i in range(p):
        # 上半部分: 渐变
        for j in range(t // 2):
            for k in range(s):
                data[i, j, k] = k * 0.1 + j * 0.01
        
        # 下半部分: 随机
        for j in range(t // 2, t):
            for k in range(s):
                data[i, j, k] = np.random.randn() * 5
    
    return save_float_data(data, "mixed_pattern_data.bin")

def generate_documentation():
    """生成测试文件说明文档"""
    doc_path = OUTPUT_DIR / "测试文件说明.md"
    
    content = """# 测试文件说明

## 概述

本目录包含7种不同类型的测试数据文件，用于验证Stage4压缩算法在不同数据特征下的表现。

## 文件列表与预期效果

### 1. random_data.bin - 完全随机数据
- **大小**: ~0.15 MB
- **数据特征**: 高熵，无规律，符合正态分布
- **预期压缩效果**: 
  - 压缩比: < 1x (可能膨胀)
  - 熵: ~7.9 bits/voxel
  - 原因: 无法预测，概率分布均匀
- **可视化呈现**:
  - 6通道特征图: 呈现噪声模式，无明显结构
  - 概率分布: 接近均匀分布，256个柱子高度相近
  - CDF: 接近直线

### 2. constant_data.bin - 常量数据
- **大小**: ~0.15 MB
- **数据特征**: 所有值相同 (3.14159)
- **预期压缩效果**:
  - 压缩比: > 50x
  - 熵: ~0.01 bits/voxel
  - 原因: 完全可预测，只需编码一次
- **可视化呈现**:
  - 6通道特征图: 完全统一的颜色
  - 概率分布: 单一柱子 (高度100%)
  - 上下文像素: 所有值相同

### 3. gradient_data.bin - 线性渐变数据
- **大小**: ~0.15 MB
- **数据特征**: 沿三个维度线性递增
- **预期压缩效果**:
  - 压缩比: 5-10x
  - 熵: ~3 bits/voxel
  - 原因: LOCO-I预测器适合线性数据
- **可视化呈现**:
  - 6通道特征图: 呈现清晰的颜色渐变
  - 通道0 (像素值): 从左到右颜色渐变
  - 残差通道: 值很小且集中
  - 概率分布: 集中在少数几个值附近

### 4. pulse_data.bin - 脉冲/方波数据
- **大小**: ~0.15 MB
- **数据特征**: 周期性方波，周期20个样本
- **预期压缩效果**:
  - 压缩比: 10-20x
  - 熵: ~1.5 bits/voxel
  - 原因: 周期性模式可预测
- **可视化呈现**:
  - 6通道特征图: 呈现条纹模式
  - 因果掩码: 在边缘处有明显变化
  - 概率分布: 两个峰值 (正负电平)

### 5. seismic_like_data.bin - 模拟地震数据
- **大小**: ~0.15 MB
- **数据特征**: 类似真实地震数据，有反射层、噪声、直达波
- **预期压缩效果**:
  - 压缩比: 3-5x
  - 熵: ~4-5 bits/voxel
  - 原因: 有局部相关性和噪声
- **可视化呈现**:
  - 6通道特征图: 呈现真实特征提取
  - 输入切片: 可见反射层结构
  - 概率分布: 多峰分布
  - 最有实用价值，接近真实场景

### 6. correlated_data.bin - 高度相关数据
- **大小**: ~0.15 MB
- **数据特征**: 相邻像素值高度相似
- **预期压缩效果**:
  - 压缩比: > 20x
  - 熵: ~0.5 bits/voxel
  - 原因: 空间冗余度极高
- **可视化呈现**:
  - 6通道特征图: 非常平滑
  - 残差通道: 值接近0
  - 概率分布: 极窄的峰值

### 7. mixed_pattern_data.bin - 混合模式数据
- **大小**: ~0.15 MB
- **数据特征**: 上半部分渐变，下半部分随机
- **预期压缩效果**:
  - 压缩比: 2-4x
  - 熵: ~6 bits/voxel (平均)
  - 原因: 不同区域有不同压缩难度
- **可视化呈现**:
  - 6通道特征图: 上半部分平滑，下半部分噪声
  - 概率分布: 较宽的分布
  - 展示算法如何处理非均匀数据

## 使用建议

1. **验证压缩功能**: 先用 constant_data.bin 测试，应该获得极高压缩比
2. **验证可视化**: 用 gradient_data.bin 观察特征图的颜色渐变
3. **验证算法鲁棒性**: 用 random_data.bin 测试最坏情况
4. **真实场景测试**: 用 seismic_like_data.bin 模拟实际应用

## 在系统中的使用

1. 打开 http://localhost:8080
2. 选择 "数据上传" 标签
3. 拖拽或点击上传上述测试文件
4. 在 "特征可视化" 页面输入坐标 (如 0,50,50)
5. 观察6通道特征图和概率分布

## 预期观察结果

### 在6通道特征图中:
- **random_data**: 所有通道都呈现噪声
- **constant_data**: 所有通道几乎一致
- **gradient_data**: 通道0有明显的梯度
- **seismic_like**: 通道0有反射层结构

### 在概率分布中:
- **random_data**: 256个柱子高度相近
- **constant_data**: 只有一个高柱子
- **gradient_data**: 少数几个高柱子
- **seismic_like**: 多个峰值，分布不均

### 在CDF中:
- **random_data**: 接近对角线
- **constant_data**: 阶跃函数
- **gradient_data**: 平滑曲线
- **seismic_like**: 不规则曲线
"""
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Generated documentation: {doc_path}")
    return doc_path

def main():
    """主函数"""
    print("\n" + "="*60)
    print("Stage4 Visualizer - 测试数据生成器")
    print("="*60 + "\n")
    
    files = []
    
    # 生成7种测试文件
    files.append(generate_random_data())
    files.append(generate_constant_data())
    files.append(generate_gradient_data())
    files.append(generate_pulse_data())
    files.append(generate_seismic_like_data())
    files.append(generate_correlated_data())
    files.append(generate_mixed_pattern_data())
    
    # 生成说明文档
    doc_path = generate_documentation()
    
    # 汇总
    print("="*60)
    print("测试文件生成完成！")
    print("-"*60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"共生成 {len(files)} 个测试文件")
    print("-"*60)
    print("\n文件列表:")
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f.name} ({size_mb:.2f} MB)")
    print("="*60)
    
    print("\n使用方式:")
    print("1. 启动系统: python serve.py")
    print("2. 访问 http://localhost:8080")
    print("3. 在'数据上传'页面选择上述文件")
    print("4. 查看说明文档了解预期效果")
    print("="*60)

if __name__ == '__main__':
    main()
