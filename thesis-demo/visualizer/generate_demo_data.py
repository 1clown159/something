#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 算法演示 - 演示数据文件生成器
生成不同特性的数据文件，用于算法演示页面各步骤的可视化展示
"""

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "uploads" / "demo_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE = (2, 60, 120)
TOTAL_FLOATS = np.prod(SHAPE)

def save(data, filename, description):
    path = OUTPUT_DIR / filename
    data.astype(np.float32).tofile(path)
    size_kb = path.stat().st_size / 1024
    shape_str = " × ".join(str(s) for s in data.shape)
    print(f"  {filename}")
    print(f"    大小: {size_kb:.1f} KB | 形状: {shape_str}")
    print(f"    值范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"    说明: {description}")
    print()


# ===== 文件1: 常量数据 =====
print("=" * 60)
print("生成演示数据文件...")
print("=" * 60)
print()

data = np.full(SHAPE, 1.0, dtype=np.float32)
save(data, "demo_constant.bin",
     "所有值相同(1.0)，完全可预测。预期CNN给出单一峰值概率分布，压缩比极高(>50x)，适合验证算法最佳情况。")

# ===== 文件2: 线性渐变 =====
p, t, s = SHAPE
grid = np.zeros(SHAPE, dtype=np.float32)
for i in range(p):
    for j in range(t):
        for k in range(s):
            grid[i, j, k] = (i * 10.0 + j * 0.1 + k * 0.01)
data = grid / grid.max() * 100.0
save(data, "demo_gradient.bin",
     "三维线性递增渐变，LOCO-I预测器可高精度预测。预期残差值极小，概率分布集中在少量符号，压缩比高(10-20x)。")

# ===== 文件3: 正弦波 =====
x = np.linspace(0, 8 * np.pi, s)
sine_row = np.sin(x) * 50.0
data = np.tile(sine_row, (p, t, 1)).astype(np.float32)
noise = np.random.RandomState(42).randn(*SHAPE).astype(np.float32) * 0.01
data += noise
save(data, "demo_sine.bin",
     "周期正弦波 + 微量噪声。预期CNN可学习周期性模式，概率分布有多个峰值对应不同相位，压缩比中等(5-10x)。")

# ===== 文件4: 随机噪声 =====
rng = np.random.RandomState(42)
data = rng.randn(*SHAPE).astype(np.float32) * 10.0
save(data, "demo_random.bin",
     "高斯随机噪声，无规律性。预期概率分布接近均匀，熵接近8 bits/voxel上界，压缩比低(<2x)，展示算法最差情况。")

# ===== 文件5: 模拟地震数据 =====
rng = np.random.RandomState(2024)
data = np.zeros(SHAPE, dtype=np.float32)

for prof in range(p):
    background = rng.randn(t, s).astype(np.float32) * 0.3
    data[prof] += background

    num_reflectors = 4
    for layer in range(num_reflectors):
        layer_pos = int(s * (0.25 + 0.55 * layer / num_reflectors))
        thickness = rng.randint(2, 5)
        for tr in range(t):
            amp = rng.randn() * 4.0 + 8.0
            phase = int(rng.randn() * 2)
            for offset in range(-thickness, thickness + 1):
                si = layer_pos + offset + phase
                if 0 <= si < s:
                    x = offset / (thickness / 2.0)
                    wavelet = amp * (1 - 2 * x * x) * np.exp(-x * x)
                    data[prof, tr, si] += wavelet

    for tr in range(t):
        delay = int(abs(tr - t / 2) * 0.4)
        if delay < s:
            data[prof, tr, delay:min(delay + 5, s)] += 15.0 * np.exp(-np.arange(min(5, s - delay)))

save(data, "demo_seismic.bin",
     "模拟地震数据：背景噪声 + 多层反射波 + 直达波。最接近真实应用场景，包含多种统计特征，压缩比约3-5x。")

# ===== 汇总 =====
print("=" * 60)
print("演示数据生成完毕")
print("-" * 60)
files = sorted(OUTPUT_DIR.glob("demo_*.bin"))
total_size = sum(f.stat().st_size for f in files)
print(f"输出目录: {OUTPUT_DIR}")
print(f"文件数量: {len(files)}")
print(f"总大小: {total_size / 1024:.1f} KB")
print("-" * 60)
for f in files:
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
print("=" * 60)
