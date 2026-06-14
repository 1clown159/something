# Stage4 地震数据压缩系统——项目完整介绍

> 基于深度学习的无损浮点数据压缩系统  
> 适用场景：地震 SEG-Y 数据压缩 | 论文答辩 | 算法演示

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心算法详解](#3-核心算法详解)
4. [功能页面详解](#4-功能页面详解)
   - [首页 (index.html)](#41-首页-indexhtml)
   - [3D 算法演示 (demo.html)](#42-3d-算法演示-demohtml)
   - [2D 算法演示 (demo2d.html)](#43-2d-算法演示-demo2dhtml)
   - [压缩工具 (compress.html)](#44-压缩工具-compresshtml)
   - [通用压缩 (general-compress.html)](#45-通用压缩-general-compresshtml)
   - [SEG-Y 查看器 (sgy-viewer.html)](#46-seg-y-查看器-sgy-viewerhtml)
5. [全局颜色系统](#5-全局颜色系统)
6. [快速启动](#6-快速启动)
7. [技术实现要点](#7-技术实现要点)

---

## 1. 项目概述

### 1.1 项目背景

地震勘探会产生海量的浮点数据（SEG-Y 格式），一个典型的三维工区可能包含数百 GB 甚至 TB 级的 float32 采样数据。传统压缩算法（如 zstd、lzma）对此类数据压缩率有限，因为它们无法利用浮点数据内部的结构化信息。

**Stage4** 提出了一种**基于深度学习的无损浮点数据压缩方法**：将 float32 按 IEEE 754 标准分解为符号位（1 bit）、指数位（8 bits）和尾数位（23 bits），利用空间因果卷积神经网络（Small2DCNN）预测指数位的概率分布，结合算术编码（Range Coder）实现接近理论熵界的无损压缩。

### 1.2 核心贡献

| 贡献 | 说明 |
|------|------|
| **三通道分解策略** | 将 float32 按位特性分离处理——可预测的指数用 CNN 建模，高度随机的符号和尾数用传统方法 |
| **因果约束** | 对角线扫描顺序保证编码器和解码器使用完全相同的因果上下文，确保无损压缩的数学可逆性 |
| **端到端可视化** | 5 步 3D/2D 算法演示，4 种通用压缩对比，SEG-Y 全流程分析 |
| **小体积实时演示** | 2×2×100 采样点截取 + 全坐标概率预计算，实现秒级实时交互 |

### 1.3 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| 深度学习 | PyTorch（Small2DCNN） | 指数符号概率预测 |
| 熵编码 | 自研 Range Coder（Numba JIT 加速） | 算术编码/解码 |
| 后端 | FastAPI + Uvicorn | REST API（20+ 端点）+ WebSocket 进度推送 |
| 数值计算 | NumPy | 数组操作、位分解、统计分析 |
| 3D 渲染 | Three.js v0.128 + Bloom 后处理 | 算法演示 3D 可视化 |
| 2D 渲染 | Canvas 2D API | 热力图、波形图、概率柱状图 |
| 前端交互 | Axios + 原生 ES6 | HTTP 通信与状态管理 |
| 传统压缩 | zlib / zstd / lzma | 符号位和尾数位压缩 |
| SGY 解析 | 自研纯 Python 解析器 | 无需 segyio 依赖 |

---

## 2. 系统架构

### 2.1 目录结构

```
thesis-demo/
├── algorithm/                       # 核心算法模块
│   ├── common.py                    # 数据结构、配置、位分解工具
│   ├── stage4.py                    # Small2DCNN 模型、特征构建、训练
│   ├── codec.py                     # Stage4 + Range 编解码器 (1915 行)
│   ├── range_coder.py              # 算术编码器 (Numba JIT, 487 行)
│   ├── compute_aux_bits.py        # 符号/尾数辅助压缩统计
│   ├── demo_pipeline.py            # 小体积分步处理器 (新增)
│   └── outputs_tui_smoke/          # 训练输出与 checkpoints
│
├── visualizer/                      # 可视化平台
│   ├── backend/
│   │   ├── app.py                   # FastAPI 应用 (20+ 端点)
│   │   └── core/
│   │       ├── stage4_bridge.py     # 算法桥接层 (784 行)
│   │       ├── sgy_parser.py        # 纯 Python SGY 解析器
│   │       ├── sgy_extractor.py    # SGY 浮点提取 (IBM/IEEE 转换)
│   │       └── demo_pipeline.py    # 小体积分步处理器
│   ├── frontend/
│   │   ├── index.html               # 首页
│   │   ├── demo.html                # 3D 算法演示
│   │   ├── demo2d.html              # 2D 算法演示
│   │   ├── compress.html            # 压缩工具
│   │   ├── general-compress.html    # 通用压缩
│   │   ├── sgy-viewer.html          # SEG-Y 查看器
│   │   ├── css/                     # 模块化 CSS（9 个文件）
│   │   └── js/                      # JavaScript 模块（20+ 文件）
│   ├── serve.py                     # 一键启动脚本
│   └── uploads/ / outputs/          # 上传与结果存储
│
├── docs/                            # 文档
│   ├── PAGE_GUIDE.md               # 页面功能详解
│   ├── ALGORITHM_USAGE.md          # 算法使用指南
│   ├── 开发总结.md                  # 开发历程总结
│   └── PROJECT_INTRO.md            # 本文档
│
└── find_postdata_motivation_rois.py # 参考可视化（matplotlib）
```

### 2.2 数据流概要

```
SEG-Y 文件
    ↓ [SGY 解析器: EBCDIC头 + 二进制头 + 道头 + float32数据]
Float32 数据体 (N_profiles × M_traces × K_samples)
    ↓ [位分解: u32 = view(np.uint32)]
┌────────┬────────┬─────────┐
│ 符号位  │ 指数位  │ 尾数位   │
│ 1 bit   │ 8 bits  │ 23 bits  │
└────────┴────────┴─────────┘
    │        │         │
    ↓        ↓         ↓
 zlib     Stage4    zlib
 压缩     CNN预测    压缩
          +Range
    │        │         │
    ↓        ↓         ↓
sign.zlib .s4rc   mant.zlib
    └────────┼────────┘
             ↓
     总压缩大小 = s4rc + sign + mant
     压缩比 = 原始 / 总压缩
```

---

## 3. 核心算法详解

### 3.1 IEEE 754 Float32 位结构

每个 float32 值由 32 位组成：

```
Bit 31     Bit 30-23       Bit 22-0
┌─────┬────────────────┬──────────────────┐
│ Sign│   Exponent     │    Mantissa      │
│ 1bit│    8 bits      │    23 bits       │
└─────┴────────────────┴──────────────────┘
   🔴         🟠                🔵
 红色        琥珀色            青色

数值 = (-1)^sign × 2^(exponent-127) × 1.mantissa
```

**为什么要分开处理？** 三个位的统计特性截然不同：
- **符号位**（1 bit）：高度随机，±1 各约 50%，无空间相关性 → zlib 通用压缩即可
- **指数位**（8 bits）：有空间相关性——相邻采样点振幅相近，指数值有规律可循 → **CNN 预测建模**
- **尾数位**（23 bits）：高位有微弱模式，低位接近均匀噪声 → zlib 通用压缩

### 3.2 Small2DCNN 网络结构

```
输入: 17×17×C 因果特征 Patch
  ↓
Conv2D(3×3, in_channels=2→16)
  ↓ ReLU
Conv2D(3×3, 16→16)
  ↓ ReLU
Conv2D(3×3, 16→16)
  ↓ ReLU
AdaptiveAvgPool2d(1) → Flatten
  ↓
Linear(16→256)
  ↓
输出: 256 类概率分布 (softmax)
```

只有 **3 层卷积**，参数量极小（约 5000 个参数），推理速度极快——单个采样点在 GPU 上约 0.1μs。

### 3.3 因果特征提取

**因果约束（Causal Constraint）**：编码当前点时，只能使用"已经编码过的"邻居作为 CNN 输入。

```
对角线扫描顺序示意：
   s→
t  ▓▓▓▓▓▓░░    ▓ = 已编码 (Causal)
↓  ▓▓▓▓▓░░░    ■ = 当前点
   ▓▓▓▓░░░░    ░ = 未编码 (Future)
   ▓▓▓░░░░░
```

**为什么必须保证因果？** 解码器也只能看到已解码的点。如果 CNN 输入了未编码点，解码器将无法复现编码器的预测。

**对角线扫描** 在 profile-trace 平面上沿 `diagonal = p + t + s` 递增的顺序遍历，保证每个点的"上游"邻居都已被处理。

### 3.4 LOCO-I 预测器

在残差模式（`target_mode=residual`）下，使用 LOCO-I 预测器计算预测值：

```python
prediction = clamp(left + up - up_left, 0, 255)
```

这是一个简单的二维线性预测器，取自 JPEG-LS 标准。实际指数值减去预测值得到残差（范围 -255 到 255），取模 256 后变为 0-255 的符号。CNN 预测残差符号的概率分布。

### 3.5 Range Coding（算术编码）

将 256 类概率分布量化为 CDF（累积分布函数），Range Encoder 根据符号的概率区间递归缩窄编码范围：

```
步骤:
1. CDF[symbol] = Σ prob[0..symbol]  (量化到 32768 级)
2. range = high - low + 1
3. low  += range × CDF[symbol] / total
4. high  = low + range × prob[symbol] / total - 1
5. 输出共同的前导 bit

概率越高 → 区间越宽 → 缩窄后需要 bit 越少 → 压缩越好
```

**关键性质**：这是一个**无损**编码——编码和解码完全可逆。CNN 预测越准，编码越高效。

### 3.6 三通道总体压缩比

```
总压缩大小 = Stage4指数(.s4rc) + 符号(zlib) + 尾数(zlib)

典型结果 (3D-Waipuku, 1000道):
  原始: 363.0 MB
  .s4rc: 85.3 MB  (指数, 70%)
  sign.zlib: 12 KB  (符号, 0.01%)
  mant.zlib: 37.3 MB (尾数, 30%)
  总压缩: 122.7 MB
  压缩比: 2.96:1
```

---

## 4. 功能页面详解

### 4.1 首页 (index.html)

**访问**: `http://localhost:8080`

系统门户页面，采用卡片式布局展示 5 个功能入口：

| 卡片 | 图标 | 功能 |
|------|------|------|
| 压缩工具 | 📦 | 上传 SGY → CNN 压缩/解压 → 结果分析 |
| 3D 算法演示 | 🔬 | 5 步 3D WebGL 可视化解构压缩全流程 |
| 2D 算法演示 | 🎨 | 4 步 Canvas 2D 算法演示，支持真实数据 |
| 通用压缩 | 📦 | LZ77/RLE/Deflate/Neural 四种算法对比 |
| SEG-Y 查看器 | 📡 | 元数据解析 + 热力图 + 波形 + 位统计 |

每个卡片悬停时有渐变发光效果和上浮动画。

---

### 4.2 3D 算法演示 (demo.html)

> **核心技术**：Three.js 3D 渲染 + Bloom 后处理 + 真实数据分步查询

用户上传 SEG-Y 文件后，系统自动截取 `2×2×100` 小体积数据（约 8KB），预加载 Small2DCNN 模型并预计算全部 400 个坐标点的概率分布。每一步向后端实时查询算法中间结果，3D 场景即时渲染。

**布局**：左侧控制面板 + 中央 3D 画布 + 右侧原理公式面板 + 底部翻页按钮

---

#### 步骤 1：Bit 拆解

**展示内容**：32 个 3D 立方体沿 X 轴排列，从左到右对应 Bit 31 → Bit 0。

**颜色与含义**：

| 颜色 | 十六进制 | 位段 | 位数 | 亮/暗含义 |
|------|---------|------|------|----------|
| 🔴 红色 | `#ef4444` | 符号位 Sign（Bit 31） | 1 | 亮=负数, 暗=正数 |
| 🟠 琥珀 | `#f59e0b` | 指数位 Exponent（Bit 30-23） | 8 | 亮=1, 暗=0 |
| 🔵 青色 | `#06b6d4` | 尾数位 Mantissa（Bit 22-0） | 23 | 亮=1, 暗=0 |

**交互**：悬停放大 + 波纹效果；点击输出该 Bit 信息；三色括号框分组；下方浮动 IEEE 754 公式。

**表达意义**：直观展示 float32 在内存中的位布局，说明 Stage4 为什么只对中间的 8 个指数位做 CNN 压缩——它们是"可预测"的。

---

#### 步骤 2：特征提取

**展示内容**：10×10×10 体素网格模拟三维数据空间，对角扫描线模拟编码顺序，右侧 17×17 Causal Patch 平面。

**颜色与含义**：

| 颜色 | 含义 |
|------|------|
| 🔵 青色 `#06b6d4` | 已编码体素——数据已知，可作为因果上下文 |
| 🔴 粉色 `#ec4899` | 当前体素——正在编码/解码的点 |
| ⚪ 灰色 `#9ca3af` | 未编码体素——数据未知，不可作为上下文 |

**Patch 平面中的颜色**：粉色中心格 = 当前点，青色格 = 因果可用，灰色格 = 不可用。

**表达意义**：因果约束是 Stage4 无损压缩的数学核心——保证编码器与解码器使用完全一致的信息。对角线顺序是实现因果性的关键技术。

---

#### 步骤 3：CNN 预测

**展示内容**：256 根概率柱沿弧形排列，柱高与概率成正比；粉色 Beacon 光柱标记真实 Symbol；左侧层叠方块展示 CNN 网络结构；粒子从 CNN 流向概率柱。

**颜色与含义**：

| 元素 | 颜色 |
|------|------|
| 概率柱 | 🌈 彩虹连续色阶（HSL 0→360 映射 0→255） |
| 真实 Symbol | 🔴 粉色 `#ec4899` |
| 概率曲面 | 🔵 靛蓝 `#5b6af0` |
| CNN 层 | 🔵 青色→深蓝渐变 |

**CNN 结构模型**（层叠方块，从上到下）：

| 层级 | 颜色 |
|------|------|
| Conv2D 3×3 | 青色 `#06b6d4` |
| ReLU | 深青 `#0891b2` |
| Conv2D 3×3 | 蓝灰 `#0e7490` |
| Conv2D 3×3 | 深蓝 `#155e75` |
| GAP → FC 256 | 靛蓝 `#1e3a8a` |

**表达意义**：CNN 输出的是"每个指数值有多大概率"的概率分布。理想情况下，真实符号的概率应最高。CNN 越准 → Range Coder 越高效 → 压缩越好。

---

#### 步骤 4：Range Coding

**展示内容**：主概率条（256 段彩色，宽度=概率）→ 粉色箭头指向当前 Symbol → 三层递归隧道模拟区间缩窄 → 底部 20 个 bitstream 方块。

**颜色与含义**：

| 元素 | 颜色 |
|------|------|
| 主概率条分段 | 🌈 HSL 连续色阶 |
| 当前 Symbol 段 | 🔴 粉色 `#ec4899` |
| bit=1 方块 | 🟢 绿色 `#10b981`（发光） |
| bit=0 方块 | ⚪ 灰色 `#9ca3af` |

**公式标注**（3D 文字悬浮）：
```
range = high - low + 1
low  += range × cdf[symbol]
range = range × prob[symbol]
```

**表达意义**：Range Coder 是算术编码的实现。概率越高，区间越宽，缩窄后需要的输出 bit 越少——揭示 CNN 预测与压缩率的直接关联。

---

#### 步骤 5：压缩汇总

**展示内容**：3D 甜甜圈图展示三通道占比 + 管道流程图展示数据流向 + 轨道环绕指标 + 脉冲光环。

**甜甜圈颜色**：

| 弧段 | 颜色 | 含义 |
|------|------|------|
| 🟠 橙色 | `#f59e0b` | 指数字节数（Stage4 CNN） |
| 🔴 红色 | `#ef4444` | 符号字节数（zlib） |
| 🔵 青色 | `#06b6d4` | 尾数字节数（zlib） |

**管道流程图**（使用真实统计数据动态生成）：
```
Float32 → Split → Stage4 + Sign + Mant = Total
363KB      |      85KB    12B   37KB    123KB
 (灰)     (灰)    (橙)     (红)   (青)    (绿)
```

**表达意义**：直观展示三通道分别压缩后的贡献——指数占 70%，尾数占 30%，符号几乎可忽略。

---

#### 演示交互

| 操作 | 效果 |
|------|------|
| 加载 SEG-Y | 上传 → 自动截取 2×2×100 → 预计算 400 点概率 → 就绪 |
| 步骤切换 | 点击标签 / 键盘 ← → / 底部按钮 |
| 播放 | 自动遍历 0→399 采样点，实时刷新当前步骤 |
| 暂停 / 空格 | 停止播放 |
| 速度调节 | 1-10 级滑块 |

---

### 4.3 2D 算法演示 (demo2d.html)

> Canvas 2D 像素级渲染，浅色主题，支持真实数据

与 3D 演示共享同一套后端 API（`SmallVolumeProcessor`），但使用 Canvas 2D 绘制。包含 4 个步骤：

**步骤 1 - Bit 拆解**：32 个像素方块 + 三色分组 + 发光阴影效果

**步骤 2 - 特征提取**：伪 3D 体素旋转投影 + 17×17 Causal Patch + 当前坐标标记

**步骤 3 - CNN 预测**（增强版）：
- 256 概率柱使用**幂次放大**（`pow(prob/max, 0.35)`），使小概率值可见
- Viridis 渐变色阶
- 信息框显示真实 Symbol + 概率 + 熵 + Top5 列表
- 左侧 CNN 架构图

**步骤 4 - Range Coding**：概率条 + 放大缩窄视图 + bitstream + 真实编码区间和累计 bit 数

---

### 4.4 压缩工具 (compress.html)

> 完整压缩/解压工作流

**三步流程**：

| 步骤 | 内容 |
|------|------|
| 1. 数据上传 | 拖拽或点击上传 .sgy/.segy/.bin/.dat |
| 2. 执行压缩 | 配置 6 项 CNN 参数，点击开始 |
| 3. 结果分析 | 压缩比、大小对比条、下载、解压验证 |

**压缩参数配置**：

| 参数 | 可选值 | 推荐 |
|------|--------|------|
| 特征模式 | diagonal_causal_edge / causal_edge | diagonal_causal_edge |
| 目标模式 | residual / raw | residual |
| Patch 高度 | 3-21（奇数） | 9 |
| Patch 宽度 | 3-33（奇数） | 17 |
| 批次大小 | 1-32768 | GPU: 8192, CPU: 1 |
| 计算设备 | GPU(CUDA) / CPU | GPU |

**压缩产出文件**（4 个）：

| 文件 | 内容 | 大小（典型） |
|------|------|------------|
| `compressed.s4rc` | Stage4 CNN + Range 编码的指数数据 | ~85MB |
| `sign.zlib` | zlib 压缩的符号位 | ~12KB |
| `mant.zlib` | zlib 压缩的尾数位 | ~37MB |
| `sgy_headers.json` | SGY 头文件元数据 | ~5KB |

**解压**：支持独立解压（上传 .s4rc + 可选 sign/mant/headers）和任务内验证（使用压缩时的头文件重建完整 SGY）。

---

### 4.5 通用压缩 (general-compress.html)

> 对比通用算法的局限性，突显 Stage4 的优势

**四种算法**：

| 算法 | 类型 | 核心原理 |
|------|------|---------|
| LZ77 + Huffman | 字典+熵编码 | 滑动窗口查找重复 → Huffman 编码 |
| RLE | 行程编码 | 连续重复用 (值, 次数) 表示 |
| Deflate | 标准实现 | Python zlib（LZ77+Huffman） |
| Neural Predictor | 神经网络 | MLP 预测下一字节 |

**两种模式**：

| 模式 | 说明 |
|------|------|
| 文本对比 | 输入文本 → 4 算法同时压缩 → 横向对比压缩比，最佳标记 🏆 |
| 文件压缩 | 上传文件 → 选一种算法 → 压缩 → 下载 |

**设计意图**：通用算法无法利用浮点数据的位结构信息。与 Stage4 的 2.96:1 相比，通用算法对同一 SGY 文件的压缩率通常只有 1.1-1.3:1。

---

### 4.6 SEG-Y 查看器 (sgy-viewer.html)

> 地震数据全流程分析工具

**四步分析**：

#### 步骤 1：文件解析

上传 .sgy 文件 → 自动扫描 EBCDIC 文本头（3200 字节）+ 二进制头（400 字节）+ 所有道头 → 显示元数据表格：

| 字段 | 说明 |
|------|------|
| 文件大小 | MB |
| 数据格式 | IEEE Float / IBM Float / 整型 |
| 道数 | 总地震道数 |
| 每道采样点 | 每道深度/时间采样数 |
| 采样间隔 | μs |
| 剖面数 | Inline 剖面数 |
| 每剖面道数 | Crossline 道/剖面 |

#### 步骤 2：剖面热力图

Canvas 像素级热力图，每道 2px 宽 × 每采样 1px 高。色阶为**蓝 → 白 → 红**发散型。

**交互**：
- 悬停：浮动 tooltip 显示道号 + 样点号 + 振幅值
- 点击：跳转到单道波形

#### 步骤 3：单道波形

Canvas 折线图 + 半透明蓝色填充 `#5b6af0`，浅灰背景网格，零值参考线。

#### 步骤 4：位统计分析

三个子标签，全文件向量化扫描：

| 标签 | 图表 |
|------|------|
| 指数位 | 256 bin 直方图 + 熵值 + 均值空间热力图 |
| 符号位 | 蓝(正)/红(负)/灰(零) 饼图 + 符号空间热力图 |
| 尾数位 | 双 Y 轴位平面熵图（蓝柱=每 bit 熵，红线=累计熵）|

#### 局部位统计观测（灰度图）

在灰度分析视图中：

| 功能 | 说明 |
|------|------|
| 灰度图 | 黑白灰阶振幅图像 |
| 观测框 | 橙色 30×40 方框跟随鼠标，点击固定 |
| 局部放大 | 灰度图（仿 matplotlib `cmap="gray"`） |
| 符号位分布 | `#F4F4F2`(0) / `#0B79B4`(1) |
| 指数位分布 | Viridis 色阶 + 3 条可拖拽纵线 |
| 尾数位分布 | Viridis 色阶 |
| 振幅曲线 | 三条纵线位置的振幅随样点变化 |
| 尾数位平面 | 高 8 位 + 低 8 位 bit-plane 网格 |

**指数位纵线**：3 条虚线（`#E53935` 红 / `#00B7FF` 蓝 / `#FF8F00` 橙），可拖拽。拖拽后振幅曲线实时更新。

**尾数位平面**：取中间道 × 40 连续浮点值，每 bit 用白/灰/琥珀着色。
- `#FFFFFF` = 0
- `#E5E5E5` = 1（非 managed nibble）
- `#FCD69A` = 1（managed nibble，高位 m22-m19 / 低位 m3-m0）

---

## 5. 全局颜色系统

### 5.1 系统主题色

| 颜色 | 十六进制 | CSS 变量 | 用途 |
|------|---------|---------|------|
| 主色 | `#5b6af0` | `--primary-color` / `--accent` | 按钮、链接、强调 |
| 主色渐变 | `#5b6af0→#8b5cf6` | `--primary-gradient` | 主要按钮背景 |
| 背景 | `#f8f9fb` | `--bg-dark` | 页面背景 |
| 卡片 | `rgba(255,255,255,0.85)` | `--bg-card` | 玻璃面板 |
| 主文字 | `#1a1a2e` | `--text-primary` | 正文 |
| 次文字 | `rgba(26,26,46,0.65)` | `--text-secondary` | 辅助说明 |
| 边框 | `rgba(0,0,0,0.08)` | `--glass-border` | 面板边框 |

### 5.2 算法语义色

| 颜色 | 十六进制 | 语义 |
|------|---------|------|
| 🔴 红色 | `#ef4444` | 符号位 / 错误 / 负数 |
| 🟠 琥珀 | `#f59e0b` | 指数位 / 警告 |
| 🔵 青色 | `#06b6d4` | 尾数位 / 已编码 / 因果可用 |
| 🔴 粉色 | `#ec4899` | 当前点 / 真实 Symbol / 高亮 |
| 🟢 绿色 | `#10b981` | 成功 / bit=1 |
| 🔵 蓝紫 | `#5b6af0` | 系统主色 / 正数符号 |
| ⚪ 浅灰 | `#9ca3af` | 未编码 / bit=0 / 网格 |
| ⚫ 深灰 | `#0f172a` | 3D 文字 |

### 5.3 尾数位平面着色

| 颜色 | 含义 |
|------|------|
| `#FFFFFF` | bit = 0 |
| `#E5E5E5` | bit = 1（普通位） |
| `#FCD69A` | bit = 1（managed nibble） |

### 5.4 指数纵线颜色

| 颜色 | 十六进制 |
|------|---------|
| L1 | `#E53935` 红色 |
| L2 | `#00B7FF` 亮蓝 |
| L3 | `#FF8F00` 橙色 |

---

## 6. 快速启动

### 6.1 环境要求

- Python ≥ 3.9（含 conda 环境推荐）
- PyTorch（CUDA 可选）
- 浏览器 Chrome 99+ / Edge 99+ / Firefox 112+

### 6.2 一键启动

```powershell
cd C:\Users\32599\Desktop\thesis-demo\visualizer
python serve.py
```

自动完成：
1. 检查并安装依赖（fastapi, uvicorn, numpy, torch, pydantic）
2. 启动后端 API 服务 → `http://localhost:8000`（含 API 文档 `/docs`）
3. 启动前端静态服务 → `http://localhost:8080`
4. 自动打开浏览器

### 6.3 手动启动

```powershell
# 终端 1 - 后端
cd visualizer\backend
python app.py

# 终端 2 - 前端
cd visualizer\frontend
python -m http.server 8080
```

### 6.4 首次使用流程

1. 访问 `http://localhost:8080`
2. 点击「压缩工具」→ 上传 SGY 文件 → 配置参数 → 开始压缩
3. 点击「3D 算法演示」→ 上传 SGY → 等待初始化 → 浏览 5 个步骤
4. 点击「SEG-Y 查看器」→ 上传 SGY → 查看热力图 → 切换到灰度分析 → 使用观测框

---

## 7. 技术实现要点

### 7.1 前端架构

```
页面加载
  ↓
variables.css (全局变量)
  ↓
layout.css (导航、侧栏、内容区)
  ↓
components.css (按钮、表单、卡片)
  ↓
页面专属 CSS (demo.css / compress.css / ...)
  ↓
utils.js (共享工具函数)
  ↓
api.js (HTTP 通信层: Stage4API 类)
  ↓
页面专属 JS (demo-app.js / compress-app.js / ...)
```

### 7.2 3D 渲染管线

```
SceneManager (Three.js 场景 + 相机 + 光照 + Bloom)
  ↓
StepRenderer (步骤注册/切换/生命周期)
  ↓
Step 0: Step1BitDecomposer  → 32 个 3D 立方体
Step 1: Step2CausalPatch     → 10×10×10 体素 + 17×17 Patch
Step 2: Step3CNNPredict      → 256 概率柱 + CNN 结构
Step 3: Step4RangeCoding     → 概率条 + 隧道 + bitstream
Step 4: Step6Dashboard       → 甜甜圈 + 管道 + 光环
```

### 7.3 后端分步演示架构

```
用户上传 SGY → SmallVolumeProcessor
  ├── _extract_small_volume()    提取 2×2×100
  ├── _load_model()             加载 checkpoint
  ├── _precompute_all_probs()   预计算 400 点全部概率
  └── 6 个 API 端点:
      POST /api/demo/decompose   → Bit 拆解
      POST /api/demo/features    → 特征提取
      POST /api/demo/predict     → CNN 预测
      POST /api/demo/encode      → Range 编码
      GET  /api/demo/stats       → 压缩统计
      POST /api/demo/upload-sgy  → 初始化
```

### 7.4 关键技术决策

| 决策 | 原因 |
|------|------|
| Canvas 替代 SVG 渲染热力图 | SVG 在数千个元素时卡顿，Canvas 像素级操作极快 |
| 纯 Python SEG-Y 解析器 | 消除 segyio C 库依赖，跨平台零配置 |
| 小体积分步演示 | 400 点全预计算，API 响应毫秒级，彻底解决大数据演示卡顿 |
| Three.js Bloom 后处理 | 3D 元素发光效果，增强科技感 |
| CSS 变量系统 | 一次切换全局主题，浅色/暗色兼容 |
| Numba JIT Range Coder | 纯 Python 编码性能接近 C，无外部依赖 |

---

> **文档生成时间**: 2026-06-13  
> **适用于**: 论文答辩、项目汇报、新人上手
