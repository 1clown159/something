# Stage4 数据压缩算法演示系统

基于深度学习的无损浮点数据压缩算法（Stage4），包含核心算法模块、可视化演示平台和实验脚本。

## 项目结构

```
thesis-demo/
├── algorithm/                 # 核心算法模块
│   ├── common.py              # 通用数据结构、浮点位拆解
│   ├── stage4.py              # Stage4 CNN指数预测模型
│   ├── codec.py               # 编解码器（raster/tile64/global_diag）
│   ├── range_coder.py         # 范围编码器底层实现
│   ├── compute_aux_bits.py    # 辅助位（符号/尾数）统计
│   ├── hybrid_codec.py        # 混合编解码器
│   ├── roi.py                 # ROI区域
│   ├── tui_blocks.py          # TUI数据块处理
│   ├── run_tui_multiblock.py  # TUI多block训练入口
│   ├── run_stage4_codec.py    # 编解码器运行入口
│   ├── postdata/              # 规则三维数据处理
│   │   ├── postdata_blocks.py
│   │   ├── run_postdata_multiblock.py
│   │   └── compute_postdata_aux_bits.py
│   └── outputs_tui_*/         # 训练输出和模型检查点
├── visualizer/                # 可视化演示平台
│   ├── backend/               # FastAPI后端
│   │   ├── app.py             # 主应用
│   │   ├── requirements.txt
│   │   └── core/
│   │       └── stage4_bridge.py  # 算法桥接层
│   ├── frontend/              # 前端界面
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/
│   │       ├── api.js         # API通信
│   │       ├── app.js         # 主应用逻辑
│   │       └── visualizer.js  # 可视化组件
│   ├── uploads/               # 上传文件存储
│   ├── outputs/               # 压缩结果存储
│   ├── serve.py               # 启动脚本
│   ├── generate_test_data.py  # 测试数据生成
│   └── test_compression.py    # 压缩测试
├── scripts/                   # 辅助实验脚本
│   ├── sgy_to_dat.py          # SEG-Y转float32 dat
│   ├── tui_sgy_to_dat.py      # TUI专用SEG-Y转换
│   └── compress_dat_zstd.py   # ZSTD压缩对比
└── docs/                      # 文档
    ├── ALGORITHM_USAGE.md     # 算法使用指南
    ├── STARTUP.md             # 启动指南
    └── FIXES.md               # 修复记录
```

## 快速开始

### 启动可视化平台

```bash
cd visualizer
python serve.py
```

然后访问 http://localhost:8080

### 运行测试

```bash
cd visualizer
python test.py              # 快速测试（导入+特征提取+API）
python test_compression.py  # 真实压缩测试
```

### 生成测试数据

```bash
cd visualizer
python generate_test_data.py
```

## 核心算法

Stage4算法采用深度学习模型（Small2DCNN）对float32数据的指数位进行概率预测，结合范围编码器（Range Coder）实现无损压缩。

详细算法使用说明见 [docs/ALGORITHM_USAGE.md](docs/ALGORITHM_USAGE.md)。

## 功能特性

- **数据上传**: 支持拖拽上传 .bin/.dat 格式文件
- **压缩控制**: 可配置特征模式、目标模式、patch大小等参数
- **特征可视化**: 展示6通道特征图（借鉴buma.html位展示方式）
- **概率分布**: 256类概率分布柱状图和CDF图
- **压缩统计**: 实时压缩比、码率、时间统计

## 技术栈

- **深度学习**: PyTorch
- **后端API**: FastAPI + Uvicorn
- **前端**: HTML5/CSS3 + D3.js + Axios
- **数值计算**: NumPy
- **熵编码**: 自定义范围编码器（Numba加速）
