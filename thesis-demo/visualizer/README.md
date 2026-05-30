# Stage4 数据压缩可视化平台

基于深度学习的无损数据压缩算法可视化系统，用于帮助用户理解 Stage4 压缩算法的工作原理。

## 系统架构

```
stage4_visualizer/
├── backend/              # 后端API (FastAPI)
│   ├── app.py           # 主应用
│   ├── core/
│   │   └── stage4_bridge.py  # 与Stage4算法桥接
│   └── requirements.txt
├── frontend/            # 前端界面
│   ├── index.html       # 主页面
│   ├── css/
│   │   └── style.css    # 样式 (借鉴buma.html设计)
│   └── js/
│       ├── api.js       # API通信
│       ├── visualizer.js # 可视化组件
│       └── app.js       # 主应用逻辑
├── uploads/             # 上传文件存储
├── outputs/             # 压缩结果存储
└── serve.py            # 启动脚本
```

## 功能特性

### 1. 数据上传
- 支持拖拽上传和点击上传
- 支持 .bin 和 .dat 格式文件
- 实时显示文件信息

### 2. 压缩控制
- 可配置压缩参数（特征模式、目标模式、patch大小等）
- 实时显示压缩进度
- 压缩日志输出
- 下载压缩结果

### 3. 特征可视化 (核心功能)
借鉴 buma.html 的"位展示"方式展示6通道特征图：
- **通道0**: 像素值 (Values) - 蓝色
- **通道1**: 可用掩码 (Valid Mask) - 绿色
- **通道2**: 因果掩码 (Causal Mask) - 橙色
- **通道3**: 映射掩码 (Mapped Mask) - 紫色
- **通道4**: 预测值 (Predicted) - 红色
- **通道5**: 残差值 (Residual) - 青色

特征通道展示特点：
- 网格布局展示每个通道的patch数据
- 鼠标悬停显示详细数值
- 颜色深浅表示数值大小
- 实时显示预测值、实际值和目标符号

### 4. 概率分布可视化
- 256类概率分布柱状图
- CDF累积分布函数图
- 高亮显示预测符号和实际符号
- 显示熵和理想码长信息

### 5. 压缩统计
- 文件大小对比图
- 码率统计 (bits/voxel)
- 压缩流程动画展示

## 快速开始

### 1. 安装依赖

```bash
cd stage4_visualizer/backend
pip install -r requirements.txt
```

### 2. 启动系统

```bash
# 在 stage4_visualizer 目录下
python serve.py
```

或者分别启动前后端：

```bash
# 启动后端 (终端1)
cd backend
python app.py

# 启动前端 (终端2)
cd frontend
python -m http.server 8080
```

### 3. 访问系统

打开浏览器访问: http://localhost:8080

API文档: http://localhost:8000/docs

## 使用流程

1. **上传数据**: 在"数据上传"页面上传 .bin 文件
2. **配置压缩**: 在"压缩控制"页面设置参数并开始压缩
3. **查看特征**: 在"特征可视化"页面查看6通道特征图
4. **查看概率**: 在"概率分布"页面查看预测概率
5. **查看统计**: 在"压缩统计"页面查看压缩结果

## API 接口

### 上传文件
```bash
POST /api/upload
Content-Type: multipart/form-data
file: <binary_file>
```

### 开始压缩
```bash
POST /api/compress/{task_id}
Content-Type: application/json
{
  "feature_mode": "diagonal_causal_edge",
  "target_mode": "raw",
  "patch_shape": [9, 17],
  "inference_batch": 1
}
```

### 提取特征
```bash
POST /api/features/{task_id}
Content-Type: application/json
{
  "coord": [0, 100, 100],
  "patch_shape": [9, 17],
  "feature_mode": "diagonal_causal_edge",
  "target_mode": "raw"
}
```

### 获取可视化数据
```bash
GET /api/visualize/{task_id}?data_type=input_slice|features|probabilities|cdf
```

### 获取压缩统计
```bash
GET /api/stats/{task_id}
```

## 可视化设计说明

### 6通道特征图展示 (借鉴 buma.html)

参考 `1304356629-buma.html` 的设计思路：

1. **位展示方式**: 
   - 每个像素单元格类似HTML中的 `.binary-digit`
   - 显示实际数值（格式化显示）
   - 鼠标悬停放大并显示详细tooltip

2. **分组着色**:
   - 每个通道有专属颜色（参考通道列表）
   - 颜色深浅映射数值大小（使用D3.js颜色插值）
   - 卡片边框颜色与通道颜色一致

3. **交互设计**:
   - 鼠标悬停显示详细数值和坐标
   - 动画过渡效果
   - 实时更新的信息面板

### 概率分布可视化

- 使用D3.js绘制柱状图
- 实际符号用红色高亮
- 预测符号用蓝色高亮
- 支持动画播放

### 压缩流程动画

展示6个步骤的动画流程：
1. 📊 输入数据
2. 🔍 特征提取
3. 🧠 神经网络
4. 📈 概率预测
5. 🗜️ 熵编码
6. ✅ 输出码流

## 技术栈

### 后端
- Python 3.8+
- FastAPI - Web框架
- Uvicorn - ASGI服务器
- NumPy - 数值计算
- PyTorch - 深度学习推理

### 前端
- HTML5 + CSS3
- D3.js v7 - 数据可视化
- Axios - HTTP请求
- 原生JavaScript (ES6+)

### 样式设计
借鉴 `1304356629-buma.html` 的设计风格：
- 卡片式布局
- 清晰的视觉层次
- 平滑的过渡动画
- 响应式设计

## 与Stage4算法集成

本系统通过 `stage4_bridge.py` 与现有的Stage4算法集成：

```python
# 特征提取
from stage4 import build_single_stage4_feature_causal_edge

# 概率预测
from stage4 import load_stage4_model, Small2DCNN

# 压缩编码
from codec import Stage4GlobalDiagonalRangeCodec
```

## 开发计划

- [x] 后端API框架
- [x] 前端基础结构
- [x] 6通道特征图可视化
- [x] 概率分布可视化
- [x] 压缩统计对比
- [ ] 3D数据切片查看（完整实现）
- [ ] 实时压缩进度WebSocket
- [ ] 多文件批量处理
- [ ] 用户配置保存

## 注意事项

1. 确保 `20260420` 目录中的Stage4算法模块可用
2. 需要有训练好的模型检查点文件
3. 大文件上传可能需要调整服务器配置
4. 推荐使用Chrome或Firefox浏览器

## License

MIT License

## 致谢

- Stage4算法实现参考王玺的论文实现
- 可视化设计参考 `1304356629-buma.html`
