# Stage4 Visualizer - 启动指南

## 解决 ERR_CONNECTION_ABORTED 错误

这个错误通常是由于以下原因导致的：
1. 后端服务没有启动
2. 后端依赖没有安装
3. CORS配置问题

## 快速启动

### 方式1：使用启动脚本（推荐）

```bash
cd stage4_visualizer
python start.py
```

这会：
1. 自动安装所需依赖
2. 启动后端服务 (http://localhost:8000)
3. 启动前端服务 (http://localhost:8080)

### 方式2：手动启动

**步骤1：安装依赖**
```bash
cd backend
pip install fastapi uvicorn python-multipart numpy torch pydantic
```

**步骤2：启动后端**
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**步骤3：启动前端（新终端）**
```bash
cd frontend
python -m http.server 8080
```

### 方式3：直接启动（当前目录）

```bash
cd stage4_visualizer/backend
python app.py
```

## 测试连接

启动服务后，访问：
- 前端: http://localhost:8080
- 连接测试: http://localhost:8080/test.html
- API文档: http://localhost:8000/docs

## 常见问题

### 1. Connection Aborted 错误

**原因**: 后端服务未启动

**解决**:
```bash
# 检查8000端口是否被占用
netstat -ano | findstr :8000

# 如果被占用，修改backend/app.py最后一行的端口
uvicorn.run(app, host="0.0.0.0", port=8001)  # 改为8001
```

### 2. CORS错误

**原因**: 跨域配置问题

**解决**: 已修复，确保后端app.py中有：
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
```

### 3. 上传大文件失败

**原因**: 请求体大小限制

**解决**: 已配置，如需更大文件请修改后端。

## 功能特性

1. **文件上传** - 支持.bin/.dat格式
2. **实时压缩进度** - WebSocket推送
3. **无损解压** - 验证数据完整性
4. **深色主题UI** - 现代化设计

## API端点

- `POST /api/upload` - 上传文件
- `POST /api/compress/{task_id}` - 开始压缩
- `POST /api/decompress/{task_id}` - 开始解压
- `WS /ws/progress/{task_id}` - 实时进度
- `GET /api/download/{task_id}` - 下载结果

## 注意事项

1. 确保后端服务先于前端访问启动
2. 如果使用不同端口，修改前端js/app.js中的BACKEND_URL
3. 首次启动需要安装依赖，可能需要几分钟
