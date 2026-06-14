#!/usr/bin/env python3
"""
Stage4 Visualizer 启动脚本
自动安装依赖并启动服务
"""

import subprocess
import sys
import os

def install_dependencies():
    """安装必要的依赖"""
    print("[*] 检查依赖...")
    
    deps = ["fastapi", "uvicorn", "python-multipart", "numpy", "torch", "pydantic"]
    
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"  [OK] {dep}")
        except ImportError:
            print(f"  [INSTALL] {dep}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    
    print("[*] 依赖检查完成")

def start_backend():
    """启动后端服务"""
    print("[*] 启动后端服务...")
    
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    
    # 启动后端
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=backend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print(f"[+] 后端服务已启动 (PID: {backend_proc.pid})")
    print("[+] API地址: http://localhost:8000")
    
    return backend_proc

def start_frontend():
    """启动前端服务"""
    print("[*] 启动前端服务...")
    
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        cwd=frontend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print(f"[+] 前端服务已启动 (PID: {frontend_proc.pid})")
    print("[+] 前端地址: http://localhost:8080")
    
    return frontend_proc

def main():
    print("="*60)
    print("Stage4 数据压缩可视化平台")
    print("="*60)
    
    # 安装依赖
    install_dependencies()
    
    # 启动服务
    backend = None
    frontend = None
    
    try:
        backend = start_backend()
        frontend = start_frontend()
        
        print("\n" + "="*60)
        print("服务已启动!")
        print("-"*60)
        print("前端界面: http://localhost:8080")
        print("后端API:  http://localhost:8000")
        print("API文档:  http://localhost:8000/docs")
        print("-"*60)
        print("按 Ctrl+C 停止服务")
        print("="*60 + "\n")
        
        # 保持运行
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] 正在停止服务...")
        if backend:
            backend.terminate()
        if frontend:
            frontend.terminate()
        print("[+] 服务已停止")

if __name__ == "__main__":
    main()
