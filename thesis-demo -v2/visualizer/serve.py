#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 Visualizer - 启动脚本
启动后端API服务和前端静态文件服务
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           Stage4 数据压缩可视化平台                          ║
║                                                              ║
║   基于深度学习的无损数据压缩算法可视化系统                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """检查依赖是否已安装"""
    print("[*] 检查依赖...")
    
    required_packages = ['fastapi', 'uvicorn', 'numpy', 'torch', 'pydantic']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[!] 缺少依赖: {', '.join(missing)}")
        print("[*] 正在安装依赖...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 
            str(BACKEND_DIR / 'requirements.txt')
        ])
        print("[+] 依赖安装完成")
    else:
        print("[+] 所有依赖已安装")

def start_backend():
    """启动后端API服务"""
    print("[*] 启动后端API服务...")
    
    backend_script = BASE_DIR / "backend" / "app.py"
    
    # 优先使用 Anaconda Python（具备 CUDA 支持）
    python_exe = sys.executable
    conda_candidates = [
        r"C:\ProgramData\anaconda3\python.exe",
        r"C:\Users\32599\anaconda3\python.exe",
        r"C:\Users\32599\miniconda3\python.exe",
    ]
    for cand in conda_candidates:
        if os.path.exists(cand):
            import subprocess as _sp
            try:
                result = _sp.run([cand, "-c", "import torch; print(torch.cuda.is_available())"],
                                 capture_output=True, text=True, timeout=10)
                if "True" in result.stdout:
                    python_exe = cand
                    print(f"[+] 使用 Anaconda Python (CUDA): {cand}")
                    break
            except Exception:
                pass
    
    startup_info = None
    if os.name == 'nt':
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
    process = subprocess.Popen(
        [python_exe, str(backend_script)],
        cwd=str(BASE_DIR / "backend"),
        startupinfo=startup_info,
    )
    
    print(f"[+] 后端服务已启动 (PID: {process.pid})")
    print("[+] API地址: http://localhost:8000")
    
    return process

def start_frontend():
    """启动前端静态文件服务"""
    print("[*] 启动前端服务...")
    
    # 使用Python的http.server提供静态文件
    process = subprocess.Popen(
        [sys.executable, '-m', 'http.server', '8080'],
        cwd=str(FRONTEND_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print(f"[+] 前端服务已启动 (PID: {process.pid})")
    print("[+] 前端地址: http://localhost:8080")
    
    return process

def open_browser():
    """打开浏览器"""
    time.sleep(2)  # 等待服务启动
    print("[*] 正在打开浏览器...")
    webbrowser.open('http://localhost:8080')

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    check_dependencies()
    
    # 启动服务
    backend_process = None
    frontend_process = None
    
    try:
        backend_process = start_backend()
        frontend_process = start_frontend()
        
        # 打开浏览器
        open_browser()
        
        print("\n" + "="*60)
        print("服务已启动！")
        print("-"*60)
        print("前端界面: http://localhost:8080")
        print("后端API:  http://localhost:8000")
        print("API文档:  http://localhost:8000/docs")
        print("-"*60)
        print("按 Ctrl+C 停止服务")
        print("="*60 + "\n")
        
        # 保持运行
        while True:
            time.sleep(1)
            
            # 检查进程是否还在运行
            if backend_process.poll() is not None:
                print("[!] 后端服务已停止")
                break
            if frontend_process.poll() is not None:
                print("[!] 前端服务已停止")
                break
                
    except KeyboardInterrupt:
        print("\n[*] 正在停止服务...")
    finally:
        # 清理进程
        if backend_process:
            backend_process.terminate()
            print("[+] 后端服务已停止")
        
        if frontend_process:
            frontend_process.terminate()
            print("[+] 前端服务已停止")
        
        print("[+] 系统已关闭")

if __name__ == '__main__':
    main()
