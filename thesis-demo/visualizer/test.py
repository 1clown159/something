#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 Visualizer - 快速测试脚本
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "visualizer" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "algorithm"))

def test_imports():
    """测试导入"""
    print("[*] 测试模块导入...")
    
    errors = []
    
    # 测试后端模块
    try:
        from core.stage4_bridge import extract_feature_data
        print("  [✓] stage4_bridge 导入成功")
    except Exception as e:
        errors.append(f"stage4_bridge: {e}")
        print(f"  [✗] stage4_bridge 导入失败: {e}")
    
    # 测试stage4模块
    try:
        from stage4 import build_single_stage4_feature_causal_edge
        print("  [✓] stage4 模块导入成功")
    except Exception as e:
        errors.append(f"stage4: {e}")
        print(f"  [✗] stage4 模块导入失败: {e}")
    
    # 测试common模块
    try:
        from common import VolumeShape
        print("  [✓] common 模块导入成功")
    except Exception as e:
        errors.append(f"common: {e}")
        print(f"  [✗] common 模块导入失败: {e}")
    
    return len(errors) == 0

def test_feature_extraction():
    """测试特征提取"""
    print("\n[*] 测试特征提取功能...")
    
    # 创建测试数据
    test_file = str(Path(__file__).resolve().parent / "uploads" / "test_data.bin")
    
    try:
        import numpy as np
        
        # 创建测试数据
        if not os.path.exists(test_file):
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            data = np.random.randn(10, 600, 2001).astype(np.float32)
            data.tofile(test_file)
            print(f"  [✓] 创建测试数据: {test_file}")
        
        # 测试特征提取
        from core.stage4_bridge import extract_feature_data
        
        result = extract_feature_data(
            file_path=test_file,
            coord=(0, 100, 100),
            patch_shape=(9, 17),
            feature_mode="diagonal_causal_edge",
            target_mode="raw"
        )
        
        if "channels" in result:
            print(f"  [✓] 特征提取成功，获取 {len(result['channels'])} 个通道")
            for ch in result['channels']:
                print(f"      - {ch['name']}: shape {len(ch['data'])}x{len(ch['data'][0]) if ch['data'] else 0}")
            return True
        else:
            print(f"  [✗] 特征提取返回数据异常")
            return False
            
    except Exception as e:
        print(f"  [✗] 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_structure():
    """测试API结构"""
    print("\n[*] 测试API结构...")
    
    try:
        from backend.app import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # 测试根路径
        response = client.get("/")
        if response.status_code == 200:
            print("  [✓] API根路径响应正常")
        else:
            print(f"  [✗] API根路径响应异常: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  [✗] API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Stage4 Visualizer - 快速测试")
    print("="*60)
    
    results = []
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试特征提取
    results.append(("特征提取", test_feature_extraction()))
    
    # 测试API
    results.append(("API结构", test_api_structure()))
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("-"*60)
    
    for name, result in results:
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"  [{symbol}] {name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("-"*60)
    print(f"总计: {passed}/{total} 项测试通过")
    print("="*60)
    
    if passed == total:
        print("\n[✓] 所有测试通过！系统可以正常运行。")
        print("\n启动命令:")
        print("  python serve.py")
        return 0
    else:
        print("\n[✗] 部分测试失败，请检查依赖和配置。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
