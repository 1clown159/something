#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实 Stage4 压缩功能
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "algorithm"))
sys.path.insert(0, str(PROJECT_ROOT / "visualizer" / "backend"))

import numpy as np

def test_real_compression():
    """测试真实压缩"""
    print("="*70)
    print("Stage4 真实压缩测试")
    print("="*70)
    
    # 选择测试文件
    test_files = [
        ("constant_data.bin", "常量数据"),
        ("gradient_data.bin", "渐变数据"),
        ("seismic_like_data.bin", "地震数据"),
        ("random_data.bin", "随机数据"),
    ]
    
    VISUALIZER_DIR = Path(__file__).resolve().parent
    
    test_dir = VISUALIZER_DIR / "uploads" / "test_samples"
    output_dir = VISUALIZER_DIR / "outputs" / "test_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, desc in test_files:
        filepath = test_dir / filename
        if not filepath.exists():
            print(f"\n[!] 文件不存在: {filename}")
            continue
        
        print(f"\n{'-'*70}")
        print(f"测试: {filename} ({desc})")
        print(f"{'-'*70}")
        
        try:
            from core.stage4_bridge import compress_data
            
            config = {
                "feature_mode": "diagonal_causal_edge",
                "target_mode": "raw",
                "patch_shape": [9, 17],
                "inference_batch": 1,
                "codec_layout": "global_diag",
                "device": "cpu"
            }
            
            result = compress_data(
                file_path=str(filepath),
                config=config,
                output_dir=str(output_dir / filename.replace('.bin', ''))
            )
            
            print(f"[结果]")
            print(f"  方法: {result.get('method', 'unknown')}")
            print(f"  原始大小: {result['original_size'] / 1024:.2f} KB")
            print(f"  压缩后: {result['compressed_size'] / 1024:.2f} KB")
            print(f"  压缩比: {result['compression_ratio']:.2f}x")
            print(f"  码率: {result.get('bits_per_voxel', 0):.4f} bits/voxel")
            
            if result.get('method') == 'stage4_real':
                print(f"  [OK] 使用了真实的 Stage4 压缩!")
            elif result.get('method') == 'zlib_fallback':
                print(f"  [WARNING] 使用了备用压缩 (Stage4 失败)")
                if 'error' in result:
                    print(f"  错误: {result['error'][:100]}...")
            
        except Exception as e:
            print(f"[ERROR] 压缩失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)

def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "="*70)
    print("特征提取测试")
    print("="*70)
    
    VISUALIZER_DIR = Path(__file__).resolve().parent
    test_file = VISUALIZER_DIR / "uploads" / "test_samples" / "gradient_data.bin"
    
    if not test_file.exists():
        print(f"[!] 测试文件不存在")
        return
    
    try:
        from core.stage4_bridge import extract_feature_data
        
        result = extract_feature_data(
            file_path=str(test_file),
            coord=(0, 50, 50),
            patch_shape=(9, 17),
            feature_mode="diagonal_causal_edge",
            target_mode="raw"
        )
        
        if 'error' in result:
            print(f"[ERROR] 特征提取失败: {result['error']}")
        else:
            print(f"[OK] 特征提取成功!")
            print(f"  坐标: {result['coord']}")
            print(f"  数据形状: {result.get('data_shape', 'unknown')}")
            print(f"  通道数: {len(result.get('channels', []))}")
            print(f"  预测值: {result.get('predicted_value')}")
            print(f"  实际值: {result.get('actual_value')}")
            
            for ch in result.get('channels', []):
                print(f"  通道 {ch['index']} ({ch['name']}): min={ch['min']:.4f}, max={ch['max']:.4f}, mean={ch['mean']:.4f}")
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()

def check_model_checkpoint():
    """检查模型检查点"""
    print("\n" + "="*70)
    print("检查模型检查点")
    print("="*70)
    
    ALGORITHM_DIR = PROJECT_ROOT / "algorithm"
    checkpoint_paths = [
        str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
        str(ALGORITHM_DIR / "outputs_tui_heldout_materialized" / "stage4" / "causal" / "checkpoint.pt"),
    ]
    
    found = False
    for cp in checkpoint_paths:
        if os.path.exists(cp):
            size_mb = os.path.getsize(cp) / 1024 / 1024
            print(f"[OK] 找到模型检查点: {cp}")
            print(f"     大小: {size_mb:.2f} MB")
            found = True
            
            # 尝试加载检查点
            try:
                import torch
                checkpoint = torch.load(cp, map_location='cpu')
                print(f"     检查点信息:")
                print(f"       - Stage: {checkpoint.get('stage', 'unknown')}")
                print(f"       - Mode: {checkpoint.get('mode', 'unknown')}")
                print(f"       - Feature Mode: {checkpoint.get('feature_mode', 'unknown')}")
                print(f"       - Target Mode: {checkpoint.get('target_mode', 'unknown')}")
            except Exception as e:
                print(f"     [WARNING] 无法读取检查点: {e}")
            
            break
    
    if not found:
        print("[!] 未找到模型检查点!")
        print("    请确保以下路径之一存在:")
        for cp in checkpoint_paths:
            print(f"      - {cp}")
    
    return found

if __name__ == '__main__':
    # 检查模型检查点
    has_model = check_model_checkpoint()
    
    # 测试特征提取
    test_feature_extraction()
    
    # 测试压缩
    test_real_compression()
