#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 Bridge Module - 连接现有算法与API
"""

import os
import sys
import json
import time
import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ALGORITHM_DIR = PROJECT_ROOT / "algorithm"
sys.path.insert(0, str(ALGORITHM_DIR))

def _detect_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

KNOWN_SHAPES = {
    14400:  (2, 60, 120),
    40000:  (2, 100, 200),
    60000:  (2, 100, 300),
}

def _infer_shape(file_path):
    num_floats = os.path.getsize(file_path) // 4
    # 1) Check .shape sidecar (saved by demo pipeline)
    sp = file_path + '.shape'
    if os.path.exists(sp):
        import json as _j
        try:
            with open(sp) as f: shape = tuple(_j.load(f))
            if shape and len(shape) == 3 and shape[0]*shape[1]*shape[2] == num_floats:
                return shape
        except: pass
    # 2) Known small shapes
    if num_floats in KNOWN_SHAPES:
        return KNOWN_SHAPES[num_floats]
    # 3) Try common module inference
    try:
        from common import VolumeShape, infer_shape_from_file
        shape = infer_shape_from_file(file_path, VolumeShape(10, 600, 2001))
        if shape.as_tuple[0]*shape.as_tuple[1]*shape.as_tuple[2] == num_floats:
            return shape.as_tuple
    except:
        pass
    # 4) Fallback: assume 1 profile, infer traces from sample count
    p = 1
    for s in [2001, 1500, 1000, 500, 250]:
        if num_floats % s == 0:
            t = num_floats // s
            return (p, t, s)
    t = max(1, int(round(num_floats / (p * 200))))
    s = num_floats // (p * t)
    return (p, t, s)

try:
    from common import (
        ExperimentConfig, VolumeData, VolumeShape, SplitConfig,
        extract_float_exponents, infer_shape_from_file
    )
    from stage4 import (
        build_single_stage4_feature_causal_edge,
        build_stage4_features_causal_edge,
        Small2DCNN,
        load_stage4_model,
        resolve_feature_mode,
        resolve_target_mode,
        feature_mode_to_in_channels,
        predictor_for_coord,
        target_symbol_for_coord
    )
    from codec import Stage4GlobalDiagonalRangeCodec
except ImportError as e:
    print(f"Warning: Could not import stage4 modules: {e}")

def extract_feature_data(
    file_path: str,
    coord: Tuple[int, int, int],
    patch_shape: Tuple[int, int] = (9, 17),
    feature_mode: str = "diagonal_causal_edge",
    target_mode: str = "raw"
) -> Dict[str, Any]:
    """
    提取指定位置的特征数据，用于前端可视化
    
    Returns:
        {
            "coord": [p, t, s],
            "patch_shape": [h, w],
            "channels": [...],
            "predicted_value": int,
            "actual_value": int,
            "context_pixels": [...]
        }
    """
    try:
        file_size = os.path.getsize(file_path)
        num_floats = file_size // 4
        
        shape_path = file_path + '.shape'
        if os.path.exists(shape_path):
            import json as _json
            with open(shape_path, 'r') as f:
                shape_tuple = tuple(_json.load(f))
        else:
            shape_tuple = _infer_shape(file_path)
        
        print(f"[INFO] Loading data: {file_path}, shape={shape_tuple}, elements={num_floats}")
        
        # 读取数据
        volume = np.memmap(file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
        exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8).reshape(shape_tuple)
        
        p, t, s = coord
        
        # 检查坐标边界
        if not (0 <= p < exps.shape[0] and 0 <= t < exps.shape[1] and 0 <= s < exps.shape[2]):
            return {
                "coord": list(coord),
                "error": f"坐标超出范围，数据形状: {exps.shape}"
            }
        
        # 查找模型检查点获取正确的 target_mode
        checkpoint_paths = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        ckpt_target_mode = target_mode
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                try:
                    import torch
                    checkpoint = torch.load(cp, map_location='cpu')
                    ckpt_target_mode = checkpoint.get('target_mode', target_mode)
                    print(f"[INFO] Using checkpoint target_mode: {ckpt_target_mode}")
                    break
                except:
                    pass
        
        # 使用 Stage4 函数提取特征 - 使用检查点的 target_mode
        feature_tensor = build_single_stage4_feature_causal_edge(
            volume=exps,
            coord=coord,
            patch_shape=patch_shape,
            target_mode=ckpt_target_mode,  # 必须与检查点匹配
            feature_mode=feature_mode,
            bos_value=0
        )
        
        # 转换为 numpy
        feature_np = feature_tensor.squeeze().numpy()
        
        in_channels = feature_mode_to_in_channels(feature_mode, target_mode)
        
        channels_data = []
        channel_names = [
            "像素值 (Values)",
            "可用掩码 (Valid Mask)",
            "因果掩码 (Causal Mask)",
            "映射掩码 (Mapped Mask)",
            "预测值 (Predicted)",
            "残差值 (Residual)"
        ]
        channel_colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c", "#1abc9c"]
        
        for i in range(min(in_channels, len(channel_names))):
            ch_data = feature_np[i]
            # 归一化到 0-1 用于可视化
            if ch_data.max() > 1.0:
                ch_data_display = (ch_data / 255.0).tolist()
            else:
                ch_data_display = ch_data.tolist()
            
            channels_data.append({
                "name": channel_names[i],
                "data": ch_data_display,
                "color": channel_colors[i],
                "index": i,
                "min": float(ch_data.min()),
                "max": float(ch_data.max()),
                "mean": float(ch_data.mean())
            })
        
        # 获取预测值
        pred_value = predictor_for_coord(exps, coord)
        actual_value = int(exps[p, t, s])
        
        # 提取上下文像素
        context_size = 5
        plane = exps[p]
        t_start = max(0, t - context_size)
        t_end = min(plane.shape[0], t + context_size + 1)
        s_start = max(0, s - context_size)
        s_end = min(plane.shape[1], s + context_size + 1)
        context = plane[t_start:t_end, s_start:s_end].tolist()
        
        return {
            "coord": list(coord),
            "patch_shape": list(patch_shape),
            "channels": channels_data,
            "predicted_value": int(pred_value),
            "actual_value": actual_value,
            "context_pixels": context,
            "target_symbol": int(target_symbol_for_coord(exps, coord, target_mode)) if target_mode == "residual" else actual_value,
            "data_shape": list(exps.shape)
        }
        
    except Exception as e:
        print(f"[ERROR] 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "coord": list(coord),
            "patch_shape": list(patch_shape),
            "channels": [],
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def predict_probabilities(
    file_path: str,
    coord: Tuple[int, int, int],
    checkpoint_path: str,
    patch_shape: Tuple[int, int] = (9, 17),
    feature_mode: str = "diagonal_causal_edge",
    target_mode: str = "raw",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    预测指定位置的概率分布
    
    Returns:
        {
            "coord": [p, t, s],
            "probabilities": [256个概率值],
            "predicted_symbol": int,
            "actual_symbol": int,
            "entropy": float,
            "top5": [{"symbol": int, "prob": float}, ...]
        }
    """
    try:
        # Load model
        model, checkpoint = load_stage4_model(checkpoint_path, device=device)
        
        # Load volume
        shape_tuple = _infer_shape(file_path)

        volume = np.memmap(file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
        exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8).reshape(shape_tuple)
        
        # Build feature
        feature = build_single_stage4_feature_causal_edge(
            volume=exps,
            coord=coord,
            patch_shape=patch_shape,
            target_mode=target_mode,
            feature_mode=feature_mode
        )
        
        # Predict
        with torch.inference_mode():
            logits = model(feature.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        actual_symbol = target_symbol_for_coord(exps, coord, target_mode)
        predicted_symbol = int(np.argmax(probs))
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Get top5
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5 = [{"symbol": int(i), "prob": float(probs[i])} for i in top5_indices]
        
        return {
            "coord": list(coord),
            "probabilities": probs.tolist(),
            "predicted_symbol": predicted_symbol,
            "actual_symbol": actual_symbol,
            "entropy": float(entropy),
            "top5": top5
        }
        
    except Exception as e:
        print(f"Error predicting probabilities: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def compress_data(
    file_path: str,
    config: Dict[str, Any],
    output_dir: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """
    使用真正的 Stage4 算法压缩数据
    
    Returns:
        {
            "original_size": int,
            "compressed_size": int,
            "compression_ratio": float,
            "bits_per_voxel": float,
            "bitstream_path": str,
            "timing": {...},
            "method": str
        }
    """
    import time
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file size
    original_size = os.path.getsize(file_path)
    
    # 输出文件路径
    bitstream_path = os.path.join(output_dir, "compressed.s4rc")
    
    try:
        # 尝试使用真正的 Stage4 编解码器
        from codec import Stage4GlobalDiagonalRangeCodec
        from common import ExperimentConfig, VolumeShape, infer_shape_from_file
        
        # 自动推断数据形状
        if shape is not None:
            shape_tuple = tuple(shape)
        else:
            shape_tuple = _infer_shape(file_path)
        
        print(f"[INFO] Compressing: {file_path}, shape={shape_tuple}")
        
        # 加载数据并提取指数部分
        volume = np.memmap(file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
        exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
        
        # 寻找模型检查点
        checkpoint_paths = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
            str(ALGORITHM_DIR / "outputs_tui_heldout_materialized" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("未找到模型检查点文件")
        
        print(f"[INFO] 使用模型检查点: {checkpoint_path}")
        
        # 从检查点获取配置
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ckpt_feature_mode = checkpoint.get('feature_mode', 'diagonal_causal_edge')
        ckpt_target_mode = checkpoint.get('target_mode', 'residual')  # 检查点使用residual
        
        print(f"[INFO] 检查点配置: feature_mode={ckpt_feature_mode}, target_mode={ckpt_target_mode}")
        
        # 创建配置 - 必须与检查点匹配
        exp_config = ExperimentConfig(
            patch_shape=tuple(config.get("patch_shape", [9, 17])),
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            range_total=1 << 15,  # 32768
            codec_device=config.get("device", "cpu")
        )
        
        # 创建编解码器 - 使用检查点的配置
        codec = Stage4GlobalDiagonalRangeCodec(
            checkpoint_path=checkpoint_path,
            config=exp_config,
            device=config.get("device", "cpu"),
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            profile_timing=True,
            inference_batch=config.get("inference_batch", 1),
            progress=True,
            progress_label="Compress"
        )
        
        # 执行压缩
        print(f"[INFO] 开始压缩，数据形状: {exps.shape}")
        encode_metrics = codec.encode_exponents(exps, bitstream_path)
        
        compressed_size = encode_metrics.get("total_bytes", 0)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        elapsed = time.time() - start_time
        
        print(f"[INFO] 压缩完成!")
        print(f"       原始大小: {original_size/1024/1024:.2f} MB")
        print(f"       压缩后: {compressed_size/1024/1024:.2f} MB")
        print(f"       压缩比: {compression_ratio:.2f}x")
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "bits_per_voxel": encode_metrics.get("bits_per_voxel", 0),
            "bits_per_modeled_voxel": encode_metrics.get("bits_per_modeled_voxel", 0),
            "bitstream_path": bitstream_path,
            "method": "stage4_real",
            "timing": {
                "total_seconds": elapsed,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
                "codec_timing": encode_metrics.get("timing", {})
            },
            "encode_metrics": encode_metrics
        }
        
    except Exception as e:
        print(f"[WARNING] 真实压缩失败，使用备用压缩: {e}")
        import traceback
        traceback.print_exc()
        
        # 备用：使用简单的无损压缩（zlib）作为对比
        import zlib
        
        with open(file_path, 'rb') as f_in:
            data = f_in.read()
            compressed_data = zlib.compress(data, level=9)
        
        with open(bitstream_path, 'wb') as f_out:
            f_out.write(compressed_data)
        
        compressed_size = os.path.getsize(bitstream_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # 计算 bits per voxel
        file_size = os.path.getsize(file_path)
        num_floats = file_size // 4
        bits_per_voxel = (compressed_size * 8) / num_floats
        
        elapsed = time.time() - start_time
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "bits_per_voxel": bits_per_voxel,
            "bitstream_path": bitstream_path,
            "method": "zlib_fallback",
            "error": str(e),
            "timing": {
                "total_seconds": elapsed,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024)
            }
        }

def get_visualization_data(task: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """获取不同类型的可视化数据"""
    
    if data_type == "input_slice":
        # Return a slice of input data
        file_path = task["file_path"]
        try:
            shape_tuple = _infer_shape(file_path)
            volume = np.memmap(file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
            exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8).reshape(shape_tuple)
            
            # Return middle slice
            p = shape_tuple[0] // 2
            slice_data = exps[p].tolist()
            
            return {
                "type": "input_slice",
                "profile_index": p,
                "shape": [shape_tuple[1], shape_tuple[2]],
                "data": slice_data[:100]  # Limit for JSON
            }
        except Exception as e:
            return {"error": str(e)}
    
    elif data_type == "probabilities":
        # Return probability distribution example
        # In real implementation, compute from model
        probs = np.random.dirichlet(np.ones(256)) * 0.5
        probs[100] = 0.3  # Peak at one symbol
        probs = probs / probs.sum()
        
        return {
            "type": "probabilities",
            "probabilities": probs.tolist(),
            "predicted_symbol": int(np.argmax(probs)),
            "entropy": float(-np.sum(probs * np.log2(probs + 1e-10)))
        }
    
    elif data_type == "cdf":
        # Return CDF example
        probs = np.random.dirichlet(np.ones(256))
        cdf = np.cumsum(probs)
        
        return {
            "type": "cdf",
            "cdf": cdf.tolist(),
            "frequencies": (probs * 32768).astype(int).tolist()
        }
    
    elif data_type == "compression_stats":
        output = task.get("output", {})
        return {
            "type": "compression_stats",
            "original_size_mb": output.get("timing", {}).get("original_size_mb", 0),
            "compressed_size_mb": output.get("timing", {}).get("compressed_size_mb", 0),
            "compression_ratio": output.get("compression_ratio", 0),
            "bits_per_voxel": output.get("bits_per_voxel", 0)
        }
    
    return {"error": "Unknown data type"}

def get_sample_data_for_visualization(file_path: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    """获取多个样本点用于可视化展示"""
    samples = []
    
    try:
        shape_tuple = _infer_shape(file_path)
        
        # Generate random sample coordinates
        np.random.seed(42)
        for _ in range(num_samples):
            p = np.random.randint(0, shape_tuple[0])
            t = np.random.randint(10, max(11, shape_tuple[1] - 10))
            s = np.random.randint(10, max(11, shape_tuple[2] - 10))
            
            samples.append({
                "coord": [int(p), int(t), int(s)],
                "description": f"Profile {p}, Trace {t}, Sample {s}"
            })
    except Exception as e:
        print(f"Error getting samples: {e}")
    
    return samples


def compress_data_with_progress(
    file_path: str,
    config: Dict[str, Any],
    output_dir: str,
    progress_callback=None,
    shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """
    带进度回调的压缩数据
    
    Args:
        file_path: 输入文件路径
        config: 配置字典
        output_dir: 输出目录
        progress_callback: 进度回调函数(percent, message)
        shape: 数据三维形状，如果为 None 则自动推断
    
    Returns:
        压缩结果字典
    """
    import time
    
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    original_size = os.path.getsize(file_path)
    bitstream_path = os.path.join(output_dir, "compressed.s4rc")
    
    def update_progress(percent, message=""):
        if progress_callback:
            progress_callback(percent, message)
    
    try:
        from codec import Stage4GlobalDiagonalRangeCodec
        from common import ExperimentConfig
        
        if shape is not None:
            shape_tuple = tuple(shape)
        else:
            shape_tuple = _infer_shape(file_path)
        
        update_progress(5, "Loading data...")
        
        volume = np.memmap(file_path, dtype=np.float32, mode='r').reshape(shape_tuple)
        exps = ((volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
        
        checkpoint_paths = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint not found")
        
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ckpt_feature_mode = checkpoint.get('feature_mode', 'diagonal_causal_edge')
        ckpt_target_mode = checkpoint.get('target_mode', 'residual')
        device = config.get("device", "cpu")
        
        update_progress(15, f"Initializing codec (device={device})...")
        
        exp_config = ExperimentConfig(
            patch_shape=tuple(config.get("patch_shape", [9, 17])),
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            range_total=1 << 15,
            codec_device=device
        )
        
        codec = Stage4GlobalDiagonalRangeCodec(
            checkpoint_path=checkpoint_path,
            config=exp_config,
            device=device,
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            profile_timing=True,
            inference_batch=config.get("inference_batch", 8192),
            progress=True,
            progress_label="Compress"
        )
        
        total_voxels = int(np.prod(exps.shape))
        update_progress(20, f"Starting CNN compression ({total_voxels:,} voxels, {exps.shape[1] + exps.shape[2] - 1} diagonals)...")
        
        encode_metrics = codec.encode_exponents(exps, bitstream_path)
        
        update_progress(98, "Finalizing bitstream...")
        
        wall_time = time.time() - start_time
        codec_timing = encode_metrics.get("timing", {})
        
        print(f"[TIMING] Total wall: {wall_time:.1f}s")
        print(f"[TIMING] Patch build: {codec_timing.get('patch_build_seconds', 0):.1f}s ({codec_timing.get('patch_build_us_per_voxel', 0):.1f}us/vox)")
        print(f"[TIMING] Model inference: {codec_timing.get('model_inference_seconds', 0):.1f}s ({codec_timing.get('model_inference_us_per_voxel', 0):.1f}us/vox)")
        print(f"[TIMING] CDF quantize: {codec_timing.get('cdf_quantization_seconds', 0):.1f}s")
        print(f"[TIMING] Range coder: {codec_timing.get('range_coder_seconds', 0):.1f}s ({codec_timing.get('range_coder_us_per_voxel', 0):.1f}us/vox)")
        print(f"[TIMING] Other overhead: {codec_timing.get('other_overhead_seconds', 0):.1f}s")
        
        compressed_size = encode_metrics.get("total_bytes", 0)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        bits_per_voxel = encode_metrics.get("bits_per_voxel", 0)
        
        update_progress(100, "Compression complete")
        
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "bits_per_voxel": bits_per_voxel,
            "bitstream_path": bitstream_path,
            "method": "stage4_real",
            "timing": {
                "total_seconds": wall_time,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
                "patch_build_s": codec_timing.get("patch_build_seconds", 0),
                "model_inference_s": codec_timing.get("model_inference_seconds", 0),
                "cdf_quantize_s": codec_timing.get("cdf_quantization_seconds", 0),
                "range_coder_s": codec_timing.get("range_coder_seconds", 0),
                "other_overhead_s": codec_timing.get("other_overhead_seconds", 0),
            }
        }
        
    except Exception as e:
        update_progress(0, f"Error: {str(e)}")
        raise


def decompress_data_with_progress(
    bitstream_path: str,
    output_path: str,
    progress_callback=None
) -> Dict[str, Any]:
    """
    带进度回调的解压数据
    
    Args:
        bitstream_path: 压缩文件路径
        output_path: 输出文件路径
        progress_callback: 进度回调函数(percent, message)
    
    Returns:
        解压结果字典
    """
    import time
    
    start_time = time.time()
    
    def update_progress(percent, message=""):
        if progress_callback:
            progress_callback(percent, message)
    
    try:
        from codec import Stage4GlobalDiagonalRangeCodec, read_bitstream
        from common import ExperimentConfig
        import torch
        
        update_progress(5, "Reading bitstream header...")
        
        header, _payload = read_bitstream(bitstream_path)
        shape = tuple(int(v) for v in header["shape"])
        
        update_progress(10, "Loading model...")
        
        checkpoint_paths = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        
        checkpoint_path = None
        for cp in checkpoint_paths:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint not found")
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_feature_mode = ckpt.get('feature_mode', 'diagonal_causal_edge')
        ckpt_target_mode = ckpt.get('target_mode', 'residual')
        
        exp_config = ExperimentConfig(
            patch_shape=tuple(header.get("patch_shape", [9, 17])),
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            range_total=int(header.get("total_freq", 1 << 15)),
            codec_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        update_progress(15, "Initializing decoder...")
        
        codec = Stage4GlobalDiagonalRangeCodec(
            checkpoint_path=checkpoint_path,
            config=exp_config,
            device=exp_config.codec_device,
            feature_mode=ckpt_feature_mode,
            target_mode=ckpt_target_mode,
            profile_timing=False,
            inference_batch=8192 if torch.cuda.is_available() else 1
        )
        
        total_voxels = int(np.prod(shape))
        update_progress(20, f"Starting CNN decompression ({total_voxels:,} voxels)...")
        
        decoded, out_header = codec.decode_exponents(bitstream_path)
        
        update_progress(95, "Saving decompressed data...")
        decoded.astype(np.uint8).tofile(output_path)
        
        update_progress(100, "Decompression complete")
        
        decompressed_size = os.path.getsize(output_path)
        
        return {
            "bitstream_path": bitstream_path,
            "output_path": output_path,
            "decompressed_size": decompressed_size,
            "shape": shape,
            "method": "stage4_real",
            "timing": {
                "total_seconds": time.time() - start_time
            }
        }
        
    except Exception as e:
        update_progress(0, f"Error: {str(e)}")
        raise
