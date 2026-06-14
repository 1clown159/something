#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmallVolumeProcessor - Demo Pipeline for 2×2×N small volume
分步处理，预计算概率，支持逐点查询
"""

import os
import sys
import json
import math
import zlib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ALGORITHM_DIR = PROJECT_ROOT / "algorithm"
sys.path.insert(0, str(ALGORITHM_DIR))

from core.sgy_extractor import extract_sgy_headers, extract_sgy_float32

try:
    from common import (
        ExperimentConfig, VolumeShape, infer_shape_from_file,
        extract_float_exponents
    )
    from stage4 import (
        build_single_stage4_feature_causal_edge,
        Small2DCNN,
        load_stage4_model,
        feature_mode_to_in_channels,
        predictor_for_coord,
        target_symbol_for_coord
    )
    from range_coder import RangeEncoder
except ImportError as e:
    print(f"[WARN] Could not import stage4 modules: {e}")
    raise


class SmallVolumeProcessor:
    """
    处理 2×2×N 小体积数据，支持 demo 分步查询
    """

    def __init__(self, sgy_path: str, checkpoint_path: Optional[str] = None,
                 n_samples: int = 100, device: str = "cpu"):
        self.sgy_path = sgy_path
        self.n_samples = n_samples
        self.device = device
        self.shape = (2, 2, n_samples)  # 固定 2×2×N
        self.total_voxels = 2 * 2 * n_samples

        # 1) 提取 small_volume
        self._extract_small_volume()

        # 2) 加载模型
        self._load_model(checkpoint_path)

        # 3) 预计算所有坐标点的概率分布
        self._precompute_all_probs()

        print(f"[SmallVolumeProcessor] Ready: shape={self.shape}, "
              f"total={self.total_voxels}, device={device}")

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _extract_small_volume(self):
        """从 SGY 中提取 2×2×N 小体积数据"""
        try:
            sgy_headers = extract_sgy_headers(self.sgy_path)
            sgy_meta = sgy_headers["meta"]

            float32_2d = extract_sgy_float32(self.sgy_path, meta=sgy_meta, dtype=np.float32)

            profile_count = sgy_meta.get("profile_count") or 1
            traces_per_profile = sgy_meta.get("traces_per_profile") or sgy_meta["trace_count"]

            if profile_count * traces_per_profile == sgy_meta["trace_count"]:
                float32_3d = float32_2d.reshape(profile_count, traces_per_profile, sgy_meta["sample_count"])
            else:
                float32_3d = float32_2d.reshape(1, sgy_meta["trace_count"], sgy_meta["sample_count"])

            # 全域分层采样找能量最高的 profile 和 trace
            p_count = float32_3d.shape[0]
            t_count = float32_3d.shape[1]
            s_max = min(self.n_samples, float32_3d.shape[2])

            best_p, best_t, best_energy = 0, 0, 0.0
            # 在全域范围内间隔采样，确保覆盖所有区域
            p_step = max(1, p_count // 30) if p_count > 30 else 1
            t_step = max(1, t_count // 80) if t_count > 80 else 1
            for p in range(0, p_count, p_step):
                for t in range(0, t_count, t_step):
                    row = float32_3d[p, t, :min(s_max, 200)]
                    nz = np.count_nonzero(np.abs(row) > 0.001)
                    energy = float(np.mean(np.abs(row)))
                    if nz > 2 and energy > best_energy:
                        best_energy = energy
                        best_p, best_t = p, t

            print(f"[SmallVolumeProcessor] Best pos: p={best_p} t={best_t} energy={best_energy:.4f} "
                  f"(scanned p_step={p_step} t_step={t_step})")

            # 从最佳位置截取 2 profile × 2 trace × N samples
            p_start = best_p
            t_start = best_t
            p_end = min(p_start + 2, p_count)
            t_end = min(t_start + 2, t_count)
            s_end = min(s_max, float32_3d.shape[2])

            small = float32_3d[p_start:p_end, t_start:t_end, :s_end].copy()

            # 如果实际尺寸不足 2×2×N，用零填充
            if small.shape != self.shape:
                padded = np.zeros(self.shape, dtype=np.float32)
                padded[:small.shape[0], :small.shape[1], :small.shape[2]] = small
                small = padded

            self.volume = small
            self.exps = ((self.volume.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)

            print(f"[SmallVolumeProcessor] Extracted shape={small.shape}, "
                  f"value_range=[{small.min():.3f}, {small.max():.3f}]")

        except Exception as e:
            print(f"[ERROR] Failed to extract small volume: {e}")
            raise

    def _load_model(self, checkpoint_path: Optional[str] = None):
        """加载 Stage4 模型"""
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        print(f"[SmallVolumeProcessor] Loading model from {checkpoint_path}")

        self.model, self.checkpoint = load_stage4_model(checkpoint_path, device=self.device)
        self.model.eval()

        # 从 checkpoint 读取配置
        self.feature_mode = self.checkpoint.get('feature_mode', 'diagonal_causal_edge')
        self.target_mode = self.checkpoint.get('target_mode', 'residual')
        self.patch_shape = tuple(self.checkpoint.get('patch_shape', [9, 17]))

        print(f"[SmallVolumeProcessor] Model config: feature_mode={self.feature_mode}, "
              f"target_mode={self.target_mode}, patch_shape={self.patch_shape}")

    def _find_checkpoint(self) -> Optional[str]:
        """查找可用的 checkpoint"""
        candidates = [
            str(ALGORITHM_DIR / "outputs_tui_smoke" / "stage4" / "causal" / "checkpoint.pt"),
            str(ALGORITHM_DIR / "outputs_tui_heldout_materialized" / "stage4" / "causal" / "checkpoint.pt"),
        ]
        for cp in candidates:
            if os.path.exists(cp):
                return cp
        # 全局搜索
        found = list(ALGORITHM_DIR.rglob("*/checkpoint.pt"))
        if found:
            return str(found[0])
        return None

    def _precompute_all_probs(self):
        """预计算所有坐标点的概率分布"""
        print(f"[SmallVolumeProcessor] Pre-computing probabilities for {self.total_voxels} voxels...")

        self.all_probs = {}      # coord -> probabilities (numpy array 256)
        self.all_symbols = {}    # coord -> actual symbol
        self.all_entropy = {}    # coord -> entropy value
        self.all_top5 = {}       # coord -> top5 list

        count = 0
        with torch.inference_mode():
            for p in range(self.shape[0]):
                for t in range(self.shape[1]):
                    for s in range(self.shape[2]):
                        coord = (p, t, s)
                        try:
                            feature = build_single_stage4_feature_causal_edge(
                                volume=self.exps,
                                coord=coord,
                                patch_shape=self.patch_shape,
                                target_mode=self.target_mode,
                                feature_mode=self.feature_mode,
                                bos_value=0
                            )
                            logits = self.model(feature.to(self.device))
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                            symbol = int(target_symbol_for_coord(self.exps, coord, self.target_mode))
                            entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
                            top5_idx = np.argsort(probs)[-5:][::-1]
                            top5 = [{"symbol": int(i), "prob": float(probs[i])} for i in top5_idx]

                            self.all_probs[coord] = probs
                            self.all_symbols[coord] = symbol
                            self.all_entropy[coord] = entropy
                            self.all_top5[coord] = top5

                            count += 1
                            if count % 50 == 0:
                                print(f"[SmallVolumeProcessor] Pre-computed {count}/{self.total_voxels}")

                        except Exception as e:
                            print(f"[WARN] Failed to precompute {coord}: {e}")
                            # Fallback: uniform distribution
                            self.all_probs[coord] = np.ones(256) / 256
                            self.all_symbols[coord] = int(self.exps[p, t, s])
                            self.all_entropy[coord] = 8.0
                            self.all_top5[coord] = [{"symbol": int(self.exps[p, t, s]), "prob": 1.0}]

        print(f"[SmallVolumeProcessor] Pre-computation done.")

    # ------------------------------------------------------------------
    # 坐标工具
    # ------------------------------------------------------------------
    def _index_to_coord(self, sample_index: int) -> Tuple[int, int, int]:
        """1D sample_index → 3D coord (p, t, s)"""
        s = sample_index % self.shape[2]
        t = (sample_index // self.shape[2]) % self.shape[1]
        p = (sample_index // (self.shape[1] * self.shape[2])) % self.shape[0]
        return (p, t, s)

    def _coord_to_index(self, coord: Tuple[int, int, int]) -> int:
        """3D coord → 1D sample_index"""
        p, t, s = coord
        return p * self.shape[1] * self.shape[2] + t * self.shape[2] + s

    def _get_diagonal_ordered_coords(self) -> List[Tuple[int, int, int]]:
        """返回按 diagonal 顺序排列的所有坐标"""
        coords = []
        max_diag = self.shape[0] + self.shape[1] + self.shape[2] - 3
        for diag in range(max_diag + 1):
            for p in range(self.shape[0]):
                for t in range(self.shape[1]):
                    for s in range(self.shape[2]):
                        if p + t + s == diag:
                            coords.append((p, t, s))
        return coords

    def _probs_to_cdf(self, probs: np.ndarray) -> np.ndarray:
        """概率分布 → CDF (int32, sum=32768)，确保严格单调递增"""
        total_freq = 1 << 15  # 32768
        scaled = probs * total_freq
        freqs = np.floor(scaled).astype(np.int64)
        # 分配余量到最大的小数部分，保证总和 = total_freq
        remainder = int(total_freq - np.sum(freqs))
        if remainder > 0:
            fracs = scaled - freqs.astype(np.float64)
            top_idx = np.argsort(-fracs)[:remainder]
            for i in top_idx:
                freqs[i] += 1
        # 确保每个非零概率的 bin 至少有 1 个计数
        nonzero = probs > 0
        zero_mask = (freqs == 0) & nonzero
        if np.any(zero_mask):
            # 从概率最高的 bin 借 1 给零计数 bin
            for zi in np.where(zero_mask)[0]:
                donor = np.argmax(freqs)
                if freqs[donor] > 1:
                    freqs[donor] -= 1
                    freqs[zi] += 1
        cdf = np.zeros(257, dtype=np.int32)
        cdf[1:] = np.cumsum(freqs.astype(np.int32))
        cdf[-1] = total_freq
        return cdf

    # ------------------------------------------------------------------
    # API 方法
    # ------------------------------------------------------------------
    def decompose(self, sample_index: int) -> Dict[str, Any]:
        """Bit 拆解"""
        coord = self._index_to_coord(sample_index)
        p, t, s = coord
        v = float(self.volume[p, t, s])
        u32 = np.array([v], dtype=np.float32).view(np.uint32)[0]
        sign = int((u32 >> 31) & 0x1)
        exp_raw = int((u32 >> 23) & 0xFF)
        mant = int(u32 & 0x7FFFFF)

        return {
            "coord": list(coord),
            "sample_index": sample_index,
            "original": v,
            "sign": sign,
            "exp_raw": exp_raw,
            "exp_value": exp_raw - 127,
            "mant": mant,
            "binary": f"{sign}|{format(exp_raw, '08b')}|{format(mant, '023b')}"
        }

    def features(self, sample_index: int) -> Dict[str, Any]:
        """特征提取"""
        coord = self._index_to_coord(sample_index)

        feature_tensor = build_single_stage4_feature_causal_edge(
            volume=self.exps,
            coord=coord,
            patch_shape=self.patch_shape,
            target_mode=self.target_mode,
            feature_mode=self.feature_mode,
            bos_value=0
        )
        feature_np = feature_tensor.squeeze().numpy()
        in_channels = feature_mode_to_in_channels(self.feature_mode, self.target_mode)

        channel_names = [
            "像素值 (Values)", "可用掩码 (Valid Mask)", "因果掩码 (Causal Mask)",
            "映射掩码 (Mapped Mask)", "预测值 (Predicted)", "残差值 (Residual)"
        ]
        channel_colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c", "#1abc9c"]

        channels = []
        for i in range(min(in_channels, len(channel_names))):
            ch_data = feature_np[i]
            ch_display = (ch_data / 255.0 if ch_data.max() > 1.0 else ch_data).tolist()
            channels.append({
                "name": channel_names[i],
                "data": ch_display,
                "color": channel_colors[i],
                "index": i,
                "min": float(ch_data.min()),
                "max": float(ch_data.max()),
                "mean": float(ch_data.mean())
            })

        pred_value = predictor_for_coord(self.exps, coord)
        actual_value = int(self.exps[coord[0], coord[1], coord[2]])

        return {
            "coord": list(coord),
            "sample_index": sample_index,
            "patch_shape": list(self.patch_shape),
            "channels": channels,
            "predicted_value": int(pred_value),
            "actual_value": actual_value,
            "target_symbol": int(target_symbol_for_coord(self.exps, coord, self.target_mode)),
            "data_shape": list(self.exps.shape)
        }

    def predict(self, sample_index: int) -> Dict[str, Any]:
        """CNN 概率预测"""
        coord = self._index_to_coord(sample_index)
        probs = self.all_probs[coord]
        symbol = self.all_symbols[coord]
        entropy = self.all_entropy[coord]
        top5 = self.all_top5[coord]

        return {
            "coord": list(coord),
            "sample_index": sample_index,
            "probabilities": probs.tolist(),
            "predicted_symbol": int(np.argmax(probs)),
            "actual_symbol": symbol,
            "entropy": entropy,
            "top5": top5
        }

    def encode(self, sample_index: int) -> Dict[str, Any]:
        """Range Coding — 顺序编码到目标点，返回该点的编码信息"""
        coord = self._index_to_coord(sample_index)
        symbol = self.all_symbols[coord]
        probs = self.all_probs[coord]

        # 计算该 symbol 的概率区间 [cdf_low, cdf_high)
        cdf_low = float(np.sum(probs[:symbol]))
        cdf_high = float(np.sum(probs[:symbol + 1]))

        # 方案 A: 从开头顺序编码到该点，获取累计 bit 数
        all_coords = self._get_diagonal_ordered_coords()
        target_idx = all_coords.index(coord)

        encoder = RangeEncoder()
        for c in all_coords[:target_idx + 1]:
            sym = self.all_symbols[c]
            p = self.all_probs[c]
            cdf = self._probs_to_cdf(p)
            encoder.encode_symbol(cdf, sym)

        encoded_bytes = encoder.finish()
        total_bits = len(encoded_bytes) * 8

        return {
            "coord": list(coord),
            "sample_index": sample_index,
            "symbol": symbol,
            "prob": float(probs[symbol]),
            "range_low": cdf_low,
            "range_high": cdf_high,
            "entropy": float(-np.sum(probs * np.log2(probs + 1e-10))),
            "bits_output": total_bits,
            "encoded_count": target_idx + 1,
            "total_coords": len(all_coords),
            "bits_string": ''.join(format(b, '08b') for b in encoded_bytes[:4]) + "..." if len(encoded_bytes) > 4 else ''.join(format(b, '08b') for b in encoded_bytes)
        }

    def stats(self) -> Dict[str, Any]:
        """压缩统计 — 对全部小体积做一次完整编码"""
        all_coords = self._get_diagonal_ordered_coords()

        encoder = RangeEncoder()
        for c in all_coords:
            sym = self.all_symbols[c]
            p = self.all_probs[c]
            cdf = self._probs_to_cdf(p)
            encoder.encode_symbol(cdf, sym)

        encoded_bytes = encoder.finish()
        exp_bytes = len(encoded_bytes)

        # Sign / Mantissa 大小（简化计算）
        volume_flat = self.volume.reshape(-1)
        u32 = volume_flat.view(np.uint32)
        signs = ((u32 >> 31) & 0x1).astype(np.uint8)
        mants = (u32 & 0x7FFFFF).astype(np.uint32)

        import zlib
        packed_signs = np.packbits(signs)
        sign_bytes = len(zlib.compress(packed_signs.tobytes(), level=1))

        mant_bytes_arr = np.zeros((mants.size, 3), dtype=np.uint8)
        mant_bytes_arr[:, 0] = (mants & 0xFF).astype(np.uint8)
        mant_bytes_arr[:, 1] = ((mants >> 8) & 0xFF).astype(np.uint8)
        mant_bytes_arr[:, 2] = ((mants >> 16) & 0xFF).astype(np.uint8)
        mant_bytes = len(zlib.compress(mant_bytes_arr.tobytes(), level=1))

        original_size = self.volume.nbytes
        total_compressed = exp_bytes + sign_bytes + mant_bytes
        ratio = original_size / total_compressed if total_compressed > 0 else 0

        return {
            "original_size": original_size,
            "compressed_size": total_compressed,
            "exponent_bytes": exp_bytes,
            "sign_bytes": sign_bytes,
            "mant_bytes": mant_bytes,
            "compression_ratio": ratio,
            "bits_per_voxel": (total_compressed * 8) / self.total_voxels,
            "sample_count": self.total_voxels
        }
