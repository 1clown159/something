#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import struct
import time
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch

try:
    from .common import ExperimentConfig, file_sha256
    from .roi import (
        detect_profile_rectangles,
        iter_rectangle_coords,
        rectangles_to_metadata,
        region_enabled,
        voxel_count_in_rectangles,
    )
    from .range_coder import RangeDecoder, RangeEncoder
    from .stage4 import (
        CAUSAL_EDGE_FEATURE_MODES,
        build_single_stage4_feature,
        build_stage4_features,
        feature_mode_to_in_channels,
        load_stage4_model,
        predictor_for_coord,
        reconstruct_exp_from_symbol,
        resolve_feature_mode,
        resolve_target_mode,
        target_symbol_for_coord,
    )
except ImportError:
    from common import ExperimentConfig, file_sha256
    from roi import (
        detect_profile_rectangles,
        iter_rectangle_coords,
        rectangles_to_metadata,
        region_enabled,
        voxel_count_in_rectangles,
    )
    from range_coder import RangeDecoder, RangeEncoder
    from stage4 import (
        CAUSAL_EDGE_FEATURE_MODES,
        build_single_stage4_feature,
        build_stage4_features,
        feature_mode_to_in_channels,
        load_stage4_model,
        predictor_for_coord,
        reconstruct_exp_from_symbol,
        resolve_feature_mode,
        resolve_target_mode,
        target_symbol_for_coord,
    )


MAGIC = b"S4RC"
VERSION = 1


def probs_to_cdf(probs: np.ndarray, total: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    freq = np.ones(256, dtype=np.int32)
    remain = total - 256
    raw = probs * remain
    add = np.floor(raw).astype(np.int32)
    freq += add
    shortfall = total - int(freq.sum())
    if shortfall > 0:
        order = np.argsort(raw - add)[::-1]
        freq[order[:shortfall]] += 1
    elif shortfall < 0:
        order = np.argsort(raw - add)
        for idx in order:
            if shortfall == 0:
                break
            removable = min(freq[idx] - 1, -shortfall)
            if removable > 0:
                freq[idx] -= removable
                shortfall += removable
    cdf = np.zeros(257, dtype=np.int32)
    cdf[1:] = np.cumsum(freq)
    if int(cdf[-1]) != total:
        raise ValueError("CDF total mismatch.")
    return cdf


def probs_to_cdfs(probs: np.ndarray, total: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        return probs_to_cdf(probs, total)[None, :]
    if probs.ndim != 2 or probs.shape[1] != 256:
        raise ValueError(f"Expected probs shape [N, 256], got {probs.shape}")

    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    freq = np.ones(probs.shape, dtype=np.int32)
    remain = total - 256
    raw = probs * remain
    add = np.floor(raw).astype(np.int32)
    freq += add

    shortfalls = total - freq.sum(axis=1)
    remainders = raw - add

    positive_rows = np.flatnonzero(shortfalls > 0)
    if positive_rows.size > 0:
        order_desc = np.argsort(remainders[positive_rows], axis=1)[:, ::-1]
        for row_offset, row_idx in enumerate(positive_rows.tolist()):
            shortfall = int(shortfalls[row_idx])
            if shortfall > 0:
                freq[row_idx, order_desc[row_offset, :shortfall]] += 1

    negative_rows = np.flatnonzero(shortfalls < 0)
    if negative_rows.size > 0:
        order_asc = np.argsort(remainders[negative_rows], axis=1)
        for row_offset, row_idx in enumerate(negative_rows.tolist()):
            need = int(-shortfalls[row_idx])
            for sym in order_asc[row_offset]:
                if need <= 0:
                    break
                removable = min(int(freq[row_idx, sym]) - 1, need)
                if removable > 0:
                    freq[row_idx, sym] -= removable
                    need -= removable

    cdfs = np.zeros((probs.shape[0], 257), dtype=np.int32)
    cdfs[:, 1:] = np.cumsum(freq, axis=1)
    if not np.all(cdfs[:, -1] == total):
        raise ValueError("CDF total mismatch.")
    return cdfs


def probs_to_cdfs_torch(probs: torch.Tensor, total: int) -> torch.Tensor:
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    if probs.ndim != 2 or probs.shape[1] != 256:
        raise ValueError(f"Expected probs shape [N, 256], got {tuple(probs.shape)}")

    probs32 = probs.float()
    probs32 = torch.clamp(probs32, min=1e-12, max=1.0)
    probs32 = probs32 / probs32.sum(dim=1, keepdim=True)

    freq = torch.ones_like(probs32, dtype=torch.int32)
    remain = total - 256
    raw = probs32 * float(remain)
    add = torch.floor(raw).to(torch.int32)
    freq = freq + add

    shortfalls = total - freq.sum(dim=1)
    remainders = raw - add.to(torch.float32)

    if torch.any(shortfalls < 0):
        return torch.from_numpy(probs_to_cdfs(probs32.cpu().numpy(), total)).to(device=probs.device)

    max_shortfall = int(shortfalls.max().item()) if shortfalls.numel() > 0 else 0
    if max_shortfall > 0:
        topk_idx = torch.topk(remainders, k=max_shortfall, dim=1, largest=True, sorted=True).indices
        rank = torch.arange(max_shortfall, device=probs.device, dtype=torch.int32).unsqueeze(0)
        updates = (rank < shortfalls.unsqueeze(1)).to(torch.int32)
        freq.scatter_add_(1, topk_idx, updates)

    cdfs = torch.zeros((probs.shape[0], 257), dtype=torch.int32, device=probs.device)
    cdfs[:, 1:] = torch.cumsum(freq, dim=1)
    if not torch.all(cdfs[:, -1] == total):
        raise ValueError("CDF total mismatch.")
    return cdfs


def _header_bytes(header: Dict[str, Any]) -> bytes:
    return json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8")


def write_bitstream(path: str, header: Dict[str, Any], payload: bytes) -> None:
    header_blob = _header_bytes(header)
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", VERSION))
        handle.write(struct.pack("<I", len(header_blob)))
        handle.write(header_blob)
        handle.write(payload)


def read_bitstream(path: str) -> Tuple[Dict[str, Any], bytes]:
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != MAGIC:
            raise ValueError("Invalid bitstream magic.")
        version = struct.unpack("<I", handle.read(4))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported bitstream version: {version}")
        header_len = struct.unpack("<I", handle.read(4))[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
        payload = handle.read()
    return header, payload


def _valid_region_from_header(header: Dict[str, Any], shape: Tuple[int, int, int]) -> np.ndarray | None:
    region = header.get("valid_region")
    if not region:
        return None
    rectangles = np.asarray(region.get("rectangles", []), dtype=np.int64)
    expected_shape = (int(shape[0]), 2)
    if rectangles.shape != expected_shape:
        raise ValueError(f"Invalid valid_region rectangles shape: expected {expected_shape}, got {rectangles.shape}")
    return rectangles


class Stage4RangeCodec:
    def __init__(
        self,
        checkpoint_path: str,
        config: ExperimentConfig,
        device: str = "cpu",
        feature_mode: str = "auto",
        target_mode: str = "auto",
        profile_timing: bool = False,
        inference_batch: int = 1,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        self.model, self.checkpoint = load_stage4_model(checkpoint_path, device=device)
        self.feature_mode = resolve_feature_mode(feature_mode, self.checkpoint)
        self.target_mode = resolve_target_mode(target_mode, self.checkpoint)
        self.mask = None
        self.patch_shape = tuple(self.checkpoint.get("config", {}).get("patch_shape", config.patch_shape))
        self.total_freq = int(config.range_total)
        self.profile_timing = bool(profile_timing)
        self.inference_batch = max(1, int(inference_batch))
        self._timing = self._new_timing()

    def _new_timing(self) -> Dict[str, float]:
        return {
            "patch_build_seconds": 0.0,
            "model_inference_seconds": 0.0,
            "cdf_quantization_seconds": 0.0,
            "range_coder_seconds": 0.0,
            "total_wall_seconds": 0.0,
            "voxel_count": 0.0,
            "inference_batch": float(self.inference_batch),
        }

    def _reset_timing(self) -> None:
        self._timing = self._new_timing()

    def _timing_report(self) -> Dict[str, float]:
        report = dict(self._timing)
        voxel_count = int(report["voxel_count"])
        if voxel_count > 0:
            report["seconds_per_voxel"] = float(report["total_wall_seconds"]) / voxel_count
            report["patch_build_us_per_voxel"] = 1e6 * float(report["patch_build_seconds"]) / voxel_count
            report["model_inference_us_per_voxel"] = 1e6 * float(report["model_inference_seconds"]) / voxel_count
            report["cdf_quantization_us_per_voxel"] = 1e6 * float(report["cdf_quantization_seconds"]) / voxel_count
            report["range_coder_us_per_voxel"] = 1e6 * float(report["range_coder_seconds"]) / voxel_count
        else:
            report["seconds_per_voxel"] = 0.0
            report["patch_build_us_per_voxel"] = 0.0
            report["model_inference_us_per_voxel"] = 0.0
            report["cdf_quantization_us_per_voxel"] = 0.0
            report["range_coder_us_per_voxel"] = 0.0
        known_seconds = (
            float(report["patch_build_seconds"])
            + float(report["model_inference_seconds"])
            + float(report["cdf_quantization_seconds"])
            + float(report["range_coder_seconds"])
        )
        report["other_overhead_seconds"] = max(0.0, float(report["total_wall_seconds"]) - known_seconds)
        return report

    def _resolve_valid_rectangles(self, exp_volume: np.ndarray) -> np.ndarray | None:
        if not region_enabled(self.config.valid_region_mode):
            return None
        return detect_profile_rectangles(
            exp_volume,
            min_nonzero_ratio=self.config.valid_region_min_nonzero_ratio,
            margin_traces=self.config.valid_region_margin_traces,
            group_size=self.config.valid_region_group_size,
        )

    def _predict_cdf(self, volume: np.ndarray, coord: Tuple[int, int, int]) -> np.ndarray:
        patch_start = time.perf_counter() if self.profile_timing else 0.0
        feature = build_single_stage4_feature(
            volume,
            coord,
            self.patch_shape,
            mask=self.mask,
            feature_mode=self.feature_mode,
            target_mode=self.target_mode,
        )
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += 1

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            logits = self.model(feature.to(self.device))
            probs = torch.softmax(logits, dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        if probs.is_cuda:
            cdf = probs_to_cdfs_torch(probs, self.total_freq)[0].cpu().numpy()
        else:
            cdf = probs_to_cdf(probs.cpu().numpy()[0], self.total_freq)
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdf

    def _predict_cdfs_batch(self, volume: np.ndarray, coords: Sequence[Tuple[int, int, int]]) -> List[np.ndarray]:
        if len(coords) == 1:
            return [self._predict_cdf(volume, coords[0])]

        coords_np = np.asarray(coords, dtype=np.int64)
        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features, _ = build_stage4_features(
            volume,
            coords_np,
            self.patch_shape,
            feature_mode=self.feature_mode,
            target_mode=self.target_mode,
        )
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += len(coords)

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            logits = self.model(features.to(self.device))
            probs = torch.softmax(logits, dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        if probs.is_cuda:
            cdfs = list(probs_to_cdfs_torch(probs, self.total_freq).cpu().numpy())
        else:
            cdfs = list(probs_to_cdfs(probs.cpu().numpy(), self.total_freq))
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdfs

    def encode_exponents(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        exp_volume = np.asarray(exp_volume, dtype=np.uint8)
        encoder = RangeEncoder()
        total_voxels = int(np.prod(exp_volume.shape))
        valid_rectangles = self._resolve_valid_rectangles(exp_volume)
        modeled_voxels = total_voxels if valid_rectangles is None else voxel_count_in_rectangles(valid_rectangles, exp_volume.shape[2])
        self._reset_timing()
        total_start = time.perf_counter()

        coord_batch: List[Tuple[int, int, int]] = []
        symbol_batch: List[int] = []

        def flush_batch() -> None:
            if not coord_batch:
                return
            cdfs = self._predict_cdfs_batch(exp_volume, coord_batch)
            range_start = time.perf_counter() if self.profile_timing else 0.0
            for cdf, symbol in zip(cdfs, symbol_batch):
                encoder.encode_symbol(cdf, symbol)
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start
            coord_batch.clear()
            symbol_batch.clear()

        for coord in iter_rectangle_coords(exp_volume.shape, valid_rectangles):
            coord_batch.append(coord)
            symbol_batch.append(target_symbol_for_coord(exp_volume, coord, self.target_mode))
            if len(coord_batch) >= self.inference_batch:
                flush_batch()
        flush_batch()

        payload = encoder.finish()
        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = {
            "codec": "stage4_causal_range",
            "shape": list(exp_volume.shape),
            "patch_shape": list(self.patch_shape),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "predictor": "loco_i_2d" if self.target_mode == "residual" else None,
            "total_freq": self.total_freq,
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "voxel_count": modeled_voxels,
            "total_volume_voxels": total_voxels,
            "valid_region": rectangles_to_metadata(
                valid_rectangles,
                exp_volume.shape,
                self.config.valid_region_mode,
                self.config.valid_region_min_nonzero_ratio,
                self.config.valid_region_margin_traces,
                self.config.valid_region_group_size,
            ),
        }
        write_bitstream(bitstream_path, header, payload)
        header_size = len(_header_bytes(header)) + 12
        total_bytes = len(payload) + header_size
        return {
            "bitstream_path": os.path.abspath(bitstream_path),
            "payload_bytes": len(payload),
            "header_bytes": header_size,
            "total_bytes": total_bytes,
            "bits_per_modeled_voxel": 8.0 * total_bytes / max(modeled_voxels, 1),
            "bits_per_total_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "valid_region": header["valid_region"],
            "timing": self._timing_report(),
        }

    def decode_exponents(self, bitstream_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        header, payload = read_bitstream(bitstream_path)
        if header.get("codec", "stage4_causal_range") != "stage4_causal_range":
            raise ValueError(f"Unsupported raster bitstream codec: {header.get('codec')}")
        shape = tuple(int(v) for v in header["shape"])
        if tuple(header["patch_shape"]) != tuple(self.patch_shape):
            raise ValueError("Patch shape mismatch between bitstream and checkpoint.")
        if int(header["total_freq"]) != self.total_freq:
            raise ValueError("Range total mismatch between bitstream and config.")
        header_feature_mode = header.get("feature_mode", "strict")
        if header_feature_mode != self.feature_mode:
            raise ValueError(f"Feature mode mismatch: bitstream={header_feature_mode}, codec={self.feature_mode}")
        header_target_mode = header.get("target_mode", "raw")
        if header_target_mode != self.target_mode:
            raise ValueError(f"Target mode mismatch: bitstream={header_target_mode}, codec={self.target_mode}")
        valid_rectangles = _valid_region_from_header(header, shape)
        decoded = np.zeros(shape, dtype=np.uint8)
        decoder = RangeDecoder(payload)
        self._reset_timing()
        total_start = time.perf_counter()
        for coord in iter_rectangle_coords(shape, valid_rectangles):
            cdf = self._predict_cdf(decoded, coord)
            range_start = time.perf_counter() if self.profile_timing else 0.0
            symbol = int(decoder.decode_symbol(cdf))
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start
            if self.target_mode == "raw":
                decoded[coord] = symbol
            else:
                pred = predictor_for_coord(decoded, coord)
                decoded[coord] = reconstruct_exp_from_symbol(symbol, pred, self.target_mode)
        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = dict(header)
        header["timing"] = self._timing_report()
        return decoded, header

    def roundtrip(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        encode_metrics = self.encode_exponents(exp_volume, bitstream_path)
        decoded, header = self.decode_exponents(bitstream_path)
        ok = np.array_equal(np.asarray(exp_volume, dtype=np.uint8), decoded)
        return {
            "ok": bool(ok),
            "encode": encode_metrics,
            "header": header,
        }


class Stage4GlobalDiagonalRangeCodec(Stage4RangeCodec):
    def __init__(
        self,
        checkpoint_path: str,
        config: ExperimentConfig,
        device: str = "cpu",
        feature_mode: str = "auto",
        target_mode: str = "auto",
        profile_timing: bool = False,
        inference_batch: int = 1,
        progress: bool = False,
        progress_label: str = "",
        progress_interval_diags: int = 25,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            feature_mode=feature_mode,
            target_mode=target_mode,
            profile_timing=profile_timing,
            inference_batch=inference_batch,
        )
        if self.feature_mode != "diagonal_causal_edge":
            raise ValueError("codec-layout=global_diag requires --feature-mode diagonal_causal_edge or a diagonal checkpoint.")
        if region_enabled(self.config.valid_region_mode):
            raise NotImplementedError("global_diag currently supports --valid-region-mode none only.")
        self.progress = bool(progress)
        self.progress_label = str(progress_label)
        self.progress_interval_diags = max(1, int(progress_interval_diags))
        patch_h, patch_w = self.patch_shape
        t_half, s_half = (dim // 2 for dim in self.patch_shape)
        jj, kk = np.indices(self.patch_shape, dtype=np.int32)
        self._global_patch_plan = {
            "patch_h": patch_h,
            "patch_w": patch_w,
            "flat_len": patch_h * patch_w,
            "dt": (jj - t_half).reshape(-1),
            "ds": (kk - s_half).reshape(-1),
            "in_channels": feature_mode_to_in_channels(self.feature_mode, self.target_mode),
        }
        self._global_patch_plan_torch: Dict[str, Dict[str, torch.Tensor]] = {}
        self._global_feature_scratch_torch: Dict[str, Dict[str, torch.Tensor]] = {}
        self._global_feature_scratch_torch_capacity: Dict[str, int] = {}

    def _use_cuda_global_fastpath(self) -> bool:
        return self.feature_mode in CAUSAL_EDGE_FEATURE_MODES and str(self.device).startswith("cuda")

    def _get_global_patch_plan_torch(self, device: str) -> Dict[str, torch.Tensor]:
        plan = self._global_patch_plan_torch.get(device)
        if plan is not None:
            return plan
        plan = {
            "dt": torch.from_numpy(self._global_patch_plan["dt"]).to(device=device, dtype=torch.int32),
            "ds": torch.from_numpy(self._global_patch_plan["ds"]).to(device=device, dtype=torch.int32),
        }
        self._global_patch_plan_torch[device] = plan
        return plan

    def _ensure_global_feature_scratch_torch(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        capacity = self._global_feature_scratch_torch_capacity.get(device, 0)
        if batch_size <= capacity:
            return self._global_feature_scratch_torch[device]

        patch_h = int(self._global_patch_plan["patch_h"])
        patch_w = int(self._global_patch_plan["patch_w"])
        flat_len = int(self._global_patch_plan["flat_len"])
        in_channels = int(self._global_patch_plan["in_channels"])
        torch_device = torch.device(device)
        scratch = {
            "tt": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "ss": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "ct": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "cs": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "linear": torch.empty((batch_size, flat_len), dtype=torch.long, device=torch_device),
            "flat": torch.empty((batch_size, 4, flat_len), dtype=torch.float32, device=torch_device),
            "features": torch.empty((batch_size, in_channels, patch_h, patch_w), dtype=torch.float32, device=torch_device),
            "inb": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "real": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "remaining": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "same_trace_mask": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "prev_trace_mask": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "residual_flat": torch.empty((batch_size, flat_len), dtype=torch.float32, device=torch_device),
        }
        self._global_feature_scratch_torch[device] = scratch
        self._global_feature_scratch_torch_capacity[device] = batch_size
        return scratch

    def _build_global_feature_batch_fast_torch(self, volume: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch_size = int(coords.shape[0])
        patch_h = int(self._global_patch_plan["patch_h"])
        patch_w = int(self._global_patch_plan["patch_w"])
        in_channels = int(self._global_patch_plan["in_channels"])
        _, height, width = volume.shape
        device = volume.device
        scale = 1.0 / 255.0

        plan = self._get_global_patch_plan_torch(str(device))
        scratch = self._ensure_global_feature_scratch_torch(batch_size, str(device))
        volume_flat = volume.reshape(-1)
        plane_stride = int(height * width)

        p = coords[:, 0].to(torch.long)
        t = coords[:, 1].to(torch.int32)
        s = coords[:, 2].to(torch.int32)
        t_col = t.unsqueeze(1)
        s_col = s.unsqueeze(1)
        diag_col = (t + s).unsqueeze(1)

        tt = scratch["tt"][:batch_size]
        ss = scratch["ss"][:batch_size]
        torch.add(t_col, plan["dt"].unsqueeze(0), out=tt)
        torch.add(s_col, plan["ds"].unsqueeze(0), out=ss)

        inb = scratch["inb"][:batch_size]
        inb.copy_((tt >= 0) & (tt < height) & (ss >= 0) & (ss < width))
        real = scratch["real"][:batch_size]
        real.copy_(inb)
        real.logical_and_((tt + ss) < diag_col)

        ct = scratch["ct"][:batch_size]
        cs = scratch["cs"][:batch_size]
        torch.clamp(tt, min=0, max=height - 1, out=ct)
        torch.clamp(ss, min=0, max=width - 1, out=cs)

        linear = scratch["linear"][:batch_size]
        linear.copy_(p.view(batch_size, 1) * plane_stride + ct.to(torch.long) * width + cs.to(torch.long))
        gathered = volume_flat.gather(0, linear.reshape(-1)).reshape(batch_size, -1).to(torch.float32) * scale

        flat = scratch["flat"][:batch_size]
        flat.zero_()
        real_mask_f = real.to(torch.float32)
        flat[:, 0, :].add_(gathered * real_mask_f)
        flat[:, 1, :].copy_(real_mask_f)
        flat[:, 2, :].copy_(real_mask_f)

        remaining = scratch["remaining"][:batch_size]
        remaining.copy_(~real)

        same_trace_mask = scratch["same_trace_mask"][:batch_size]
        same_trace_mask.copy_(remaining)
        same_trace_mask.logical_and_(ct == t_col)
        same_trace_mask.logical_and_((s > 0).unsqueeze(1))
        left_s = torch.clamp(s.to(torch.long) - 1, min=0)
        left_idx = p * plane_stride + t.to(torch.long) * width + left_s
        left_values = volume_flat.gather(0, left_idx).to(torch.float32).view(batch_size, 1) * scale
        flat[:, 0, :].add_(left_values * same_trace_mask.to(torch.float32))

        prev_trace_mask = scratch["prev_trace_mask"][:batch_size]
        prev_trace_mask.copy_(remaining)
        prev_trace_mask.logical_and_(~same_trace_mask)
        prev_trace_mask.logical_and_((t > 0).unsqueeze(1))
        prev_t = torch.clamp(t.to(torch.long) - 1, min=0)
        prev_ct = torch.minimum(ct.to(torch.long), prev_t.unsqueeze(1))
        prev_trace_mask.logical_and_((prev_ct.to(torch.int32) + cs) < diag_col)
        prev_idx = p.view(batch_size, 1) * plane_stride + prev_ct * width + cs.to(torch.long)
        prev_values = volume_flat.gather(0, prev_idx.reshape(-1)).reshape(batch_size, -1).to(torch.float32) * scale
        flat[:, 0, :].add_(prev_values * prev_trace_mask.to(torch.float32))

        mapped_mask_f = same_trace_mask.to(torch.float32) + prev_trace_mask.to(torch.float32)
        flat[:, 1, :].add_(mapped_mask_f)
        flat[:, 3, :].copy_(mapped_mask_f)

        feature4 = flat.reshape((batch_size, 4, patch_h, patch_w))
        if self.target_mode == "raw":
            return feature4

        up_t = torch.clamp(t.to(torch.long) - 1, min=0)
        up_idx = p * plane_stride + up_t * width + s.to(torch.long)
        up_left_idx = p * plane_stride + up_t * width + left_s
        up_values = volume_flat.gather(0, up_idx).to(torch.int16)
        up_left_values = volume_flat.gather(0, up_left_idx).to(torch.int16)
        left_values_i16 = volume_flat.gather(0, left_idx).to(torch.int16)

        pred_i16 = torch.zeros(batch_size, dtype=torch.int16, device=device)
        has_t = t > 0
        has_s = s > 0
        both = has_t & has_s
        pred_i16 = torch.where((~has_t) & has_s, left_values_i16, pred_i16)
        pred_i16 = torch.where(has_t & (~has_s), up_values, pred_i16)
        pred_i16 = torch.where(both, torch.clamp(left_values_i16 + up_values - up_left_values, 0, 255), pred_i16)

        features = scratch["features"][:batch_size]
        features.zero_()
        features[:, :4] = feature4
        features[:, 4] = pred_i16.to(torch.float32).view(batch_size, 1, 1) * scale

        if self.target_mode == "residual":
            values_u8 = torch.round(flat[:, 0, :] * 255.0).to(torch.uint8)
            residual_u8 = ((values_u8.to(torch.int16) - pred_i16.view(batch_size, 1)) & 0xFF).to(torch.uint8)
            residual_flat = scratch["residual_flat"][:batch_size]
            residual_flat.copy_(residual_u8.to(torch.float32) * scale)
            residual_flat.mul_(flat[:, 1, :])
            features[:, 5] = residual_flat.reshape((batch_size, patch_h, patch_w))
        return features[:, :in_channels]

    def _predict_cdfs_global_batch_cuda(self, volume: torch.Tensor, coords_np: np.ndarray) -> np.ndarray:
        if coords_np.size == 0:
            return np.zeros((0, 257), dtype=np.int32)
        coords = torch.from_numpy(coords_np).to(device=self.device, dtype=torch.long)

        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features = self._build_global_feature_batch_fast_torch(volume, coords)
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += int(coords.shape[0])

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            probs = torch.softmax(self.model(features), dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        cdfs = probs_to_cdfs_torch(probs, self.total_freq).cpu().numpy()
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdfs

    def _predict_boundaries_global_batch_cuda(self, volume: torch.Tensor, coords_np: np.ndarray, symbols_np: np.ndarray):
        """GPU predict CDFs + extract boundary values → minimal CPU transfer.
        
        Returns (total, sym_low, sym_high) as (int, np.ndarray, np.ndarray).
        Only transfers 2 ints per voxel instead of 257, cutting transfer by 128x.
        """
        if coords_np.size == 0:
            return self.total_freq, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        coords = torch.from_numpy(coords_np).to(device=self.device, dtype=torch.long)

        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features = self._build_global_feature_batch_fast_torch(volume, coords)
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += int(coords.shape[0])

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            probs = torch.softmax(self.model(features), dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        cdfs = probs_to_cdfs_torch(probs, self.total_freq)  # stays on GPU
        symbols_t = torch.from_numpy(symbols_np).to(device=self.device, dtype=torch.long)
        idx = torch.arange(cdfs.shape[0], device=self.device)
        total = int(cdfs[0, 256].item())
        sym_low = cdfs[idx, symbols_t].cpu().numpy()
        sym_high = cdfs[idx, symbols_t + 1].cpu().numpy()
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return total, sym_low, sym_high

    def _target_symbols_for_coords(self, volume: np.ndarray, coords_np: np.ndarray) -> np.ndarray:
        p = coords_np[:, 0]
        t = coords_np[:, 1]
        s = coords_np[:, 2]
        values = volume[p, t, s].astype(np.int16)
        if self.target_mode == "raw":
            return values.astype(np.uint8)

        pred = np.zeros(coords_np.shape[0], dtype=np.int16)
        has_t = t > 0
        has_s = s > 0
        if np.any((~has_t) & has_s):
            mask = (~has_t) & has_s
            pred[mask] = volume[p[mask], t[mask], s[mask] - 1].astype(np.int16)
        if np.any(has_t & (~has_s)):
            mask = has_t & (~has_s)
            pred[mask] = volume[p[mask], t[mask] - 1, s[mask]].astype(np.int16)
        if np.any(has_t & has_s):
            mask = has_t & has_s
            left = volume[p[mask], t[mask], s[mask] - 1].astype(np.int16)
            up = volume[p[mask], t[mask] - 1, s[mask]].astype(np.int16)
            up_left = volume[p[mask], t[mask] - 1, s[mask] - 1].astype(np.int16)
            pred[mask] = np.clip(left + up - up_left, 0, 255).astype(np.int16)
        return ((values - pred) & 0xFF).astype(np.uint8)

    def _reconstruct_values_for_coords(self, decoded: np.ndarray, coords_np: np.ndarray, symbols: np.ndarray) -> np.ndarray:
        if self.target_mode == "raw":
            return np.asarray(symbols, dtype=np.uint8)

        p = coords_np[:, 0]
        t = coords_np[:, 1]
        s = coords_np[:, 2]
        pred = np.zeros(coords_np.shape[0], dtype=np.int16)
        has_t = t > 0
        has_s = s > 0
        if np.any((~has_t) & has_s):
            mask = (~has_t) & has_s
            pred[mask] = decoded[p[mask], t[mask], s[mask] - 1].astype(np.int16)
        if np.any(has_t & (~has_s)):
            mask = has_t & (~has_s)
            pred[mask] = decoded[p[mask], t[mask] - 1, s[mask]].astype(np.int16)
        if np.any(has_t & has_s):
            mask = has_t & has_s
            left = decoded[p[mask], t[mask], s[mask] - 1].astype(np.int16)
            up = decoded[p[mask], t[mask] - 1, s[mask]].astype(np.int16)
            up_left = decoded[p[mask], t[mask] - 1, s[mask] - 1].astype(np.int16)
            pred[mask] = np.clip(left + up - up_left, 0, 255).astype(np.int16)
        return ((pred + np.asarray(symbols, dtype=np.int16)) & 0xFF).astype(np.uint8)

    def _iter_global_diagonal_coord_batches_with_diag(self, shape: Tuple[int, int, int]) -> Iterator[Tuple[int, int, np.ndarray]]:
        profiles, height, width = (int(v) for v in shape)
        total_diags = height + width - 1
        max_diag_len = min(height, width)
        max_coords = np.empty((profiles * max_diag_len, 3), dtype=np.int64)
        profile_ids = np.arange(profiles, dtype=np.int64)
        for diag in range(height + width - 1):
            t_min = max(0, diag - width + 1)
            t_max = min(height - 1, diag)
            ts = np.arange(t_min, t_max + 1, dtype=np.int64)
            diag_len = int(ts.size)
            if diag_len == 0:
                continue
            n = profiles * diag_len
            coords = max_coords[:n]
            coords[:, 0] = np.repeat(profile_ids, diag_len)
            coords[:, 1] = np.tile(ts, profiles)
            coords[:, 2] = diag - coords[:, 1]
            for start in range(0, n, self.inference_batch):
                yield diag, total_diags, coords[start : start + self.inference_batch]

    def _iter_global_diagonal_chunks_with_diag(self, shape: Tuple[int, int, int]) -> Iterator[Tuple[int, int, List[Tuple[int, int, int]]]]:
        for diag, total_diags, coords in self._iter_global_diagonal_coord_batches_with_diag(shape):
            yield diag, total_diags, [tuple(int(v) for v in row) for row in coords]

    def _iter_global_diagonal_chunks(self, shape: Tuple[int, int, int]) -> Iterator[List[Tuple[int, int, int]]]:
        for _, _, chunk in self._iter_global_diagonal_chunks_with_diag(shape):
            yield chunk

    def _print_global_progress(
        self,
        phase: str,
        diag: int,
        total_diags: int,
        processed_voxels: int,
        total_voxels: int,
        *,
        force: bool = False,
    ) -> None:
        if not self.progress:
            return
        should_print = force or diag == 0 or (diag + 1) == total_diags or (diag % self.progress_interval_diags) == 0
        if not should_print:
            return
        label = f" {self.progress_label}" if self.progress_label else ""
        pct = 100.0 * processed_voxels / max(total_voxels, 1)
        print(
            f"[Progress]{label} {phase}: diag {diag + 1}/{total_diags} "
            f"({pct:.2f}%, {processed_voxels}/{total_voxels} voxels)",
            flush=True,
        )

    def encode_exponents(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        exp_volume = np.asarray(exp_volume, dtype=np.uint8)
        total_voxels = int(np.prod(exp_volume.shape))
        encoder = RangeEncoder()
        use_cuda_fastpath = self._use_cuda_global_fastpath()
        if use_cuda_fastpath:
            volume_t = torch.from_numpy(exp_volume).to(device=self.device)
        self._reset_timing()
        total_start = time.perf_counter()

        processed_voxels = 0
        last_reported_diag = -1
        for diag, total_diags, coords_np in self._iter_global_diagonal_coord_batches_with_diag(exp_volume.shape):
            if diag != last_reported_diag:
                self._print_global_progress("encode", diag, total_diags, processed_voxels, total_voxels)
                last_reported_diag = diag
            if use_cuda_fastpath:
                symbols = self._target_symbols_for_coords(exp_volume, coords_np)
                total, sym_low, sym_high = self._predict_boundaries_global_batch_cuda(volume_t, coords_np, symbols)
                range_start = time.perf_counter() if self.profile_timing else 0.0
                encoder.encode_boundaries(total, sym_low, sym_high)
            else:
                coord_batch = [tuple(int(v) for v in row) for row in coords_np]
                cdfs = self._predict_cdfs_batch(exp_volume, coord_batch)
                range_start = time.perf_counter() if self.profile_timing else 0.0
                symbols = self._target_symbols_for_coords(exp_volume, coords_np)
                encoder.encode_symbols(cdfs, symbols)
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start
            processed_voxels += int(coords_np.shape[0])
        self._print_global_progress("encode", max(0, exp_volume.shape[1] + exp_volume.shape[2] - 2), exp_volume.shape[1] + exp_volume.shape[2] - 1, processed_voxels, total_voxels, force=True)

        payload = encoder.finish()
        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = {
            "codec": "stage4_causal_range_global_diag",
            "shape": list(exp_volume.shape),
            "patch_shape": list(self.patch_shape),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "predictor": "loco_i_2d_global_diag" if self.target_mode == "residual" else None,
            "total_freq": self.total_freq,
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "voxel_count": total_voxels,
            "total_volume_voxels": total_voxels,
            "encode_schedule": "profile_global_diagonal_batch",
            "valid_region": None,
        }
        write_bitstream(bitstream_path, header, payload)
        header_size = len(_header_bytes(header)) + 12
        total_bytes = len(payload) + header_size
        return {
            "bitstream_path": os.path.abspath(bitstream_path),
            "payload_bytes": len(payload),
            "header_bytes": header_size,
            "total_bytes": total_bytes,
            "bits_per_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "bits_per_modeled_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "bits_per_total_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "codec_layout": "global_diag",
            "encode_schedule": header["encode_schedule"],
            "timing": self._timing_report(),
        }

    def decode_exponents(self, bitstream_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        header, payload = read_bitstream(bitstream_path)
        if header.get("codec") != "stage4_causal_range_global_diag":
            raise ValueError(f"Unsupported global_diag bitstream codec: {header.get('codec')}")
        shape = tuple(int(v) for v in header["shape"])
        if tuple(header["patch_shape"]) != tuple(self.patch_shape):
            raise ValueError("Patch shape mismatch between bitstream and checkpoint.")
        if int(header["total_freq"]) != self.total_freq:
            raise ValueError("Range total mismatch between bitstream and config.")
        header_feature_mode = header.get("feature_mode", "strict")
        if header_feature_mode != self.feature_mode:
            raise ValueError(f"Feature mode mismatch: bitstream={header_feature_mode}, codec={self.feature_mode}")
        header_target_mode = header.get("target_mode", "raw")
        if header_target_mode != self.target_mode:
            raise ValueError(f"Target mode mismatch: bitstream={header_target_mode}, codec={self.target_mode}")

        decoded = np.zeros(shape, dtype=np.uint8)
        use_cuda_fastpath = self._use_cuda_global_fastpath()
        if use_cuda_fastpath:
            decoded_t = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        decoder = RangeDecoder(payload)
        self._reset_timing()
        total_start = time.perf_counter()

        total_voxels = int(np.prod(shape))
        processed_voxels = 0
        last_reported_diag = -1
        for diag, total_diags, coords_np in self._iter_global_diagonal_coord_batches_with_diag(shape):
            if diag != last_reported_diag:
                self._print_global_progress("decode", diag, total_diags, processed_voxels, total_voxels)
                last_reported_diag = diag
            if use_cuda_fastpath:
                cdfs = self._predict_cdfs_global_batch_cuda(decoded_t, coords_np)
            else:
                coord_batch = [tuple(int(v) for v in row) for row in coords_np]
                cdfs = self._predict_cdfs_batch(decoded, coord_batch)
            range_start = time.perf_counter() if self.profile_timing else 0.0
            symbols = decoder.decode_symbols(cdfs)
            decoded_values = self._reconstruct_values_for_coords(decoded, coords_np, symbols)
            decoded[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]] = decoded_values
            if use_cuda_fastpath and decoded_values.size > 0:
                coords_t = torch.from_numpy(coords_np).to(device=self.device, dtype=torch.long)
                decoded_t[coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = torch.from_numpy(decoded_values).to(
                    device=self.device,
                    dtype=torch.uint8,
                )
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start
            processed_voxels += int(coords_np.shape[0])
        self._print_global_progress("decode", max(0, shape[1] + shape[2] - 2), shape[1] + shape[2] - 1, processed_voxels, total_voxels, force=True)

        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = dict(header)
        header["timing"] = self._timing_report()
        header["decode_schedule"] = header.get("encode_schedule", "profile_global_diagonal_batch")
        return decoded, header


def iter_tiles(shape: Tuple[int, int, int], tile_shape: Tuple[int, int]) -> Sequence[Dict[str, int]]:
    tile_h, tile_w = (max(1, int(v)) for v in tile_shape)
    tiles = []
    for p in range(int(shape[0])):
        for t0 in range(0, int(shape[1]), tile_h):
            h = min(tile_h, int(shape[1]) - t0)
            for s0 in range(0, int(shape[2]), tile_w):
                w = min(tile_w, int(shape[2]) - s0)
                tiles.append({"p": p, "t0": t0, "s0": s0, "h": h, "w": w})
    return tiles


class Stage4TileRangeCodec(Stage4RangeCodec):
    def __init__(
        self,
        checkpoint_path: str,
        config: ExperimentConfig,
        device: str = "cpu",
        feature_mode: str = "auto",
        target_mode: str = "auto",
        profile_timing: bool = False,
        inference_batch: int = 1,
        tile_shape: Tuple[int, int] = (64, 64),
    ) -> None:
        if region_enabled(config.valid_region_mode):
            raise NotImplementedError("20260414 currently supports valid-region coding with codec-layout=raster only.")
        super().__init__(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            feature_mode=feature_mode,
            target_mode=target_mode,
            profile_timing=profile_timing,
            inference_batch=inference_batch,
        )
        self.tile_shape = tuple(max(1, int(v)) for v in tile_shape)
        patch_h, patch_w = self.patch_shape
        t_half, s_half = (dim // 2 for dim in self.patch_shape)
        jj, kk = np.indices(self.patch_shape, dtype=np.int32)
        self._tile_patch_plan = {
            "patch_h": patch_h,
            "patch_w": patch_w,
            "flat_len": patch_h * patch_w,
            "dt": (jj - t_half).reshape(-1),
            "ds": (kk - s_half).reshape(-1),
            "in_channels": feature_mode_to_in_channels(self.feature_mode, self.target_mode),
        }
        self._tile_coord_cache = self._build_tile_coord_cache()
        self._tile_coord_cache_torch: Dict[str, List[List[Dict[str, torch.Tensor]]]] = {}
        self._tile_feature_scratch: Dict[str, np.ndarray] = {}
        self._tile_feature_scratch_capacity = 0
        self._tile_feature_scratch_torch: Dict[str, Dict[str, torch.Tensor]] = {}
        self._tile_feature_scratch_torch_capacity: Dict[str, int] = {}

    def _use_cuda_tile_fastpath(self) -> bool:
        return self.feature_mode in CAUSAL_EDGE_FEATURE_MODES and str(self.device).startswith("cuda")

    def _tile_context_mask(self, tt: np.ndarray, ss: np.ndarray, t: int, s: int) -> np.ndarray:
        if self.feature_mode == "causal_edge":
            return (tt < t) | ((tt == t) & (ss < s))
        if self.feature_mode == "diagonal_causal_edge":
            return (tt + ss) < (t + s)
        raise ValueError(f"Unsupported causal edge feature mode: {self.feature_mode}")

    def _build_tile_coord_cache(self) -> List[List[Dict[str, np.ndarray]]]:
        dt = self._tile_patch_plan["dt"]
        ds = self._tile_patch_plan["ds"]
        tile_h, tile_w = self.tile_shape
        cache: List[List[Dict[str, np.ndarray]]] = []
        for t in range(tile_h):
            row: List[Dict[str, np.ndarray]] = []
            for s in range(tile_w):
                tt = (t + dt).astype(np.int32, copy=True)
                ss = (s + ds).astype(np.int32, copy=True)
                row.append(
                    {
                        "tt": tt,
                        "ss": ss,
                        "causal": self._tile_context_mask(tt, ss, t, s),
                        "same_trace": (tt == t),
                    }
                )
            cache.append(row)
        return cache

    def _get_tile_coord_cache_torch(self, device: str) -> List[List[Dict[str, torch.Tensor]]]:
        cache = self._tile_coord_cache_torch.get(device)
        if cache is not None:
            return cache

        cache = []
        for row in self._tile_coord_cache:
            torch_row: List[Dict[str, torch.Tensor]] = []
            for item in row:
                torch_row.append(
                    {
                        "tt": torch.from_numpy(item["tt"]).to(device=device, dtype=torch.int32),
                        "ss": torch.from_numpy(item["ss"]).to(device=device, dtype=torch.int32),
                        "causal": torch.from_numpy(item["causal"]).to(device=device),
                        "same_trace": torch.from_numpy(item["same_trace"]).to(device=device),
                        "linear_clip": torch.from_numpy(
                            np.clip(item["tt"], 0, self.tile_shape[0] - 1).astype(np.int64) * self.tile_shape[1]
                            + np.clip(item["ss"], 0, self.tile_shape[1] - 1).astype(np.int64)
                        ).to(device=device, dtype=torch.long),
                    }
                )
            cache.append(torch_row)
        self._tile_coord_cache_torch[device] = cache
        return cache

    def _ensure_tile_feature_scratch(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size <= self._tile_feature_scratch_capacity:
            return self._tile_feature_scratch

        flat_len = int(self._tile_patch_plan["flat_len"])
        patch_h = int(self._tile_patch_plan["patch_h"])
        patch_w = int(self._tile_patch_plan["patch_w"])
        in_channels = int(self._tile_patch_plan["in_channels"])
        self._tile_feature_scratch = {
            "plane_ids": np.zeros((batch_size, flat_len), dtype=np.int32),
            "tt": np.zeros((batch_size, flat_len), dtype=np.int32),
            "ss": np.zeros((batch_size, flat_len), dtype=np.int32),
            "ct": np.zeros((batch_size, flat_len), dtype=np.int32),
            "cs": np.zeros((batch_size, flat_len), dtype=np.int32),
            "mt": np.zeros((batch_size, flat_len), dtype=np.int32),
            "ms": np.zeros((batch_size, flat_len), dtype=np.int32),
            "heights": np.zeros((batch_size, 1), dtype=np.int32),
            "widths": np.zeros((batch_size, 1), dtype=np.int32),
            "flat": np.zeros((batch_size, 4, flat_len), dtype=np.float32),
            "features": np.zeros((batch_size, in_channels, patch_h, patch_w), dtype=np.float32),
            "residual_flat": np.zeros((batch_size, flat_len), dtype=np.float32),
            "inb": np.zeros((batch_size, flat_len), dtype=bool),
            "real": np.zeros((batch_size, flat_len), dtype=bool),
            "remaining": np.zeros((batch_size, flat_len), dtype=bool),
            "mapped_valid": np.zeros((batch_size, flat_len), dtype=bool),
            "same_trace_mask": np.zeros((batch_size, flat_len), dtype=bool),
            "prev_trace_mask": np.zeros((batch_size, flat_len), dtype=bool),
            "usable": np.zeros((batch_size, flat_len), dtype=bool),
            "preds": np.zeros(batch_size, dtype=np.uint8),
        }
        self._tile_feature_scratch_capacity = batch_size
        return self._tile_feature_scratch

    def _prepare_tile_planes(self, tile_states: List[Dict[str, Any]], volume_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tile_h, tile_w = self.tile_shape
        planes = np.zeros((len(tile_states), tile_h, tile_w), dtype=np.uint8)
        heights = np.zeros(len(tile_states), dtype=np.int32)
        widths = np.zeros(len(tile_states), dtype=np.int32)
        for idx, tile_state in enumerate(tile_states):
            h = int(tile_state["h"])
            w = int(tile_state["w"])
            heights[idx] = h
            widths[idx] = w
            tile_state["plane_index"] = idx
            tile_view = np.asarray(tile_state[volume_key][0], dtype=np.uint8)
            planes[idx, :h, :w] = tile_view
        return planes, heights, widths

    def _prepare_tile_tensors(
        self,
        planes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        return (
            torch.from_numpy(planes).to(device=device),
            torch.from_numpy(heights).to(device=device, dtype=torch.int32),
            torch.from_numpy(widths).to(device=device, dtype=torch.int32),
        )

    def _prepare_schedule_tensors(self, schedule: List[Tuple[int, int, np.ndarray]]) -> List[Tuple[int, int, torch.Tensor]]:
        device = self.device
        return [
            (t, s, torch.from_numpy(active_indices).to(device=device, dtype=torch.long))
            for t, s, active_indices in schedule
        ]

    def _ensure_tile_feature_scratch_torch(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        capacity = self._tile_feature_scratch_torch_capacity.get(device, 0)
        if batch_size <= capacity:
            return self._tile_feature_scratch_torch[device]

        tile_h, tile_w = self.tile_shape
        patch_h = int(self._tile_patch_plan["patch_h"])
        patch_w = int(self._tile_patch_plan["patch_w"])
        flat_len = int(self._tile_patch_plan["flat_len"])
        in_channels = int(self._tile_patch_plan["in_channels"])
        torch_device = torch.device(device)

        scratch = {
            "plane_subset": torch.empty((batch_size, tile_h, tile_w), dtype=torch.uint8, device=torch_device),
            "heights": torch.empty(batch_size, dtype=torch.int32, device=torch_device),
            "widths": torch.empty(batch_size, dtype=torch.int32, device=torch_device),
            "flat": torch.empty((batch_size, 4, flat_len), dtype=torch.float32, device=torch_device),
            "features": torch.empty((batch_size, in_channels, patch_h, patch_w), dtype=torch.float32, device=torch_device),
            "inb": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "real": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "remaining": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "same_trace_mask": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "prev_trace_mask": torch.empty((batch_size, flat_len), dtype=torch.bool, device=torch_device),
            "ct": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "cs": torch.empty((batch_size, flat_len), dtype=torch.int32, device=torch_device),
            "residual_flat": torch.empty((batch_size, flat_len), dtype=torch.float32, device=torch_device),
        }
        self._tile_feature_scratch_torch[device] = scratch
        self._tile_feature_scratch_torch_capacity[device] = batch_size
        return scratch

    def _build_lockstep_schedule(self, heights: np.ndarray, widths: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        schedule: List[Tuple[int, int, np.ndarray]] = []
        max_h = int(np.max(heights)) if heights.size > 0 else 0
        max_w = int(np.max(widths)) if widths.size > 0 else 0
        for t in range(max_h):
            active_rows = np.flatnonzero(heights > t)
            if active_rows.size == 0:
                continue
            widths_active = widths[active_rows]
            for s in range(max_w):
                active_indices = active_rows[widths_active > s]
                if active_indices.size > 0:
                    schedule.append((t, s, active_indices))
        return schedule

    def _build_diagonal_lockstep_schedule(self, heights: np.ndarray, widths: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        schedule: List[Tuple[int, int, np.ndarray]] = []
        max_h = int(np.max(heights)) if heights.size > 0 else 0
        max_w = int(np.max(widths)) if widths.size > 0 else 0
        for diag in range(max_h + max_w - 1):
            t_min = max(0, diag - max_w + 1)
            t_max = min(max_h - 1, diag)
            for t in range(t_min, t_max + 1):
                s = diag - t
                active_indices = np.flatnonzero((heights > t) & (widths > s))
                if active_indices.size > 0:
                    schedule.append((t, s, active_indices))
        return schedule

    def _tile_encode_schedule_name(self) -> str:
        if self.feature_mode == "diagonal_causal_edge":
            return "tile_diagonal_lockstep_batch"
        return "tile_lockstep_batch"

    def _build_schedule_by_name(
        self,
        heights: np.ndarray,
        widths: np.ndarray,
        schedule_name: str,
    ) -> List[Tuple[int, int, np.ndarray]]:
        if schedule_name == "tile_lockstep_batch":
            return self._build_lockstep_schedule(heights, widths)
        if schedule_name == "tile_diagonal_lockstep_batch":
            return self._build_diagonal_lockstep_schedule(heights, widths)
        raise ValueError(f"Unsupported tile schedule: {schedule_name}")

    def _build_tile_feature_batch_fast(
        self,
        planes: np.ndarray,
        heights_all: np.ndarray,
        widths_all: np.ndarray,
        active_indices: np.ndarray,
        coord: Tuple[int, int, int],
    ) -> torch.Tensor:
        _, t, s = coord
        batch_size = int(active_indices.size)
        patch_h = int(self._tile_patch_plan["patch_h"])
        patch_w = int(self._tile_patch_plan["patch_w"])
        flat_len = int(self._tile_patch_plan["flat_len"])
        in_channels = int(self._tile_patch_plan["in_channels"])
        coord_plan = self._tile_coord_cache[t][s]

        scratch = self._ensure_tile_feature_scratch(batch_size)
        plane_ids = scratch["plane_ids"][:batch_size]
        plane_ids[:] = active_indices[:, None]

        tt = scratch["tt"][:batch_size]
        ss = scratch["ss"][:batch_size]
        tt[:] = coord_plan["tt"]
        ss[:] = coord_plan["ss"]

        heights = scratch["heights"][:batch_size]
        widths = scratch["widths"][:batch_size]
        heights[:, 0] = heights_all[active_indices]
        widths[:, 0] = widths_all[active_indices]

        flat = scratch["flat"][:batch_size]
        flat.fill(0.0)

        inb = scratch["inb"][:batch_size]
        inb[:] = (tt >= 0) & (tt < heights) & (ss >= 0) & (ss < widths)

        real = scratch["real"][:batch_size]
        real[:] = inb
        real &= coord_plan["causal"]

        if np.any(real):
            real_values = planes[plane_ids[real], tt[real], ss[real]].astype(np.float32)
            flat[:, 0, :][real] = real_values * (1.0 / 255.0)
            flat[:, 1, :][real] = 1.0
            flat[:, 2, :][real] = 1.0

        remaining = scratch["remaining"][:batch_size]
        remaining[:] = ~real
        if np.any(remaining):
            ct = scratch["ct"][:batch_size]
            cs = scratch["cs"][:batch_size]
            np.clip(tt, 0, heights - 1, out=ct)
            np.clip(ss, 0, widths - 1, out=cs)

            mapped_valid = scratch["mapped_valid"][:batch_size]
            mapped_valid.fill(False)
            mt = scratch["mt"][:batch_size]
            ms = scratch["ms"][:batch_size]

            if s > 0:
                same_trace_mask = scratch["same_trace_mask"][:batch_size]
                same_trace_mask[:] = remaining
                same_trace_mask &= coord_plan["same_trace"]
                mapped_valid[:] = same_trace_mask
                mt[same_trace_mask] = t
                ms[same_trace_mask] = s - 1

            if t > 0:
                prev_trace_mask = scratch["prev_trace_mask"][:batch_size]
                prev_trace_mask[:] = remaining
                prev_trace_mask &= ~mapped_valid
                if self.feature_mode == "diagonal_causal_edge":
                    prev_trace_mask &= ((np.minimum(ct, t - 1) + cs) < (t + s))
                mapped_valid |= prev_trace_mask
                mt[prev_trace_mask] = np.minimum(ct[prev_trace_mask], t - 1)
                ms[prev_trace_mask] = cs[prev_trace_mask]

            if np.any(mapped_valid):
                mapped_values = planes[plane_ids[mapped_valid], mt[mapped_valid], ms[mapped_valid]].astype(np.float32)
                flat[:, 0, :][mapped_valid] = mapped_values * (1.0 / 255.0)
                flat[:, 1, :][mapped_valid] = 1.0
                flat[:, 3, :][mapped_valid] = 1.0

        feature4 = flat.reshape((batch_size, 4, patch_h, patch_w))
        if self.target_mode == "raw":
            return torch.from_numpy(feature4)

        preds = scratch["preds"][:batch_size]
        preds.fill(0)
        if t > 0 or s > 0:
            if t <= 0:
                preds[:] = planes[active_indices, t, s - 1]
            elif s <= 0:
                preds[:] = planes[active_indices, t - 1, s]
            else:
                preds[:] = np.clip(
                    planes[active_indices, t, s - 1].astype(np.int16)
                    + planes[active_indices, t - 1, s].astype(np.int16)
                    - planes[active_indices, t - 1, s - 1].astype(np.int16),
                    0,
                    255,
                ).astype(np.uint8)

        features = scratch["features"][:batch_size]
        features.fill(0.0)
        features[:, :4] = feature4
        features[:, 4] = preds[:, None, None].astype(np.float32) * (1.0 / 255.0)

        usable = scratch["usable"][:batch_size]
        usable[:] = flat[:, 1, :] > 0.5
        if np.any(usable):
            residual_flat = scratch["residual_flat"][:batch_size]
            residual_flat.fill(0.0)
            values_u8 = np.rint(flat[:, 0, :] * 255.0).astype(np.uint8)
            residual_u8 = ((values_u8.astype(np.int16) - preds[:, None].astype(np.int16)) & 0xFF).astype(np.uint8)
            residual_flat[usable] = residual_u8[usable].astype(np.float32) * (1.0 / 255.0)
            features[:, 5] = residual_flat.reshape((batch_size, patch_h, patch_w))
        return torch.from_numpy(features[:, :in_channels])

    def _build_tile_feature_batch_fast_torch(
        self,
        planes: torch.Tensor,
        heights_all: torch.Tensor,
        widths_all: torch.Tensor,
        active_indices: torch.Tensor,
        coord: Tuple[int, int, int],
    ) -> torch.Tensor:
        _, t, s = coord
        batch_size = int(active_indices.numel())
        patch_h = int(self._tile_patch_plan["patch_h"])
        patch_w = int(self._tile_patch_plan["patch_w"])
        flat_len = int(self._tile_patch_plan["flat_len"])
        in_channels = int(self._tile_patch_plan["in_channels"])
        tile_w = int(self.tile_shape[1])
        device = planes.device

        coord_plan = self._get_tile_coord_cache_torch(str(device))[t][s]
        scratch = self._ensure_tile_feature_scratch_torch(batch_size, str(device))
        plane_subset = scratch["plane_subset"][:batch_size]
        torch.index_select(planes, 0, active_indices, out=plane_subset)
        plane_flat = plane_subset.reshape(batch_size, -1)

        heights = scratch["heights"][:batch_size]
        widths = scratch["widths"][:batch_size]
        torch.index_select(heights_all, 0, active_indices, out=heights)
        torch.index_select(widths_all, 0, active_indices, out=widths)
        heights_col = heights.unsqueeze(1)
        widths_col = widths.unsqueeze(1)

        tt = coord_plan["tt"].unsqueeze(0)
        ss = coord_plan["ss"].unsqueeze(0)
        causal = coord_plan["causal"].unsqueeze(0)
        same_trace = coord_plan["same_trace"].unsqueeze(0)
        linear_clip = coord_plan["linear_clip"].unsqueeze(0).expand(batch_size, -1)
        scale = 1.0 / 255.0

        flat = scratch["flat"][:batch_size]
        flat.zero_()
        flat0 = flat[:, 0, :]
        flat1 = flat[:, 1, :]
        flat2 = flat[:, 2, :]
        flat3 = flat[:, 3, :]

        inb = scratch["inb"][:batch_size]
        inb.copy_((tt >= 0) & (tt < heights_col) & (ss >= 0) & (ss < widths_col))
        real = scratch["real"][:batch_size]
        real.copy_(inb)
        real.logical_and_(causal)

        real_values = plane_flat.gather(1, linear_clip).to(torch.float32) * scale
        real_mask_f = real.to(torch.float32)
        flat0.add_(real_values * real_mask_f)
        flat1.copy_(real_mask_f)
        flat2.copy_(real_mask_f)

        remaining = scratch["remaining"][:batch_size]
        remaining.copy_(~real)

        same_trace_mask = scratch["same_trace_mask"][:batch_size]
        same_trace_mask.zero_()
        if s > 0:
            same_trace_mask.copy_(remaining)
            same_trace_mask.logical_and_(same_trace)
            left_vals = plane_subset[:, t, s - 1].to(torch.float32).unsqueeze(1) * scale
            flat0.add_(left_vals * same_trace_mask.to(torch.float32))

        prev_trace_mask = scratch["prev_trace_mask"][:batch_size]
        prev_trace_mask.zero_()
        if t > 0:
            prev_trace_mask.copy_(remaining)
            if s > 0:
                prev_trace_mask.logical_and_(~same_trace_mask)
            ct = scratch["ct"][:batch_size]
            cs = scratch["cs"][:batch_size]
            torch.clamp(tt.expand(batch_size, -1), min=0, out=ct)
            torch.minimum(ct, heights_col - 1, out=ct)
            torch.clamp(ss.expand(batch_size, -1), min=0, out=cs)
            torch.minimum(cs, widths_col - 1, out=cs)
            if self.feature_mode == "diagonal_causal_edge":
                prev_trace_mask.logical_and_((torch.clamp(ct, max=t - 1) + cs) < (t + s))
            prev_idx = (torch.clamp(ct, max=t - 1).to(torch.int64) * tile_w + cs.to(torch.int64))
            prev_values = plane_flat.gather(1, prev_idx).to(torch.float32) * scale
            flat0.add_(prev_values * prev_trace_mask.to(torch.float32))

        mapped_mask_f = same_trace_mask.to(torch.float32)
        if t > 0:
            mapped_mask_f = mapped_mask_f + prev_trace_mask.to(torch.float32)
        flat1.add_(mapped_mask_f)
        flat3.copy_(mapped_mask_f)

        feature4 = flat.reshape((batch_size, 4, patch_h, patch_w))
        if self.target_mode == "raw":
            return feature4

        preds = torch.zeros(batch_size, dtype=torch.uint8, device=device)
        if t > 0 or s > 0:
            if t <= 0:
                preds = plane_subset[:, t, s - 1]
            elif s <= 0:
                preds = plane_subset[:, t - 1, s]
            else:
                preds = torch.clamp(
                    plane_subset[:, t, s - 1].to(torch.int16)
                    + plane_subset[:, t - 1, s].to(torch.int16)
                    - plane_subset[:, t - 1, s - 1].to(torch.int16),
                    0,
                    255,
                ).to(torch.uint8)

        features = scratch["features"][:batch_size]
        features.zero_()
        features[:, :4] = feature4
        features[:, 4] = preds.to(torch.float32).view(batch_size, 1, 1) * (1.0 / 255.0)

        if self.target_mode == "residual":
            values_u8 = torch.round(flat[:, 0, :] * 255.0).to(torch.uint8)
            residual_u8 = ((values_u8.to(torch.int16) - preds.view(batch_size, 1).to(torch.int16)) & 0xFF).to(torch.uint8)
            residual_flat = scratch["residual_flat"][:batch_size]
            residual_flat.copy_(residual_u8.to(torch.float32) * scale)
            residual_flat.mul_(flat1)
            features[:, 5] = residual_flat.reshape((batch_size, patch_h, patch_w))
        return features

    def _encode_one_tile(self, tile_volume: np.ndarray) -> bytes:
        encoder = RangeEncoder()
        coord_batch: List[Tuple[int, int, int]] = []
        symbol_batch: List[int] = []

        def flush_batch() -> None:
            if not coord_batch:
                return
            cdfs = self._predict_cdfs_batch(tile_volume, coord_batch)
            range_start = time.perf_counter() if self.profile_timing else 0.0
            for cdf, symbol in zip(cdfs, symbol_batch):
                encoder.encode_symbol(cdf, symbol)
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start
            coord_batch.clear()
            symbol_batch.clear()

        for t in range(tile_volume.shape[1]):
            for s in range(tile_volume.shape[2]):
                coord = (0, t, s)
                coord_batch.append(coord)
                symbol_batch.append(target_symbol_for_coord(tile_volume, coord, self.target_mode))
                if len(coord_batch) >= self.inference_batch:
                    flush_batch()
        flush_batch()
        return encoder.finish()

    def _decode_one_tile(self, tile_payload: bytes, tile_shape: Tuple[int, int]) -> np.ndarray:
        h, w = (int(v) for v in tile_shape)
        tile_decoded = np.zeros((1, h, w), dtype=np.uint8)
        decoder = RangeDecoder(tile_payload)
        for t in range(h):
            for s in range(w):
                coord = (0, t, s)
                cdf = self._predict_cdf(tile_decoded, coord)
                range_start = time.perf_counter() if self.profile_timing else 0.0
                symbol = int(decoder.decode_symbol(cdf))
                if self.profile_timing:
                    self._timing["range_coder_seconds"] += time.perf_counter() - range_start
                if self.target_mode == "raw":
                    tile_decoded[0, t, s] = symbol
                else:
                    pred = predictor_for_coord(tile_decoded, coord)
                    tile_decoded[0, t, s] = reconstruct_exp_from_symbol(symbol, pred, self.target_mode)
        return tile_decoded

    def _tile_entries_from_header(self, shape: Tuple[int, int, int], header: Dict[str, Any], payload_size: int) -> List[Dict[str, int]]:
        tiles = list(iter_tiles(shape, self.tile_shape))
        if "tile_payload_sizes" in header:
            payload_sizes = [int(v) for v in header["tile_payload_sizes"]]
            if len(payload_sizes) != len(tiles):
                raise ValueError(f"Tile payload count mismatch: header={len(payload_sizes)}, expected={len(tiles)}")
            payload_offset = 0
            tile_entries = []
            for tile, tile_payload_size in zip(tiles, payload_sizes):
                tile_entries.append({**tile, "payload_offset": payload_offset, "payload_size": tile_payload_size})
                payload_offset += tile_payload_size
            if payload_offset != payload_size:
                raise ValueError(f"Payload size mismatch: header sum={payload_offset}, actual={payload_size}")
            return tile_entries

        tile_entries = header.get("tiles", [])
        if len(tile_entries) != len(tiles):
            raise ValueError(f"Legacy tile metadata count mismatch: header={len(tile_entries)}, expected={len(tiles)}")
        normalized_entries = []
        for expected_tile, tile in zip(tiles, tile_entries):
            entry = {
                "p": int(tile["p"]),
                "t0": int(tile["t0"]),
                "s0": int(tile["s0"]),
                "h": int(tile["h"]),
                "w": int(tile["w"]),
                "payload_offset": int(tile["payload_offset"]),
                "payload_size": int(tile["payload_size"]),
            }
            if (entry["p"], entry["t0"], entry["s0"], entry["h"], entry["w"]) != (
                expected_tile["p"],
                expected_tile["t0"],
                expected_tile["s0"],
                expected_tile["h"],
                expected_tile["w"],
            ):
                raise ValueError(f"Tile geometry mismatch: header={entry}, expected={expected_tile}")
            normalized_entries.append(entry)
        return normalized_entries

    def _build_tile_feature_batch(self, active_states: Sequence[Dict[str, Any]], coord: Tuple[int, int, int], volume_key: str) -> torch.Tensor:
        if self.feature_mode not in CAUSAL_EDGE_FEATURE_MODES:
            return torch.cat(
                [
                    build_single_stage4_feature(
                        tile_state[volume_key],
                        coord,
                        self.patch_shape,
                        mask=self.mask,
                        feature_mode=self.feature_mode,
                        target_mode=self.target_mode,
                    )
                    for tile_state in active_states
                ],
                dim=0,
            )

        _, t, s = coord
        batch_size = len(active_states)
        patch_h, patch_w = self.patch_shape
        in_channels = feature_mode_to_in_channels(self.feature_mode, self.target_mode)
        flat_len = patch_h * patch_w
        t_half = patch_h // 2
        s_half = patch_w // 2
        jj, kk = np.indices(self.patch_shape, dtype=np.int32)
        dt = (jj - t_half).reshape(-1)
        ds = (kk - s_half).reshape(-1)
        tt = np.broadcast_to(t + dt[None, :], (batch_size, flat_len))
        ss = np.broadcast_to(s + ds[None, :], (batch_size, flat_len))

        heights = np.asarray([int(tile_state["h"]) for tile_state in active_states], dtype=np.int32)[:, None]
        widths = np.asarray([int(tile_state["w"]) for tile_state in active_states], dtype=np.int32)[:, None]
        tile_h, tile_w = self.tile_shape
        planes = np.zeros((batch_size, tile_h, tile_w), dtype=np.uint8)
        for idx, tile_state in enumerate(active_states):
            h = int(tile_state["h"])
            w = int(tile_state["w"])
            planes[idx, :h, :w] = tile_state[volume_key][0]

        inb = (tt >= 0) & (tt < heights) & (ss >= 0) & (ss < widths)
        real = inb & self._tile_context_mask(tt, ss, t, s)

        flat = np.zeros((batch_size, 4, flat_len), dtype=np.float32)
        tile_idx, pos_idx = np.nonzero(real)
        if tile_idx.size > 0:
            flat[tile_idx, 0, pos_idx] = planes[tile_idx, tt[tile_idx, pos_idx], ss[tile_idx, pos_idx]].astype(np.float32) / 255.0
            flat[tile_idx, 1, pos_idx] = 1.0
            flat[tile_idx, 2, pos_idx] = 1.0

        remaining = ~real
        if np.any(remaining):
            ct = np.clip(tt, 0, heights - 1)
            cs = np.clip(ss, 0, widths - 1)
            mapped_valid = np.zeros((batch_size, flat_len), dtype=bool)
            mt = np.zeros((batch_size, flat_len), dtype=np.int32)
            ms = np.zeros((batch_size, flat_len), dtype=np.int32)

            if s > 0:
                same_trace = remaining & (ct == t)
                mapped_valid |= same_trace
                mt[same_trace] = t
                ms[same_trace] = s - 1

            if t > 0:
                prev_trace = remaining & (~mapped_valid)
                if self.feature_mode == "diagonal_causal_edge":
                    prev_trace &= ((np.minimum(ct, t - 1) + cs) < (t + s))
                mapped_valid |= prev_trace
                mt[prev_trace] = np.minimum(ct[prev_trace], t - 1)
                ms[prev_trace] = cs[prev_trace]

            tile_idx, pos_idx = np.nonzero(mapped_valid)
            if tile_idx.size > 0:
                flat[tile_idx, 0, pos_idx] = planes[tile_idx, mt[tile_idx, pos_idx], ms[tile_idx, pos_idx]].astype(np.float32) / 255.0
                flat[tile_idx, 1, pos_idx] = 1.0
                flat[tile_idx, 3, pos_idx] = 1.0

        feature4 = flat.reshape((batch_size, 4, patch_h, patch_w))
        if self.target_mode == "raw":
            return torch.from_numpy(feature4)

        preds = np.zeros(batch_size, dtype=np.uint8)
        if t <= 0 and s <= 0:
            pass
        elif t <= 0:
            preds = planes[:, t, s - 1].astype(np.uint8)
        elif s <= 0:
            preds = planes[:, t - 1, s].astype(np.uint8)
        else:
            preds = np.clip(
                planes[:, t, s - 1].astype(np.int16) + planes[:, t - 1, s].astype(np.int16) - planes[:, t - 1, s - 1].astype(np.int16),
                0,
                255,
            ).astype(np.uint8)

        features = np.zeros((batch_size, in_channels, patch_h, patch_w), dtype=np.float32)
        features[:, :4] = feature4
        features[:, 4] = preds[:, None, None].astype(np.float32) / 255.0
        usable = flat[:, 1] > 0.5
        if np.any(usable):
            values_u8 = np.rint(flat[:, 0] * 255.0).astype(np.uint8)
            residual_u8 = ((values_u8.astype(np.int16) - preds[:, None].astype(np.int16)) & 0xFF).astype(np.uint8)
            residual_flat = np.zeros((batch_size, flat_len), dtype=np.float32)
            residual_flat[usable] = residual_u8[usable].astype(np.float32) / 255.0
            features[:, 5] = residual_flat.reshape((batch_size, patch_h, patch_w))
        return torch.from_numpy(features)

    def _predict_cdfs_for_tile_states(self, active_states: Sequence[Dict[str, Any]], coord: Tuple[int, int, int], volume_key: str = "decoded") -> List[np.ndarray]:
        if not active_states:
            return []

        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features = self._build_tile_feature_batch(active_states, coord, volume_key)
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += len(active_states)

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            probs = torch.softmax(self.model(features.to(self.device)), dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        if probs.is_cuda:
            cdfs = list(probs_to_cdfs_torch(probs, self.total_freq).cpu().numpy())
        else:
            cdfs = list(probs_to_cdfs(probs.cpu().numpy(), self.total_freq))
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdfs

    def _predict_cdfs_for_tile_batch(
        self,
        planes: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        active_indices: np.ndarray,
        coord: Tuple[int, int, int],
    ) -> List[np.ndarray]:
        if active_indices.size == 0:
            return []

        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features = self._build_tile_feature_batch_fast(planes, heights, widths, active_indices, coord)
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += len(active_indices)

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            probs = torch.softmax(self.model(features.to(self.device)), dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        if probs.is_cuda:
            cdfs = list(probs_to_cdfs_torch(probs, self.total_freq).cpu().numpy())
        else:
            cdfs = list(probs_to_cdfs(probs.cpu().numpy(), self.total_freq))
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdfs

    def _predict_cdfs_for_tile_batch_cuda(
        self,
        planes: torch.Tensor,
        heights: torch.Tensor,
        widths: torch.Tensor,
        active_indices: torch.Tensor,
        coord: Tuple[int, int, int],
    ) -> List[np.ndarray]:
        if active_indices.numel() == 0:
            return []

        patch_start = time.perf_counter() if self.profile_timing else 0.0
        features = self._build_tile_feature_batch_fast_torch(planes, heights, widths, active_indices, coord)
        if self.profile_timing:
            self._timing["patch_build_seconds"] += time.perf_counter() - patch_start
            self._timing["voxel_count"] += int(active_indices.numel())

        infer_start = time.perf_counter() if self.profile_timing else 0.0
        with torch.inference_mode():
            probs = torch.softmax(self.model(features), dim=1)
        if self.profile_timing:
            self._timing["model_inference_seconds"] += time.perf_counter() - infer_start

        cdf_start = time.perf_counter() if self.profile_timing else 0.0
        cdfs = list(probs_to_cdfs_torch(probs, self.total_freq).cpu().numpy())
        if self.profile_timing:
            self._timing["cdf_quantization_seconds"] += time.perf_counter() - cdf_start
        return cdfs

    def encode_exponents(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        exp_volume = np.asarray(exp_volume, dtype=np.uint8)
        total_voxels = int(np.prod(exp_volume.shape))
        tile_states = []
        for tile in iter_tiles(exp_volume.shape, self.tile_shape):
            p, t0, s0, h, w = tile["p"], tile["t0"], tile["s0"], tile["h"], tile["w"]
            tile_states.append(
                {
                    **tile,
                    "source": np.ascontiguousarray(exp_volume[p : p + 1, t0 : t0 + h, s0 : s0 + w]),
                    "encoder": RangeEncoder(),
                }
            )
        planes, heights, widths = self._prepare_tile_planes(tile_states, "source")
        encode_schedule = self._tile_encode_schedule_name()
        schedule = self._build_schedule_by_name(heights, widths, encode_schedule)
        use_cuda_fastpath = self._use_cuda_tile_fastpath()
        if use_cuda_fastpath:
            planes_t, heights_t, widths_t = self._prepare_tile_tensors(planes, heights, widths)
            schedule_t = self._prepare_schedule_tensors(schedule)

        self._reset_timing()
        total_start = time.perf_counter()
        schedule_iter = schedule_t if use_cuda_fastpath else schedule
        for t, s, active_indices in schedule_iter:
            coord = (0, t, s)
            if use_cuda_fastpath:
                cdfs = self._predict_cdfs_for_tile_batch_cuda(planes_t, heights_t, widths_t, active_indices, coord)
                active_index_list = active_indices.tolist()
            elif self.feature_mode in CAUSAL_EDGE_FEATURE_MODES:
                cdfs = self._predict_cdfs_for_tile_batch(planes, heights, widths, active_indices, coord)
                active_index_list = active_indices.tolist()
            else:
                active_states = [tile_states[idx] for idx in active_indices.tolist()]
                cdfs = self._predict_cdfs_for_tile_states(active_states, coord, volume_key="source")
                active_index_list = active_indices.tolist()
            range_start = time.perf_counter() if self.profile_timing else 0.0
            for plane_idx, cdf in zip(active_index_list, cdfs):
                tile_state = tile_states[plane_idx]
                symbol = target_symbol_for_coord(tile_state["source"], coord, self.target_mode)
                tile_state["encoder"].encode_symbol(cdf, symbol)
            if self.profile_timing:
                self._timing["range_coder_seconds"] += time.perf_counter() - range_start

        tile_payload_sizes = []
        payload_chunks = []
        for tile_state in tile_states:
            tile_payload = tile_state["encoder"].finish()
            payload_chunks.append(tile_payload)
            tile_payload_sizes.append(len(tile_payload))

        payload = b"".join(payload_chunks)
        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = {
            "codec": "stage4_causal_range_tile64",
            "shape": list(exp_volume.shape),
            "patch_shape": list(self.patch_shape),
            "tile_shape": list(self.tile_shape),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "predictor": "loco_i_2d_tile" if self.target_mode == "residual" else None,
            "total_freq": self.total_freq,
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "voxel_count": total_voxels,
            "tile_count": len(tile_payload_sizes),
            "encode_schedule": encode_schedule,
            "tile_payload_sizes": tile_payload_sizes,
        }
        write_bitstream(bitstream_path, header, payload)
        header_size = len(_header_bytes(header)) + 12
        total_bytes = len(payload) + header_size
        return {
            "bitstream_path": os.path.abspath(bitstream_path),
            "payload_bytes": len(payload),
            "header_bytes": header_size,
            "total_bytes": total_bytes,
            "bits_per_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "feature_mode": self.feature_mode,
            "target_mode": self.target_mode,
            "codec_layout": "tile64",
            "tile_shape": list(self.tile_shape),
            "tile_count": len(tile_payload_sizes),
            "timing": self._timing_report(),
        }

    def decode_exponents(self, bitstream_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        header, payload = read_bitstream(bitstream_path)
        if header.get("codec") != "stage4_causal_range_tile64":
            raise ValueError(f"Unsupported tile bitstream codec: {header.get('codec')}")
        shape = tuple(int(v) for v in header["shape"])
        if tuple(header["patch_shape"]) != tuple(self.patch_shape):
            raise ValueError("Patch shape mismatch between bitstream and checkpoint.")
        if tuple(header.get("tile_shape", self.tile_shape)) != tuple(self.tile_shape):
            raise ValueError("Tile shape mismatch between bitstream and codec.")
        if int(header["total_freq"]) != self.total_freq:
            raise ValueError("Range total mismatch between bitstream and config.")
        header_feature_mode = header.get("feature_mode", "strict")
        if header_feature_mode != self.feature_mode:
            raise ValueError(f"Feature mode mismatch: bitstream={header_feature_mode}, codec={self.feature_mode}")
        header_target_mode = header.get("target_mode", "raw")
        if header_target_mode != self.target_mode:
            raise ValueError(f"Target mode mismatch: bitstream={header_target_mode}, codec={self.target_mode}")

        decoded = np.zeros(shape, dtype=np.uint8)
        tile_entries = self._tile_entries_from_header(shape, header, len(payload))
        tile_states = []
        tile_h, tile_w = self.tile_shape
        decoded_planes = np.zeros((len(tile_entries), tile_h, tile_w), dtype=np.uint8)
        for tile in tile_entries:
            p = int(tile["p"])
            t0 = int(tile["t0"])
            s0 = int(tile["s0"])
            h = int(tile["h"])
            w = int(tile["w"])
            payload_offset = int(tile["payload_offset"])
            payload_size = int(tile["payload_size"])
            tile_payload = payload[payload_offset : payload_offset + payload_size]
            if len(tile_payload) != payload_size:
                raise ValueError(f"Truncated tile payload at p={p}, t0={t0}, s0={s0}")
            tile_states.append(
                {
                    "p": p,
                    "t0": t0,
                    "s0": s0,
                    "h": h,
                    "w": w,
                    "decoded": decoded_planes[len(tile_states) : len(tile_states) + 1, :h, :w],
                    "decoder": RangeDecoder(tile_payload),
                }
            )
        heights = np.asarray([int(tile_state["h"]) for tile_state in tile_states], dtype=np.int32)
        widths = np.asarray([int(tile_state["w"]) for tile_state in tile_states], dtype=np.int32)
        for idx, tile_state in enumerate(tile_states):
            tile_state["plane_index"] = idx
        planes = decoded_planes
        decode_schedule = header.get("encode_schedule", "tile_serial")
        schedule = []
        if decode_schedule in {"tile_lockstep_batch", "tile_diagonal_lockstep_batch"}:
            schedule = self._build_schedule_by_name(heights, widths, decode_schedule)
        use_cuda_fastpath = self._use_cuda_tile_fastpath()
        if use_cuda_fastpath:
            planes_t, heights_t, widths_t = self._prepare_tile_tensors(planes, heights, widths)
            schedule_t = self._prepare_schedule_tensors(schedule)

        self._reset_timing()
        total_start = time.perf_counter()
        if decode_schedule in {"tile_lockstep_batch", "tile_diagonal_lockstep_batch"}:
            schedule_iter = schedule_t if use_cuda_fastpath else schedule
            for t, s, active_indices in schedule_iter:
                coord = (0, t, s)
                if use_cuda_fastpath:
                    cdfs = self._predict_cdfs_for_tile_batch_cuda(planes_t, heights_t, widths_t, active_indices, coord)
                    active_index_list = active_indices.tolist()
                elif self.feature_mode in CAUSAL_EDGE_FEATURE_MODES:
                    cdfs = self._predict_cdfs_for_tile_batch(planes, heights, widths, active_indices, coord)
                    active_index_list = active_indices.tolist()
                else:
                    active_states = [tile_states[idx] for idx in active_indices.tolist()]
                    cdfs = self._predict_cdfs_for_tile_states(active_states, coord)
                    active_index_list = active_indices.tolist()
                range_start = time.perf_counter() if self.profile_timing else 0.0
                decoded_values: List[int] = []
                for plane_idx, cdf in zip(active_index_list, cdfs):
                    tile_state = tile_states[plane_idx]
                    symbol = int(tile_state["decoder"].decode_symbol(cdf))
                    if self.target_mode == "raw":
                        decoded_value = symbol
                    else:
                        pred = predictor_for_coord(tile_state["decoded"], coord)
                        decoded_value = reconstruct_exp_from_symbol(symbol, pred, self.target_mode)
                    tile_state["decoded"][0, t, s] = decoded_value
                    decoded_values.append(int(decoded_value))
                if use_cuda_fastpath and decoded_values:
                    planes_t[active_indices, t, s] = torch.tensor(decoded_values, device=self.device, dtype=torch.uint8)
                if self.profile_timing:
                    self._timing["range_coder_seconds"] += time.perf_counter() - range_start
        elif decode_schedule == "tile_serial":
            for tile_state in tile_states:
                tile_state["decoded"] = self._decode_one_tile(tile_state["decoder"].input.data, (tile_state["h"], tile_state["w"]))
        else:
            raise ValueError(f"Unsupported tile decode schedule: {decode_schedule}")

        for tile_state in tile_states:
            p = int(tile_state["p"])
            t0 = int(tile_state["t0"])
            s0 = int(tile_state["s0"])
            h = int(tile_state["h"])
            w = int(tile_state["w"])
            decoded[p : p + 1, t0 : t0 + h, s0 : s0 + w] = tile_state["decoded"]

        self._timing["total_wall_seconds"] = time.perf_counter() - total_start
        header = dict(header)
        header["timing"] = self._timing_report()
        header["decode_schedule"] = decode_schedule
        return decoded, header
