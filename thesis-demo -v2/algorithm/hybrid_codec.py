#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import zstandard as zstd

try:
    from .codec import Stage4RangeCodec
    from .common import ExperimentConfig
    from .roi import (
        detect_profile_rectangles,
        rectangles_to_metadata,
        region_enabled,
        voxel_count_in_rectangles,
    )
except ImportError:
    from codec import Stage4RangeCodec
    from common import ExperimentConfig
    from roi import (
        detect_profile_rectangles,
        rectangles_to_metadata,
        region_enabled,
        voxel_count_in_rectangles,
    )


HYBRID_MAGIC = b"S4HZ"
HYBRID_VERSION = 1


def _hybrid_header_bytes(header: Dict[str, Any]) -> bytes:
    return json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8")


def write_hybrid_bitstream(path: str, header: Dict[str, Any], payload: bytes) -> None:
    header_blob = _hybrid_header_bytes(header)
    with open(path, "wb") as handle:
        handle.write(HYBRID_MAGIC)
        handle.write(struct.pack("<I", HYBRID_VERSION))
        handle.write(struct.pack("<I", len(header_blob)))
        handle.write(header_blob)
        handle.write(payload)


def read_hybrid_bitstream(path: str) -> Tuple[Dict[str, Any], bytes]:
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != HYBRID_MAGIC:
            raise ValueError("Invalid hybrid bitstream magic.")
        version = struct.unpack("<I", handle.read(4))[0]
        if version != HYBRID_VERSION:
            raise ValueError(f"Unsupported hybrid bitstream version: {version}")
        header_len = struct.unpack("<I", handle.read(4))[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
        payload = handle.read()
    return header, payload


def _temp_bitstream_path(base_dir: str, suffix: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="tmp_roi_", suffix=suffix, dir=base_dir)
    os.close(fd)
    return path


def _rectangles_from_header(header: Dict[str, Any], shape: Tuple[int, int, int]) -> np.ndarray:
    region = header.get("valid_region")
    if not region:
        raise ValueError("Hybrid bitstream is missing valid_region metadata.")
    rectangles = np.asarray(region.get("rectangles", []), dtype=np.int64)
    expected_shape = (int(shape[0]), 2)
    if rectangles.shape != expected_shape:
        raise ValueError(f"Invalid rectangles shape in hybrid header: expected {expected_shape}, got {rectangles.shape}")
    return rectangles


def extract_non_roi_bytes(exp_volume: np.ndarray, rectangles: np.ndarray) -> bytes:
    exp_volume = np.asarray(exp_volume, dtype=np.uint8)
    n_profiles, n_traces, _ = exp_volume.shape
    rects = np.asarray(rectangles, dtype=np.int64)
    parts: List[bytes] = []
    for p in range(n_profiles):
        left = max(0, min(n_traces, int(rects[p, 0])))
        right = max(left, min(n_traces, int(rects[p, 1])))
        if left > 0:
            parts.append(np.ascontiguousarray(exp_volume[p, :left, :]).tobytes())
        if right < n_traces:
            parts.append(np.ascontiguousarray(exp_volume[p, right:, :]).tobytes())
    return b"".join(parts)


def restore_non_roi_bytes(decoded: np.ndarray, raw_bytes: bytes, rectangles: np.ndarray) -> None:
    decoded = np.asarray(decoded, dtype=np.uint8)
    n_profiles, n_traces, n_samples = decoded.shape
    rects = np.asarray(rectangles, dtype=np.int64)
    view = memoryview(raw_bytes)
    offset = 0
    for p in range(n_profiles):
        left = max(0, min(n_traces, int(rects[p, 0])))
        right = max(left, min(n_traces, int(rects[p, 1])))
        left_count = left * n_samples
        if left_count > 0:
            chunk = np.frombuffer(view[offset : offset + left_count], dtype=np.uint8).reshape(left, n_samples)
            decoded[p, :left, :] = chunk
            offset += left_count
        right_width = n_traces - right
        right_count = right_width * n_samples
        if right_count > 0:
            chunk = np.frombuffer(view[offset : offset + right_count], dtype=np.uint8).reshape(right_width, n_samples)
            decoded[p, right:, :] = chunk
            offset += right_count
    if offset != len(raw_bytes):
        raise ValueError(f"Non-ROI byte count mismatch: consumed={offset}, actual={len(raw_bytes)}")


class Stage4HybridROIzstdCodec:
    def __init__(
        self,
        checkpoint_path: str,
        config: ExperimentConfig,
        device: str = "cpu",
        feature_mode: str = "auto",
        target_mode: str = "auto",
        profile_timing: bool = False,
        inference_batch: int = 1,
        zstd_level: int = 9,
        zstd_threads: int = -1,
    ) -> None:
        if not region_enabled(config.valid_region_mode):
            raise ValueError("Hybrid ROI+zstd codec requires --valid-region-mode auto_rect.")
        self.config = config
        self.zstd_level = int(zstd_level)
        self.zstd_threads = int(zstd_threads)
        self.roi_codec = Stage4RangeCodec(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            feature_mode=feature_mode,
            target_mode=target_mode,
            profile_timing=profile_timing,
            inference_batch=inference_batch,
        )

    def _resolve_rectangles(self, exp_volume: np.ndarray) -> np.ndarray:
        return detect_profile_rectangles(
            exp_volume,
            min_nonzero_ratio=self.config.valid_region_min_nonzero_ratio,
            margin_traces=self.config.valid_region_margin_traces,
            group_size=self.config.valid_region_group_size,
        )

    def encode_exponents(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        exp_volume = np.asarray(exp_volume, dtype=np.uint8)
        total_voxels = int(np.prod(exp_volume.shape))
        rectangles = self._resolve_rectangles(exp_volume)
        roi_meta = rectangles_to_metadata(
            rectangles,
            exp_volume.shape,
            self.config.valid_region_mode,
            self.config.valid_region_min_nonzero_ratio,
            self.config.valid_region_margin_traces,
            self.config.valid_region_group_size,
        )
        modeled_voxels = voxel_count_in_rectangles(rectangles, exp_volume.shape[2])
        non_roi_voxels = total_voxels - modeled_voxels

        temp_dir = os.path.dirname(os.path.abspath(bitstream_path)) or os.getcwd()
        roi_tmp_path = _temp_bitstream_path(temp_dir, ".s4rc")
        try:
            roi_metrics = self.roi_codec.encode_exponents(exp_volume, roi_tmp_path)
            with open(roi_tmp_path, "rb") as handle:
                roi_blob = handle.read()
        finally:
            if os.path.exists(roi_tmp_path):
                os.remove(roi_tmp_path)

        non_roi_raw = extract_non_roi_bytes(exp_volume, rectangles)
        compressor = zstd.ZstdCompressor(level=self.zstd_level, threads=self.zstd_threads)
        non_roi_zstd = compressor.compress(non_roi_raw)

        header = {
            "codec": "stage4_roi_zstd_hybrid",
            "shape": list(exp_volume.shape),
            "feature_mode": self.roi_codec.feature_mode,
            "target_mode": self.roi_codec.target_mode,
            "zstd_level": self.zstd_level,
            "zstd_threads": self.zstd_threads,
            "roi_bitstream_bytes": len(roi_blob),
            "non_roi_raw_bytes": len(non_roi_raw),
            "non_roi_zstd_bytes": len(non_roi_zstd),
            "modeled_voxels": modeled_voxels,
            "non_roi_voxels": non_roi_voxels,
            "total_voxels": total_voxels,
            "valid_region": roi_meta,
        }
        payload = roi_blob + non_roi_zstd
        write_hybrid_bitstream(bitstream_path, header, payload)

        header_size = len(_hybrid_header_bytes(header)) + 12
        total_bytes = len(payload) + header_size
        return {
            "bitstream_path": os.path.abspath(bitstream_path),
            "codec": header["codec"],
            "roi_metrics": roi_metrics,
            "roi_bitstream_bytes": len(roi_blob),
            "non_roi_raw_bytes": len(non_roi_raw),
            "non_roi_zstd_bytes": len(non_roi_zstd),
            "header_bytes": header_size,
            "total_bytes": total_bytes,
            "bits_per_total_voxel": 8.0 * total_bytes / max(total_voxels, 1),
            "bits_per_modeled_voxel": 8.0 * total_bytes / max(modeled_voxels, 1),
            "zstd_level": self.zstd_level,
            "zstd_threads": self.zstd_threads,
            "valid_region": roi_meta,
        }

    def decode_exponents(self, bitstream_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        header, payload = read_hybrid_bitstream(bitstream_path)
        if header.get("codec") != "stage4_roi_zstd_hybrid":
            raise ValueError(f"Unsupported hybrid codec: {header.get('codec')}")

        shape = tuple(int(v) for v in header["shape"])
        rectangles = _rectangles_from_header(header, shape)
        roi_len = int(header["roi_bitstream_bytes"])
        roi_blob = payload[:roi_len]
        non_roi_blob = payload[roi_len:]

        temp_dir = os.path.dirname(os.path.abspath(bitstream_path)) or os.getcwd()
        roi_tmp_path = _temp_bitstream_path(temp_dir, ".s4rc")
        try:
            with open(roi_tmp_path, "wb") as handle:
                handle.write(roi_blob)
            decoded, roi_header = self.roi_codec.decode_exponents(roi_tmp_path)
        finally:
            if os.path.exists(roi_tmp_path):
                os.remove(roi_tmp_path)

        expected_non_roi = int(header["non_roi_raw_bytes"])
        if expected_non_roi > 0:
            raw_non_roi = zstd.ZstdDecompressor().decompress(non_roi_blob, max_output_size=expected_non_roi)
            if len(raw_non_roi) != expected_non_roi:
                raise ValueError(f"Unexpected non-ROI decode length: got {len(raw_non_roi)}, expected {expected_non_roi}")
            restore_non_roi_bytes(decoded, raw_non_roi, rectangles)

        out_header = dict(header)
        out_header["roi_header"] = roi_header
        return decoded, out_header

    def roundtrip(self, exp_volume: np.ndarray, bitstream_path: str) -> Dict[str, Any]:
        encode_metrics = self.encode_exponents(exp_volume, bitstream_path)
        decoded, header = self.decode_exponents(bitstream_path)
        ok = np.array_equal(np.asarray(exp_volume, dtype=np.uint8), decoded)
        return {
            "ok": bool(ok),
            "encode": encode_metrics,
            "header": header,
        }
