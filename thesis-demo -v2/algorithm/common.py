#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEFAULT_BIN_PATH = r"E:\code\thesis\20260122\profiles_combined.bin"
DEFAULT_OUTPUT_DIR = r"E:\code\thesis\20260416\outputs"
DEFAULT_META_PATH = r"E:\code\thesis\20260416\profiles_combined_meta.json"


@dataclass(frozen=True)
class VolumeShape:
    n_profiles: int = 10
    traces_per_profile: int = 600
    samples_per_trace: int = 2001

    @property
    def total_samples(self) -> int:
        return self.n_profiles * self.traces_per_profile * self.samples_per_trace

    @property
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.n_profiles, self.traces_per_profile, self.samples_per_trace)


@dataclass(frozen=True)
class SplitConfig:
    profile_offset: int = 0
    train_profiles: int = 8
    val_profiles: int = 1
    test_profiles: int = 1

    @property
    def total_profiles(self) -> int:
        return self.train_profiles + self.val_profiles + self.test_profiles

    def profile_slices(self) -> Dict[str, slice]:
        start = self.profile_offset
        train_end = start + self.train_profiles
        val_end = train_end + self.val_profiles
        test_end = val_end + self.test_profiles
        return {
            "train": slice(start, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, test_end),
        }

    def validate(self, n_profiles: int) -> None:
        if self.profile_offset < 0:
            raise ValueError("profile_offset must be >= 0.")
        if self.train_profiles < 0 or self.val_profiles < 0 or self.test_profiles < 0:
            raise ValueError("Split profile counts must be >= 0.")
        if self.total_profiles <= 0:
            raise ValueError("At least one profile must be assigned to train/val/test.")
        if self.profile_offset + self.total_profiles > n_profiles:
            raise ValueError(
                f"SplitConfig exceeds available profiles: offset={self.profile_offset}, total={self.total_profiles}, available={n_profiles}"
            )


@dataclass
class ExperimentConfig:
    seed: int = 20260401
    patch_shape: Tuple[int, int] = (9, 17)
    tile_shape: Tuple[int, int] = (64, 64)
    stage4_train_samples: int = 32000
    stage4_val_samples: int = 8000
    stage4_eval_samples: int = 32000
    stage4_base_channels: int = 16
    batch_size: int = 128
    epochs_stage4: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    eval_trace_stride: int = 4
    eval_sample_stride: int = 8
    output_dir: str = DEFAULT_OUTPUT_DIR
    range_total: int = 1 << 15
    codec_device: str = "cpu"
    feature_mode: str = "diagonal_causal_edge"
    target_mode: str = "raw"
    valid_region_mode: str = "none"
    valid_region_min_nonzero_ratio: float = 0.0
    valid_region_margin_traces: int = 0
    valid_region_group_size: int = 1


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def environment_report() -> Dict[str, Any]:
    report = {
        "torch_available": torch is not None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
    }
    if torch is not None:
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
    return report


def infer_shape_from_file(bin_path: str, fallback: VolumeShape) -> VolumeShape:
    n_float32 = os.path.getsize(bin_path) // 4
    if n_float32 == fallback.total_samples:
        return fallback
    trace_span = fallback.traces_per_profile * fallback.samples_per_trace
    if n_float32 % trace_span != 0:
        raise ValueError(f"Cannot infer volume shape from {bin_path}.")
    return VolumeShape(
        n_profiles=n_float32 // trace_span,
        traces_per_profile=fallback.traces_per_profile,
        samples_per_trace=fallback.samples_per_trace,
    )


def write_sidecar_meta(path: str, bin_path: str, shape: VolumeShape) -> None:
    save_json(
        path,
        {
            "source_bin_path": os.path.abspath(bin_path),
            "shape": list(shape.as_tuple),
            "dtype": "float32",
        },
    )


def extract_float_components(data_float32: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_u32 = np.asarray(data_float32, dtype=np.float32).view(np.uint32)
    signs = ((data_u32 >> 31) & 0x1).astype(np.uint8)
    exps = ((data_u32 >> 23) & 0xFF).astype(np.uint8)
    mants = (data_u32 & 0x7FFFFF).astype(np.uint32)
    return signs, exps, mants


def extract_float_exponents(data_float32: np.ndarray) -> np.ndarray:
    data_u32 = np.asarray(data_float32, dtype=np.float32).view(np.uint32)
    return ((data_u32 >> 23) & 0xFF).astype(np.uint8)


@dataclass
class VolumeData:
    bin_path: str
    shape: VolumeShape
    split_config: SplitConfig = field(default_factory=SplitConfig)
    float_memmap: np.memmap = field(init=False, repr=False)
    exps: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.float_memmap = np.memmap(self.bin_path, dtype=np.float32, mode="r")
        self.exps = extract_float_exponents(self.float_memmap).reshape(self.shape.as_tuple)
        self.split_config.validate(self.shape.n_profiles)

    @property
    def profile_slices(self) -> Dict[str, slice]:
        return self.split_config.profile_slices()

    def get_split(self, name: str) -> np.ndarray:
        return self.exps[self.profile_slices[name]]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "bin_path": os.path.abspath(self.bin_path),
            "shape": list(self.shape.as_tuple),
            "split_slices": {k: [v.start, v.stop] for k, v in self.profile_slices.items()},
            "split_config": {
                "profile_offset": self.split_config.profile_offset,
                "train_profiles": self.split_config.train_profiles,
                "val_profiles": self.split_config.val_profiles,
                "test_profiles": self.split_config.test_profiles,
            },
        }


def normalize_uint8_patch(patch: np.ndarray) -> np.ndarray:
    return patch.astype(np.float32) / 255.0


def lexicographic_causal_mask_2d(patch_shape: Tuple[int, int]) -> np.ndarray:
    t_half, s_half = (dim // 2 for dim in patch_shape)
    mask = np.zeros(patch_shape, dtype=np.float32)
    for j in range(patch_shape[0]):
        for k in range(patch_shape[1]):
            dt = j - t_half
            ds = k - s_half
            if dt < 0 or (dt == 0 and ds < 0):
                mask[j, k] = 1.0
    return mask


def choose_random_indices(shape: Tuple[int, int, int], limit: Optional[int], seed: int) -> np.ndarray:
    n_profiles, n_traces, n_samples = shape
    total = n_profiles * n_traces * n_samples
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if limit is None or limit >= total:
        linear = np.arange(total, dtype=np.int64)
    else:
        linear = rng.choice(total, size=limit, replace=False)
    profile_span = n_traces * n_samples
    p = linear // profile_span
    rem = linear % profile_span
    t = rem // n_samples
    s = rem % n_samples
    return np.stack([p, t, s], axis=1).astype(np.int64)


def make_regular_grid_indices(shape: Tuple[int, int, int], trace_stride: int, sample_stride: int) -> np.ndarray:
    n_profiles, n_traces, n_samples = shape
    coords: List[Tuple[int, int, int]] = []
    for p in range(n_profiles):
        for t in range(0, n_traces, max(1, trace_stride)):
            for s in range(0, n_samples, max(1, sample_stride)):
                coords.append((p, t, s))
    return np.asarray(coords, dtype=np.int64)


def stage_output_dir(root: str, stage_name: str, mode: str) -> str:
    return ensure_dir(os.path.join(root, stage_name, mode))


def checkpoint_path(root: str, stage_name: str, mode: str) -> str:
    return os.path.join(stage_output_dir(root, stage_name, mode), "checkpoint.pt")


def metrics_path(root: str, stage_name: str, mode: str) -> str:
    return os.path.join(stage_output_dir(root, stage_name, mode), "metrics.json")


def to_serializable_config(config: ExperimentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["patch_shape"] = list(config.patch_shape)
    return payload


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
