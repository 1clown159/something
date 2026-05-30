#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


RectangleArray = np.ndarray


def region_enabled(mode: str) -> bool:
    return str(mode).lower() != "none"


def full_profile_rectangles(shape: Tuple[int, int, int]) -> RectangleArray:
    n_profiles, n_traces, _ = (int(v) for v in shape)
    rectangles = np.zeros((n_profiles, 2), dtype=np.int64)
    rectangles[:, 1] = n_traces
    return rectangles


def _trace_threshold_count(n_samples: int, min_nonzero_ratio: float) -> int:
    ratio = max(0.0, float(min_nonzero_ratio))
    if ratio <= 0.0:
        return 1
    return max(1, int(np.ceil(ratio * int(n_samples))))


def detect_profile_rectangles(
    volume: np.ndarray,
    min_nonzero_ratio: float = 0.0,
    margin_traces: int = 0,
    group_size: int = 1,
) -> RectangleArray:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    n_profiles, n_traces, n_samples = (int(v) for v in volume.shape)
    threshold = _trace_threshold_count(n_samples, min_nonzero_ratio)
    margin = max(0, int(margin_traces))
    rectangles = np.zeros((n_profiles, 2), dtype=np.int64)
    rectangles[:, 1] = n_traces

    for p in range(n_profiles):
        trace_nonzero = np.count_nonzero(volume[p] != 0, axis=1)
        valid = trace_nonzero >= threshold
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            left = 0
            right = n_traces
        else:
            left = max(0, int(valid_idx[0]) - margin)
            right = min(n_traces, int(valid_idx[-1]) + 1 + margin)
        rectangles[p, 0] = left
        rectangles[p, 1] = right

    group = max(1, int(group_size))
    if group > 1 and n_profiles > 0:
        grouped = rectangles.copy()
        for start in range(0, n_profiles, group):
            stop = min(n_profiles, start + group)
            grouped[start:stop, 0] = int(np.min(rectangles[start:stop, 0]))
            grouped[start:stop, 1] = int(np.max(rectangles[start:stop, 1]))
        rectangles = grouped

    return rectangles


def voxel_count_in_rectangles(rectangles: RectangleArray, n_samples: int) -> int:
    rects = np.asarray(rectangles, dtype=np.int64)
    widths = np.maximum(0, rects[:, 1] - rects[:, 0])
    return int(np.sum(widths, dtype=np.int64) * int(n_samples))


def iter_rectangle_coords(shape: Tuple[int, int, int], rectangles: Optional[RectangleArray]) -> Iterator[Tuple[int, int, int]]:
    n_profiles, n_traces, n_samples = (int(v) for v in shape)
    rects = full_profile_rectangles(shape) if rectangles is None else np.asarray(rectangles, dtype=np.int64)
    if rects.shape != (n_profiles, 2):
        raise ValueError(f"Expected rectangles shape {(n_profiles, 2)}, got {rects.shape}")

    for p in range(n_profiles):
        left = max(0, min(n_traces, int(rects[p, 0])))
        right = max(left, min(n_traces, int(rects[p, 1])))
        for t in range(left, right):
            for s in range(n_samples):
                yield (p, t, s)


def choose_random_indices_in_rectangles(
    shape: Tuple[int, int, int],
    rectangles: RectangleArray,
    limit: Optional[int],
    seed: int,
) -> np.ndarray:
    n_profiles, _, n_samples = (int(v) for v in shape)
    rects = np.asarray(rectangles, dtype=np.int64)
    widths = np.maximum(0, rects[:, 1] - rects[:, 0]).astype(np.int64)
    per_profile = widths * int(n_samples)
    total = int(np.sum(per_profile, dtype=np.int64))
    if total <= 0:
        return np.empty((0, 3), dtype=np.int64)

    rng = np.random.default_rng(seed)
    if limit is None or int(limit) >= total:
        linear = np.arange(total, dtype=np.int64)
    else:
        linear = np.sort(rng.choice(total, size=int(limit), replace=False).astype(np.int64))

    offsets = np.concatenate([[0], np.cumsum(per_profile, dtype=np.int64)])
    profile_idx = np.searchsorted(offsets[1:], linear, side="right")
    local = linear - offsets[profile_idx]
    trace_offset = local // int(n_samples)
    sample_idx = local % int(n_samples)
    trace_idx = rects[profile_idx, 0] + trace_offset
    return np.stack([profile_idx, trace_idx, sample_idx], axis=1).astype(np.int64)


def make_regular_grid_indices_in_rectangles(
    shape: Tuple[int, int, int],
    rectangles: RectangleArray,
    trace_stride: int,
    sample_stride: int,
) -> np.ndarray:
    n_profiles, n_traces, n_samples = (int(v) for v in shape)
    rects = np.asarray(rectangles, dtype=np.int64)
    coords: List[Tuple[int, int, int]] = []
    trace_step = max(1, int(trace_stride))
    sample_step = max(1, int(sample_stride))
    for p in range(n_profiles):
        left = max(0, min(n_traces, int(rects[p, 0])))
        right = max(left, min(n_traces, int(rects[p, 1])))
        for t in range(left, right, trace_step):
            for s in range(0, n_samples, sample_step):
                coords.append((p, t, s))
    return np.asarray(coords, dtype=np.int64)


def rectangles_to_metadata(
    rectangles: Optional[RectangleArray],
    shape: Tuple[int, int, int],
    mode: str,
    min_nonzero_ratio: float,
    margin_traces: int,
    group_size: int,
) -> Optional[Dict[str, Any]]:
    if rectangles is None:
        return None
    n_profiles, n_traces, n_samples = (int(v) for v in shape)
    rects = np.asarray(rectangles, dtype=np.int64)
    widths = np.maximum(0, rects[:, 1] - rects[:, 0]).astype(np.int64)
    return {
        "mode": mode,
        "min_nonzero_ratio": float(min_nonzero_ratio),
        "margin_traces": int(margin_traces),
        "group_size": int(group_size),
        "rectangles": rects.tolist(),
        "profile_count": n_profiles,
        "trace_count": n_traces,
        "sample_count": n_samples,
        "voxel_count": voxel_count_in_rectangles(rects, n_samples),
        "trace_width_min": int(widths.min()) if widths.size else 0,
        "trace_width_max": int(widths.max()) if widths.size else 0,
        "trace_width_mean": float(widths.mean()) if widths.size else 0.0,
    }
