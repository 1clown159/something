#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError:
    print("[Error] Library 'segyio' is required. Run: pip install segyio")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer likely 2D grid dimensions from SGY SourceX/SourceY coordinates."
    )
    parser.add_argument("--input", required=True, help="Path to a single .sgy or .segy file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of best candidate grid shapes to print.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=int,
        default=1000,
        help="Absolute delta threshold used to detect row breaks in sequential coordinates.",
    )
    return parser.parse_args()


def factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs


def score_candidate(sx: np.ndarray, sy: np.ndarray, width: int) -> dict[str, object]:
    grid_x = sx.reshape(-1, width)
    grid_y = sy.reshape(-1, width)

    row_dx = np.diff(grid_x, axis=1)
    row_dy = np.diff(grid_y, axis=1)
    col_dx = np.diff(grid_x, axis=0)
    col_dy = np.diff(grid_y, axis=0)

    row_step = (int(np.median(row_dx)), int(np.median(row_dy))) if row_dx.size else (0, 0)
    col_step = (int(np.median(col_dx)), int(np.median(col_dy))) if col_dx.size else (0, 0)

    row_err = 0.0
    if row_dx.size:
        row_err = float(np.mean(np.abs(row_dx - row_step[0])) + np.mean(np.abs(row_dy - row_step[1])))
    col_err = 0.0
    if col_dx.size:
        col_err = float(np.mean(np.abs(col_dx - col_step[0])) + np.mean(np.abs(col_dy - col_step[1])))

    total_err = row_err + col_err
    return {
        "shape": (int(grid_x.shape[0]), int(grid_x.shape[1])),
        "width": int(width),
        "height": int(grid_x.shape[0]),
        "row_step": row_step,
        "col_step": col_step,
        "row_err": row_err,
        "col_err": col_err,
        "total_err": total_err,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return 1
    if input_path.suffix.lower() not in {".sgy", ".segy"}:
        print(f"[Error] Input file must be .sgy or .segy: {input_path}")
        return 1

    with segyio.open(str(input_path), "r", ignore_geometry=True) as segy_file:
        try:
            segy_file.mmap()
        except Exception:
            pass

        sx = np.asarray(segy_file.attributes(segyio.TraceField.SourceX)[:], dtype=np.int64)
        sy = np.asarray(segy_file.attributes(segyio.TraceField.SourceY)[:], dtype=np.int64)

    if sx.size == 0:
        print("[Error] No traces found.")
        return 1

    deltas = np.stack([np.diff(sx), np.diff(sy)], axis=1)
    jump_mask = (np.abs(deltas[:, 0]) > args.jump_threshold) | (np.abs(deltas[:, 1]) > args.jump_threshold)
    jump_idx = np.where(jump_mask)[0]

    print(f"file: {input_path}")
    print(f"trace_count: {sx.size}")
    print(f"jump_threshold: {args.jump_threshold}")
    print(f"jump_count: {int(jump_idx.size)}")

    if jump_idx.size:
        bounds = np.concatenate(([-1], jump_idx, [sx.size - 1]))
        segment_lengths = np.diff(bounds)
        unique_lengths, counts = np.unique(segment_lengths, return_counts=True)
        order = np.argsort(counts)[::-1]
        print("sequential_segment_lengths:")
        for idx in order[:10]:
            print(f"  length={int(unique_lengths[idx])}, count={int(counts[idx])}")

    candidates: list[dict[str, object]] = []
    for a, b in factor_pairs(int(sx.size)):
        for width in {a, b}:
            if width <= 1 or width >= sx.size:
                continue
            candidates.append(score_candidate(sx, sy, width))

    candidates.sort(key=lambda item: (float(item["total_err"]), -int(item["width"])))

    print("best_grid_candidates:")
    for item in candidates[: args.top_k]:
        shape = item["shape"]
        print(
            "  "
            f"shape={shape[0]}x{shape[1]} "
            f"row_step={item['row_step']} col_step={item['col_step']} "
            f"row_err={item['row_err']:.6f} col_err={item['col_err']:.6f} total_err={item['total_err']:.6f}"
        )

    if candidates:
        best = candidates[0]
        print("best_guess:")
        print(f"  profile_count={best['height']}")
        print(f"  traces_per_profile={best['width']}")
        print(f"  fast_axis_step={best['row_step']}")
        print(f"  slow_axis_step={best['col_step']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
