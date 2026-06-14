#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError:
    print("[Error] Library 'segyio' is required. Run: pip install segyio")
    sys.exit(1)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dat"
COMMON_TRACES_PER_PROFILE = [600, 601, 2001, 500, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a single SGY/SEGY file into a float32 .dat file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single .sgy or .segy file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .dat path. Defaults to experiments/dat/<input_name>.dat",
    )
    parser.add_argument(
        "--batch-traces",
        type=int,
        default=256,
        help="Number of traces to read per batch while writing the .dat file.",
    )
    parser.add_argument(
        "--traces-per-profile",
        type=int,
        default=0,
        help="Trace count in one profile. If omitted, the script tries a common heuristic.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output .dat file if it already exists.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return (DEFAULT_OUTPUT_DIR / f"{input_path.name}.dat").resolve()


def build_metadata_path(dat_path: Path) -> Path:
    return dat_path.with_suffix(dat_path.suffix + ".json")


def inspect_header_axis(segy_file: segyio.SegyFile, field: int, axis_name: str) -> dict[str, object]:
    try:
        values = np.asarray(segy_file.attributes(field)[:], dtype=np.int64)
    except Exception:
        return {
            "axis_name": axis_name,
            "available": False,
            "unique_count": None,
            "traces_per_profile": None,
            "remainder": None,
        }

    if values.size == 0:
        return {
            "axis_name": axis_name,
            "available": False,
            "unique_count": None,
            "traces_per_profile": None,
            "remainder": None,
        }

    unique_all = np.unique(values)
    unique_nonzero = unique_all[unique_all != 0]
    unique_values = unique_nonzero if unique_nonzero.size > 1 else unique_all
    unique_count = int(unique_values.size)

    if unique_count <= 1:
        return {
            "axis_name": axis_name,
            "available": False,
            "unique_count": unique_count,
            "traces_per_profile": None,
            "remainder": None,
        }

    trace_count = int(values.size)
    traces_per_profile = trace_count // unique_count
    remainder = trace_count % unique_count
    return {
        "axis_name": axis_name,
        "available": True,
        "unique_count": unique_count,
        "traces_per_profile": int(traces_per_profile),
        "remainder": int(remainder),
    }


def resolve_profile_layout(
    trace_count: int,
    traces_per_profile_arg: int,
    inline_info: dict[str, object],
    crossline_info: dict[str, object],
) -> tuple[int | None, int | None, int, str]:
    if traces_per_profile_arg > 0:
        traces_per_profile = traces_per_profile_arg
        profile_count = trace_count // traces_per_profile
        remainder = trace_count % traces_per_profile
        return traces_per_profile, profile_count, remainder, "manual"

    if bool(inline_info["available"]):
        return (
            int(inline_info["traces_per_profile"]),
            int(inline_info["unique_count"]),
            int(inline_info["remainder"]),
            "header_inline",
        )

    if bool(crossline_info["available"]):
        return (
            int(crossline_info["traces_per_profile"]),
            int(crossline_info["unique_count"]),
            int(crossline_info["remainder"]),
            "header_crossline",
        )

    for candidate in COMMON_TRACES_PER_PROFILE:
        if trace_count % candidate == 0:
            return candidate, trace_count // candidate, 0, "heuristic"

    return None, None, trace_count, "unknown"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return 1
    if input_path.suffix.lower() not in {".sgy", ".segy"}:
        print(f"[Error] Input file must be .sgy or .segy: {input_path}")
        return 1
    if args.batch_traces <= 0:
        print("[Error] --batch-traces must be a positive integer.")
        return 1

    output_path = resolve_output_path(input_path, args.output)
    meta_path = build_metadata_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"[Error] Output file already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1

    print(f"[Input]  {input_path}")
    print(f"[Output] {output_path}")
    print(f"[Meta]   {meta_path}")
    print(f"[Batch]  {args.batch_traces} traces")

    started = time.perf_counter()
    total_samples = 0

    with segyio.open(str(input_path), "r", ignore_geometry=True) as segy_file:
        try:
            segy_file.mmap()
        except Exception:
            pass

        trace_count = int(segy_file.tracecount)
        samples_per_trace = int(len(segy_file.samples))
        sample_interval = None
        if samples_per_trace > 1:
            sample_interval = float(segy_file.samples[1] - segy_file.samples[0])
        inline_info = inspect_header_axis(segy_file, segyio.TraceField.INLINE_3D, "inline")
        crossline_info = inspect_header_axis(segy_file, segyio.TraceField.CROSSLINE_3D, "crossline")
        traces_per_profile, profile_count, profile_remainder, profile_source = resolve_profile_layout(
            trace_count, args.traces_per_profile, inline_info, crossline_info
        )

        print(f"[Info]   traces={trace_count:,}, samples_per_trace={samples_per_trace:,}")
        if bool(inline_info["available"]):
            print(
                f"[Info]   inline_profile_count={int(inline_info['unique_count']):,}, inline_traces_per_profile={int(inline_info['traces_per_profile']):,}, inline_remainder={int(inline_info['remainder'])}"
            )
        else:
            print("[Info]   inline_profile_count=unavailable")
        if bool(crossline_info["available"]):
            print(
                f"[Info]   crossline_profile_count={int(crossline_info['unique_count']):,}, crossline_traces_per_profile={int(crossline_info['traces_per_profile']):,}, crossline_remainder={int(crossline_info['remainder'])}"
            )
        else:
            print("[Info]   crossline_profile_count=unavailable")
        if profile_count is not None:
            print(
                f"[Info]   selected_profile_source={profile_source}, traces_per_profile={traces_per_profile:,}, profile_count={profile_count:,}, remainder={profile_remainder}"
            )
        else:
            print("[Info]   selected_profile_source=unknown, profile_count=unknown")

        with open(output_path, "wb") as dat_handle:
            for start in range(0, trace_count, args.batch_traces):
                end = min(start + args.batch_traces, trace_count)
                batch = np.asarray(segy_file.trace.raw[start:end], dtype=np.float32)
                batch = np.ascontiguousarray(batch)
                batch.tofile(dat_handle)
                total_samples += int(batch.size)

                progress = end / trace_count * 100 if trace_count else 100.0
                print(
                    f"\r[Write]  {progress:6.2f}% ({end:,}/{trace_count:,} traces)",
                    end="",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print()

    expected_samples = trace_count * samples_per_trace
    expected_bytes = expected_samples * np.dtype(np.float32).itemsize
    actual_bytes = output_path.stat().st_size

    metadata = {
        "source_file": str(input_path),
        "dat_path": str(output_path),
        "trace_count": trace_count,
        "samples_per_trace": samples_per_trace,
        "profile_source": profile_source,
        "traces_per_profile": int(traces_per_profile) if traces_per_profile is not None else None,
        "profile_count": int(profile_count) if profile_count is not None else None,
        "profile_remainder_traces": int(profile_remainder),
        "inline_profile_count": int(inline_info["unique_count"]) if inline_info["available"] else None,
        "inline_traces_per_profile": int(inline_info["traces_per_profile"]) if inline_info["available"] else None,
        "inline_remainder_traces": int(inline_info["remainder"]) if inline_info["available"] else None,
        "crossline_profile_count": int(crossline_info["unique_count"]) if crossline_info["available"] else None,
        "crossline_traces_per_profile": int(crossline_info["traces_per_profile"]) if crossline_info["available"] else None,
        "crossline_remainder_traces": int(crossline_info["remainder"]) if crossline_info["available"] else None,
        "total_samples": int(total_samples),
        "expected_total_samples": int(expected_samples),
        "dtype": "float32",
        "storage_order": "trace_major_row_contiguous",
        "sample_interval": sample_interval,
        "batch_traces": int(args.batch_traces),
        "expected_bytes": int(expected_bytes),
        "actual_bytes": int(actual_bytes),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
    }

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if total_samples != expected_samples or actual_bytes != expected_bytes:
        print("[Error] Extracted .dat size does not match expected float32 payload.")
        return 1

    print(f"[Done]   wrote {actual_bytes / 1024 / 1024:.2f} MB in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
