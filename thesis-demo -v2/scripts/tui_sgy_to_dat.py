#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np


TEXT_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240
SEG_Y_HEADER_BYTES = TEXT_HEADER_BYTES + BINARY_HEADER_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TUI SEG-Y IBM-float traces to raw float32 .dat with ragged-grid metadata."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input TUI .sgy files in trace order.")
    parser.add_argument("--output", required=True, help="Output raw float32 .dat path.")
    parser.add_argument("--batch-traces", type=int, default=4096, help="Traces per conversion batch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect headers and write only metadata, no .dat payload.")
    return parser.parse_args()


def read_binary_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        handle.seek(TEXT_HEADER_BYTES)
        header = handle.read(BINARY_HEADER_BYTES)
    if len(header) != BINARY_HEADER_BYTES:
        raise ValueError(f"File is too small to contain a SEG-Y binary header: {path}")

    sample_interval_us = struct.unpack(">H", header[16:18])[0]
    samples_per_trace = struct.unpack(">H", header[20:22])[0]
    sample_format = struct.unpack(">H", header[24:26])[0]
    extended_headers = struct.unpack(">H", header[304:306])[0]
    if extended_headers != 0:
        raise ValueError(f"Extended SEG-Y headers are not supported yet: {path}")
    if sample_format != 1:
        raise ValueError(f"Expected SEG-Y sample format 1 (IBM float32), got {sample_format}: {path}")
    if samples_per_trace <= 0:
        raise ValueError(f"Invalid samples_per_trace={samples_per_trace}: {path}")

    trace_bytes = TRACE_HEADER_BYTES + samples_per_trace * 4
    payload_bytes = path.stat().st_size - SEG_Y_HEADER_BYTES
    if payload_bytes < 0 or payload_bytes % trace_bytes != 0:
        raise ValueError(f"File size is not an integer number of traces: {path}")

    return {
        "sample_interval_us": int(sample_interval_us),
        "samples_per_trace": int(samples_per_trace),
        "sample_format": int(sample_format),
        "trace_bytes": int(trace_bytes),
        "trace_count": int(payload_bytes // trace_bytes),
    }


def parse_trace_header(header: bytes) -> tuple[int, int, int]:
    cdp = struct.unpack(">i", header[20:24])[0]
    xline = struct.unpack(">i", header[184:188])[0]
    subline = struct.unpack(">i", header[188:192])[0]
    return cdp, subline, xline


def ibm_float32_to_ieee(raw: bytes, samples_per_trace: int) -> np.ndarray:
    words = np.frombuffer(raw, dtype=">u4").astype(np.uint32)
    if words.size % samples_per_trace != 0:
        raise ValueError("Raw sample payload is not aligned to trace length.")

    zero = words == 0
    sign = np.where((words >> 31) != 0, -1.0, 1.0).astype(np.float64)
    exponent = ((words >> 24) & 0x7F).astype(np.int32) - 64
    fraction = (words & 0x00FFFFFF).astype(np.float64) / float(0x01000000)
    values = sign * fraction * np.power(16.0, exponent, dtype=np.float64)
    values[zero] = 0.0
    return values.astype(np.float32, copy=False)


def scan_headers(path: Path, info: dict[str, int]) -> dict[str, object]:
    trace_bytes = info["trace_bytes"]
    trace_count = info["trace_count"]

    pairs: set[tuple[int, int]] = set()
    by_subline: dict[int, list[int]] = {}
    cdp_mismatch = 0
    duplicate_pairs = 0

    with path.open("rb") as handle:
        handle.seek(SEG_Y_HEADER_BYTES)
        for _ in range(trace_count):
            header = handle.read(TRACE_HEADER_BYTES)
            cdp, subline, xline = parse_trace_header(header)
            pair = (subline, xline)
            if pair in pairs:
                duplicate_pairs += 1
            pairs.add(pair)
            by_subline.setdefault(subline, []).append(xline)
            if cdp != subline * 10000 + xline:
                cdp_mismatch += 1
            handle.seek(trace_bytes - TRACE_HEADER_BYTES, os.SEEK_CUR)

    ranges: list[dict[str, int]] = []
    contiguous = True
    for subline in sorted(by_subline):
        xlines = sorted(set(by_subline[subline]))
        xline_start = xlines[0]
        xline_end = xlines[-1]
        count = len(xlines)
        if count != xline_end - xline_start + 1:
            contiguous = False
        ranges.append(
            {
                "subline": int(subline),
                "xline_start": int(xline_start),
                "xline_end": int(xline_end),
                "trace_count": int(count),
            }
        )

    return {
        "trace_count": int(trace_count),
        "unique_pairs": int(len(pairs)),
        "duplicate_pairs": int(duplicate_pairs),
        "cdp_mismatch": int(cdp_mismatch),
        "subline_min": int(min(by_subline)),
        "subline_max": int(max(by_subline)),
        "subline_count": int(len(by_subline)),
        "xline_min": int(min(min(values) for values in by_subline.values())),
        "xline_max": int(max(max(values) for values in by_subline.values())),
        "one_contiguous_xline_run_per_subline": bool(contiguous),
        "subline_ranges": ranges,
    }


def convert_file(path: Path, out_handle, info: dict[str, int], batch_traces: int) -> int:
    samples_per_trace = info["samples_per_trace"]
    trace_bytes = info["trace_bytes"]
    trace_count = info["trace_count"]
    total_values = 0

    with path.open("rb") as handle:
        for start in range(0, trace_count, batch_traces):
            end = min(start + batch_traces, trace_count)
            batch_count = end - start
            raw_samples = bytearray(batch_count * samples_per_trace * 4)
            write_offset = 0
            handle.seek(SEG_Y_HEADER_BYTES + start * trace_bytes)
            for _ in range(batch_count):
                handle.seek(TRACE_HEADER_BYTES, os.SEEK_CUR)
                sample_bytes = handle.read(samples_per_trace * 4)
                raw_samples[write_offset : write_offset + len(sample_bytes)] = sample_bytes
                write_offset += len(sample_bytes)
            values = ibm_float32_to_ieee(raw_samples, samples_per_trace)
            np.ascontiguousarray(values).tofile(out_handle)
            total_values += int(values.size)
            progress = end / trace_count * 100.0
            print(f"\r[Convert] {path.name}: {progress:6.2f}% ({end:,}/{trace_count:,} traces)", end="", flush=True)
    print()
    return total_values


def main() -> int:
    args = parse_args()
    if args.batch_traces <= 0:
        raise ValueError("--batch-traces must be positive.")

    input_paths = [Path(item).expanduser().resolve() for item in args.input]
    output_path = Path(args.output).expanduser().resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and output_path.exists() and not args.overwrite:
        print(f"[Error] Output already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1
    if meta_path.exists() and not args.overwrite:
        print(f"[Error] Metadata already exists: {meta_path}")
        print("Use --overwrite to replace it.")
        return 1

    started = time.perf_counter()
    file_entries = []
    total_traces = 0
    total_values = 0
    reference_samples = None

    for path in input_paths:
        if not path.exists():
            print(f"[Error] Input not found: {path}")
            return 1
        info = read_binary_header(path)
        if reference_samples is None:
            reference_samples = info["samples_per_trace"]
        elif info["samples_per_trace"] != reference_samples:
            print(f"[Error] Mixed samples_per_trace values are not supported: {path}")
            return 1
        scan = scan_headers(path, info)
        total_traces += int(info["trace_count"])
        file_entries.append({"path": str(path), **info, **scan})

    if args.dry_run:
        print("[Mode]    dry-run, no .dat payload written")
    else:
        print(f"[Output]  {output_path}")
        with output_path.open("wb") as out_handle:
            for path, entry in zip(input_paths, file_entries):
                total_values += convert_file(path, out_handle, entry, args.batch_traces)

    elapsed = time.perf_counter() - started
    samples_per_trace = int(reference_samples or 0)
    expected_values = int(total_traces * samples_per_trace)
    expected_bytes = expected_values * 4
    actual_bytes = int(output_path.stat().st_size) if output_path.exists() and not args.dry_run else 0

    all_ranges = []
    for entry in file_entries:
        all_ranges.extend(entry["subline_ranges"])
    subline_min = min((item["subline"] for item in all_ranges), default=None)
    subline_max = max((item["subline"] for item in all_ranges), default=None)
    xline_min = min((item["xline_start"] for item in all_ranges), default=None)
    xline_max = max((item["xline_end"] for item in all_ranges), default=None)
    bounding_trace_count = 0
    missing_traces_in_bounding_box = None
    if subline_min is not None and xline_min is not None:
        bounding_trace_count = (subline_max - subline_min + 1) * (xline_max - xline_min + 1)
        missing_traces_in_bounding_box = bounding_trace_count - total_traces

    metadata = {
        "source_files": [str(path) for path in input_paths],
        "dat_path": str(output_path) if not args.dry_run else None,
        "dtype": "float32",
        "input_sample_format": "seg_y_ibm_float32",
        "storage_order": "trace_major_existing_traces_only",
        "samples_per_trace": samples_per_trace,
        "trace_count": int(total_traces),
        "total_values": expected_values,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "subline_min": subline_min,
        "subline_max": subline_max,
        "xline_min": xline_min,
        "xline_max": xline_max,
        "bounding_trace_count": int(bounding_trace_count),
        "missing_traces_in_bounding_box": missing_traces_in_bounding_box,
        "missing_trace_fraction_in_bounding_box": (
            float(missing_traces_in_bounding_box / bounding_trace_count) if bounding_trace_count else None
        ),
        "one_contiguous_xline_run_per_subline": all(
            bool(entry["one_contiguous_xline_run_per_subline"]) for entry in file_entries
        ),
        "files": file_entries,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
    }

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if not args.dry_run and (total_values != expected_values or actual_bytes != expected_bytes):
        print("[Error] Converted .dat size does not match expected float32 payload.")
        return 1

    print(f"[Meta]    {meta_path}")
    print(f"[Traces]  {total_traces:,}")
    print(f"[Values]  {expected_values:,}")
    print(f"[Missing] {missing_traces_in_bounding_box:,} traces in bounding box")
    print(f"[Done]    {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
