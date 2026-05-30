#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import segyio
except ImportError:
    print("[Error] Library 'segyio' is required. Run: pip install segyio")
    sys.exit(1)


CANDIDATE_FIELD_NAMES = [
    "INLINE_3D",
    "CROSSLINE_3D",
    "CDP",
    "CDP_X",
    "CDP_Y",
    "FieldRecord",
    "TraceNumber",
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
    "offset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SGY/SEGY trace headers to understand inline/crossline related fields."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single .sgy or .segy file.",
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=5,
        help="Number of leading traces to print in detail.",
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=32,
        help="Number of leading traces used for automatic varying-field scan.",
    )
    parser.add_argument(
        "--max-varying-fields",
        type=int,
        default=20,
        help="Maximum number of varying fields to print from the scan.",
    )
    return parser.parse_args()


def collect_tracefield_constants() -> dict[str, int]:
    fields: dict[str, int] = {}
    for name in dir(segyio.TraceField):
        if name.startswith("_"):
            continue
        value = getattr(segyio.TraceField, name)
        if isinstance(value, int):
            fields[name] = value
    return fields


def read_field_values(segy_file: segyio.SegyFile, field_code: int, limit: int) -> list[int]:
    values: list[int] = []
    for trace_idx in range(limit):
        try:
            value = int(segy_file.header[trace_idx][field_code])
        except Exception:
            value = 0
        values.append(value)
    return values


def summarize_values(values: list[int]) -> str:
    unique_values = sorted(set(values))
    if len(unique_values) <= 8:
        return str(unique_values)
    preview = ", ".join(str(v) for v in unique_values[:8])
    return f"[{preview}, ...] (unique={len(unique_values)})"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return 1
    if input_path.suffix.lower() not in {".sgy", ".segy"}:
        print(f"[Error] Input file must be .sgy or .segy: {input_path}")
        return 1
    if args.trace_count <= 0 or args.scan_count <= 0 or args.max_varying_fields <= 0:
        print("[Error] --trace-count, --scan-count, and --max-varying-fields must be positive integers.")
        return 1

    fields = collect_tracefield_constants()

    with segyio.open(str(input_path), "r", ignore_geometry=True) as segy_file:
        try:
            segy_file.mmap()
        except Exception:
            pass

        total_traces = int(segy_file.tracecount)
        detail_count = min(args.trace_count, total_traces)
        scan_count = min(args.scan_count, total_traces)

        print(f"file: {input_path}")
        print(f"trace_count: {total_traces}")
        print(f"samples_per_trace: {int(len(segy_file.samples))}")
        print(f"detail_trace_count: {detail_count}")
        print(f"scan_trace_count: {scan_count}")
        print()
        print("=== Common Header Fields ===")

        available_candidate_names: list[str] = []
        for field_name in CANDIDATE_FIELD_NAMES:
            if field_name not in fields:
                continue
            available_candidate_names.append(field_name)
            values = read_field_values(segy_file, fields[field_name], detail_count)
            unique_count = len(set(values))
            print(f"{field_name}: values={values} unique={unique_count}")

        if not available_candidate_names:
            print("No common candidate fields were found in segyio.TraceField.")

        print()
        print("=== Per-Trace Snapshot ===")
        for trace_idx in range(detail_count):
            row_parts = [f"trace[{trace_idx}]"]
            for field_name in available_candidate_names:
                field_code = fields[field_name]
                try:
                    field_value = int(segy_file.header[trace_idx][field_code])
                except Exception:
                    field_value = 0
                row_parts.append(f"{field_name}={field_value}")
            print(" | ".join(row_parts))

        print()
        print("=== Varying Header Fields In First Traces ===")
        varying_rows: list[tuple[str, int, int, str]] = []
        for field_name, field_code in fields.items():
            values = read_field_values(segy_file, field_code, scan_count)
            unique_values = set(values)
            if len(unique_values) <= 1:
                continue
            nonzero_unique = {value for value in unique_values if value != 0}
            if not nonzero_unique:
                continue
            varying_rows.append(
                (
                    field_name,
                    field_code,
                    len(unique_values),
                    summarize_values(values),
                )
            )

        varying_rows.sort(key=lambda item: (-item[2], item[0]))
        if not varying_rows:
            print("No varying non-zero header fields were found in the scanned traces.")
        else:
            for field_name, field_code, unique_count, summary in varying_rows[: args.max_varying_fields]:
                print(
                    f"{field_name} (code={field_code}): unique={unique_count}, values={summary}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
