#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError:
    print("[Error] Library 'segyio' is required. Run: pip install segyio")
    sys.exit(1)


FORMAT_FAMILY = {
    1: ("float", "IBM 4-byte floating point"),
    2: ("integer", "4-byte signed integer"),
    3: ("integer", "2-byte signed integer"),
    5: ("float", "IEEE 4-byte floating point"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one SGY/SEGY file and report whether its sample format is integer or float."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single .sgy or .segy file.",
    )
    return parser.parse_args()


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

        format_code = int(segy_file.format)
        family, format_name = FORMAT_FAMILY.get(format_code, ("unknown", "Unknown or uncommon SEG-Y sample format"))

        first_trace = np.asarray(segy_file.trace[0])
        trace_dtype = str(first_trace.dtype)
        dtype_family = "float" if first_trace.dtype.kind == "f" else "integer" if first_trace.dtype.kind in {"i", "u"} else "unknown"

        print(f"file: {input_path}")
        print(f"trace_count: {int(segy_file.tracecount)}")
        print(f"samples_per_trace: {int(len(segy_file.samples))}")
        print(f"segy_format_code: {format_code}")
        print(f"segy_format_name: {format_name}")
        print(f"segy_numeric_family: {family}")
        print(f"trace_array_dtype: {trace_dtype}")
        print(f"trace_array_family: {dtype_family}")

        if family == "unknown":
            print("note: sample format code is not in the common mapping table; fallback dtype is shown above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
