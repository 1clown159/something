#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("[Error] Library 'zstandard' is required. Run: pip install zstandard")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress a single .dat file with zstd."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single .dat file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .zst path. Defaults to <input>.zst",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=9,
        help="Zstd compression level.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Zstd thread count. Use -1 to let zstd choose all cores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output .zst file if it already exists.",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Summary JSON path. Defaults to <output>.json",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return Path(str(input_path) + ".zst").resolve()


def resolve_summary_path(output_path: Path, summary_arg: str) -> Path:
    if summary_arg:
        return Path(summary_arg).expanduser().resolve()
    return Path(str(output_path) + ".json").resolve()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return 1
    if input_path.suffix.lower() != ".dat":
        print(f"[Error] Input file must be a .dat file: {input_path}")
        return 1

    output_path = resolve_output_path(input_path, args.output)
    summary_path = resolve_summary_path(output_path, args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"[Error] Output file already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1

    source_size = input_path.stat().st_size
    print(f"[Input]   {input_path}")
    print(f"[Output]  {output_path}")
    print(f"[Summary] {summary_path}")
    print(f"[Info]    source_size={source_size / 1024 / 1024:.2f} MB, level={args.level}, threads={args.threads}")

    compressor = zstd.ZstdCompressor(level=args.level, threads=args.threads)
    started = time.perf_counter()
    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        read_bytes, written_bytes = compressor.copy_stream(src, dst)
    elapsed = time.perf_counter() - started

    compressed_size = output_path.stat().st_size
    ratio = (source_size / compressed_size) if compressed_size else 0.0
    saving = (1.0 - compressed_size / source_size) if source_size else 0.0

    source_meta_path = input_path.with_suffix(input_path.suffix + ".json")
    summary = {
        "input_path": str(input_path),
        "input_meta_path": str(source_meta_path) if source_meta_path.exists() else "",
        "output_path": str(output_path),
        "source_size_bytes": int(source_size),
        "compressed_size_bytes": int(compressed_size),
        "copy_stream_read_bytes": int(read_bytes),
        "copy_stream_written_bytes": int(written_bytes),
        "compression_ratio": ratio,
        "space_saving_ratio": saving,
        "zstd_level": int(args.level),
        "threads": int(args.threads),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[Done]    compressed_size={compressed_size / 1024 / 1024:.2f} MB, ratio={ratio:.4f}, elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
