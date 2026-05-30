#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import compute_tui_aux_bits as aux  # noqa: E402


DEFAULT_MANIFEST = r"E:\code\thesis\20260420\postdata\postdata_blocks_manifest.json"
DEFAULT_OUTPUT_JSON = r"E:\code\thesis\20260420\postdata\outputs_postdata_aux_bits\postdata_aux_bits.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact sign and mantissa accounting for PostData blocks.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument("--zstd-threads", type=int, default=0)
    parser.add_argument("--sign-rle-level", type=int, default=3)
    parser.add_argument("--lzma-preset", type=int, default=6)
    parser.add_argument("--chunk-values", type=int, default=8_000_000)
    parser.add_argument("--average-nll-bits", type=float, default=None)
    parser.add_argument("--exp-bytes", type=float, default=None)
    parser.add_argument("--exp-seconds", type=float, default=None)
    parser.add_argument("--profile-range", default="", help="Inclusive PostData profile range, for example 325-349.")
    parser.add_argument("--limit-blocks", type=int, default=0)
    parser.add_argument("--limit-values", type=int, default=0)
    parser.add_argument("--skip-bitshuffle", action="store_true")
    parser.add_argument("--bitshuffle-mode", choices=["auto", "temp", "multipass"], default="auto")
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _parse_profile_range(value: str) -> Tuple[int, int] | None:
    text = value.strip()
    if not text:
        return None
    parts = text.replace(":", "-").split("-")
    if len(parts) != 2:
        raise ValueError("--profile-range must look like START-END, for example 325-349.")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError("--profile-range START must be <= END.")
    return start, end


def _select_profile_range_blocks(blocks: List[Dict[str, Any]], profile_range: Tuple[int, int] | None) -> List[Dict[str, Any]]:
    if profile_range is None:
        return blocks

    wanted_start, wanted_end = profile_range
    selected: List[Dict[str, Any]] = []
    for block in blocks:
        block_start = int(block["profile_start"])
        block_end = int(block["profile_end"])
        start = max(wanted_start, block_start)
        end = min(wanted_end, block_end)
        if start > end:
            continue

        traces = int(block["traces_per_profile"])
        samples = int(block["samples_per_trace"])
        local_profile_start = start - block_start
        profiles = end - start + 1
        value_delta = local_profile_start * traces * samples
        value_count = profiles * traces * samples

        sliced = dict(block)
        sliced["block_id"] = f"{block['block_id']}_aux_p{start:04d}_{end:04d}"
        sliced["profile_start"] = start
        sliced["profile_end"] = end
        sliced["profiles"] = profiles
        sliced["trace_offset"] = int(block.get("trace_offset", 0)) + local_profile_start * traces
        sliced["trace_count"] = profiles * traces
        sliced["value_offset"] = int(block.get("value_offset", 0)) + value_delta
        sliced["value_count"] = value_count
        sliced["byte_offset"] = int(block.get("byte_offset", 0)) + value_delta * 4
        sliced["byte_count"] = value_count * 4
        sliced["shape"] = [profiles, traces, samples]
        # PostData profile slices point into the original source file, not a standalone block file.
        sliced.pop("dat_path", None)
        selected.append(sliced)

    if not selected:
        raise ValueError(f"--profile-range {wanted_start}-{wanted_end} does not overlap selected PostData blocks.")
    return selected


def _retag_summary(summary: Dict) -> Dict:
    for row in summary.get("combined_schemes", []):
        if isinstance(row.get("scheme"), str):
            row["scheme"] = row["scheme"].replace("TUI predicted exp", "PostData predicted exp")
        if row.get("exp_source") == "provided_exp_bytes":
            row["exponent_method"] = "actual global_diag range-coded exponent bitstream"
        elif row.get("exp_source") == "average_nll_bits":
            row["exponent_method"] = "sampled NLL estimate, not a real bitstream"
    return summary


def main() -> int:
    args = parse_args()
    if args.chunk_values <= 0:
        raise ValueError("--chunk-values must be positive.")

    manifest = aux.load_json(args.manifest)
    blocks = aux.selected_blocks(manifest, args.limit_blocks)
    profile_range = _parse_profile_range(args.profile_range)
    blocks = _select_profile_range_blocks(blocks, profile_range)
    count = aux.total_selected_values(blocks, args.limit_values)
    if count <= 0:
        raise ValueError("No PostData values selected.")

    aux.print_progress(f"[Info] Selected {len(blocks)} blocks, {count} float32 values.", args.quiet)
    aux.print_progress("[Info] Pass 1: sign bitpack, RLE stats, mantissa whole/raw PathL streams.", args.quiet)
    raw = aux.sign_pack_and_raw_mantissa_pass(manifest, blocks, args)

    aux.print_progress("[Info] Pass 2: sign RLE compressed candidates.", args.quiet)
    rle = aux.sign_rle_compression_pass(manifest, blocks, args, raw["sign_rle"]["length_dtype"], int(raw["sign_rle"]["run_count"]))

    bitshuffle: Dict[str, Any] = {}
    if not args.skip_bitshuffle:
        mode = args.bitshuffle_mode
        if mode == "auto":
            mode = "temp" if count % 8 == 0 else "multipass"
        if mode == "temp":
            aux.print_progress("[Info] Pass 3: PathL bitshuffle candidates via temporary bit-plane streams.", args.quiet)
            bitshuffle = aux.bitshuffle_temp_pass(manifest, blocks, args)
        else:
            aux.print_progress("[Info] Pass 3: PathL bitshuffle candidates. This scans PostData once per bit plane.", args.quiet)
            bitshuffle = aux.bitshuffle_pass(manifest, blocks, args)

    summary = _retag_summary(aux.build_summary(raw, rle, bitshuffle, args))
    summary.setdefault("settings", {})["profile_range"] = None if profile_range is None else list(profile_range)
    aux.save_json(args.output_json, summary)
    aux.print_progress(f"[OK] Saved exact PostData auxiliary-bit summary to {args.output_json}", args.quiet)

    sign = summary["sign"]
    mant = summary["mantissa"]
    aux.print_progress(
        f"[OK] sign best: {sign['best_bytes']} bytes ({sign['best_method']}, {sign['best_seconds']:.3f}s); "
        f"PathL-lite: {mant['pathL_lite_bytes']} bytes ({mant['pathL_lite_method']}, {mant['pathL_lite_seconds']:.3f}s); "
        f"whole-zstd: {mant['whole_zstd_bytes']} bytes ({mant['whole_zstd_seconds']:.3f}s)",
        args.quiet,
    )
    for row in summary["combined_schemes"]:
        seconds_text = f", aux_seconds={row['aux_seconds']:.3f}s"
        if row.get("total_seconds") is not None:
            seconds_text += f", total_seconds={row['total_seconds']:.3f}s"
        aux.print_progress(
            "[OK] {scheme}: total={total:.0f} bytes, bps={bps:.6f}, ratio={ratio:.6f}x{seconds}".format(
                scheme=row["scheme"],
                total=row["total_bytes"],
                bps=row["bits_per_value"],
                ratio=row["compression_ratio_vs_float32"],
                seconds=seconds_text,
            ),
            args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
