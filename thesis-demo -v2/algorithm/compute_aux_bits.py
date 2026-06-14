#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import zstandard as zstd


DEFAULT_MANIFEST = r"E:\code\thesis\20260420\tui_blocks_manifest.json"
DEFAULT_OUTPUT_JSON = r"E:\code\thesis\20260420\outputs_tui_aux_bits\tui_aux_bits.json"


class CountingSink:
    def __init__(self) -> None:
        self.count = 0

    def write(self, data: bytes) -> int:
        self.count += len(data)
        return len(data)

    def flush(self) -> None:
        return None


class StreamCounter:
    def __init__(
        self,
        codec: str,
        *,
        zstd_level: int,
        zstd_threads: int,
        lzma_preset: int,
        known_size: int | None = None,
    ) -> None:
        self.codec = codec
        self.count = 0
        self.write_seconds = 0.0
        self.close_seconds = 0.0
        self._sink: CountingSink | None = None
        self._writer: Any = None
        self._zstd_obj: Any = None
        self._lzma: lzma.LZMACompressor | None = None
        if codec == "zstd":
            compressor = zstd.ZstdCompressor(level=zstd_level, threads=zstd_threads)
            if known_size is None:
                self._sink = CountingSink()
                self._writer = compressor.stream_writer(self._sink, closefd=False)
                self._writer.__enter__()
            else:
                self._zstd_obj = compressor.compressobj(size=int(known_size))
        elif codec == "lzma":
            self._lzma = lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=lzma_preset)
        else:
            raise ValueError(f"Unsupported codec: {codec}")

    def write(self, data: bytes) -> None:
        if not data:
            return
        start = time.perf_counter()
        if self.codec == "zstd":
            if self._zstd_obj is not None:
                self.count += len(self._zstd_obj.compress(data))
            else:
                self._writer.write(data)
        else:
            assert self._lzma is not None
            self.count += len(self._lzma.compress(data))
        self.write_seconds += time.perf_counter() - start

    def close(self) -> int:
        start = time.perf_counter()
        if self.codec == "zstd":
            if self._zstd_obj is not None:
                self.count += len(self._zstd_obj.flush())
            else:
                self._writer.__exit__(None, None, None)
                assert self._sink is not None
                self.count = self._sink.count
        else:
            assert self._lzma is not None
            self.count += len(self._lzma.flush())
        self.close_seconds += time.perf_counter() - start
        return self.count

    @property
    def total_seconds(self) -> float:
        return float(self.write_seconds + self.close_seconds)


class BitPacker:
    def __init__(self, bitorder: str) -> None:
        self.bitorder = bitorder
        self._pending = np.empty(0, dtype=np.uint8)

    def pack(self, bits: np.ndarray) -> bytes:
        flat = bits.reshape(-1).astype(np.uint8, copy=False)
        if self._pending.size:
            flat = np.concatenate([self._pending, flat])
        full = (flat.size // 8) * 8
        if full == 0:
            self._pending = flat.copy()
            return b""
        out = np.packbits(flat[:full], bitorder=self.bitorder).tobytes()
        self._pending = flat[full:].copy()
        return out

    def flush(self) -> bytes:
        if self._pending.size == 0:
            return b""
        out = np.packbits(self._pending, bitorder=self.bitorder).tobytes()
        self._pending = np.empty(0, dtype=np.uint8)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact sign and mantissa accounting for TUI blocks.")
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
    parser.add_argument("--limit-blocks", type=int, default=0)
    parser.add_argument("--limit-values", type=int, default=0)
    parser.add_argument("--skip-bitshuffle", action="store_true")
    parser.add_argument("--bitshuffle-mode", choices=["auto", "temp", "multipass"], default="auto")
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def selected_blocks(manifest: Dict[str, Any], limit_blocks: int) -> List[Dict[str, Any]]:
    blocks = list(manifest.get("blocks", []))
    if limit_blocks > 0:
        blocks = blocks[:limit_blocks]
    return blocks


def block_flat_memmap(block: Dict[str, Any], manifest: Dict[str, Any]) -> np.memmap:
    value_count = int(block["value_count"])
    dat_path = block.get("dat_path")
    if dat_path:
        return np.memmap(str(dat_path), dtype=np.float32, mode="r", shape=(value_count,))
    source = str(manifest["source_dat_path"])
    return np.memmap(source, dtype=np.float32, mode="r", offset=int(block["byte_offset"]), shape=(value_count,))


def iter_u32_chunks(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    chunk_values: int,
    limit_values: int,
) -> Iterator[Tuple[str, np.ndarray]]:
    remaining = int(limit_values) if limit_values > 0 else None
    for block in blocks:
        block_id = str(block["block_id"])
        flat = block_flat_memmap(block, manifest)
        size = int(flat.shape[0])
        if remaining is not None:
            size = min(size, remaining)
        for start in range(0, size, chunk_values):
            stop = min(start + chunk_values, size)
            chunk = np.asarray(flat[start:stop], dtype=np.float32)
            yield block_id, chunk.view(np.uint32)
        if remaining is not None:
            remaining -= size
            if remaining <= 0:
                break


def total_selected_values(blocks: List[Dict[str, Any]], limit_values: int) -> int:
    total = sum(int(block["value_count"]) for block in blocks)
    if limit_values > 0:
        total = min(total, int(limit_values))
    return int(total)


def packed_mantissa23(mants: np.ndarray) -> bytes:
    vals = mants.reshape(-1).astype(np.uint32, copy=False)
    packed = np.empty((vals.size, 3), dtype=np.uint8)
    packed[:, 0] = (vals & 0xFF).astype(np.uint8)
    packed[:, 1] = ((vals >> 8) & 0xFF).astype(np.uint8)
    packed[:, 2] = ((vals >> 16) & 0x7F).astype(np.uint8)
    return packed.reshape(-1).tobytes()


def sign_run_lengths(signs: np.ndarray) -> Tuple[int, np.ndarray]:
    bits = signs.reshape(-1).astype(np.uint8, copy=False)
    changes = np.where(np.diff(bits))[0] + 1
    boundaries = np.concatenate(([0], changes, [bits.size]))
    return int(bits[0]), np.diff(boundaries).astype(np.int64)


def merge_run_lengths(
    first_sign: int,
    lengths: np.ndarray,
    current_sign: int | None,
    current_run: int,
) -> Tuple[np.ndarray, int, int]:
    if current_sign is None:
        sequence = lengths
        sequence_first_sign = first_sign
    elif first_sign == current_sign:
        if lengths.size == 1:
            return np.empty(0, dtype=np.int64), current_sign, int(current_run + int(lengths[0]))
        sequence = np.concatenate(([current_run + int(lengths[0])], lengths[1:]))
        sequence_first_sign = current_sign
    else:
        sequence = np.concatenate(([current_run], lengths))
        sequence_first_sign = current_sign

    if sequence.size == 1:
        return np.empty(0, dtype=np.int64), sequence_first_sign, int(sequence[0])
    completed = sequence[:-1].astype(np.int64, copy=False)
    next_sign = sequence_first_sign ^ ((int(sequence.size) - 1) & 1)
    return completed, int(next_sign), int(sequence[-1])


def sign_run_length_batches(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    chunk_values: int,
    limit_values: int,
) -> Iterator[np.ndarray]:
    current_sign: int | None = None
    current_run = 0
    for _, u32 in iter_u32_chunks(manifest, blocks, chunk_values, limit_values):
        signs = ((u32 >> 31) & 1).astype(np.uint8, copy=False)
        if signs.size == 0:
            continue
        first_sign, lengths = sign_run_lengths(signs)
        completed, current_sign, current_run = merge_run_lengths(first_sign, lengths, current_sign, current_run)
        if completed.size:
            yield completed
    if current_sign is not None:
        yield np.asarray([current_run], dtype=np.int64)


def sign_pack_and_raw_mantissa_pass(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pass_start = time.perf_counter()
    value_count = total_selected_values(blocks, args.limit_values)
    sign_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=math.ceil(value_count / 8),
    )
    sign_lzma = StreamCounter("lzma", zstd_level=args.zstd_level, zstd_threads=args.zstd_threads, lzma_preset=args.lzma_preset)
    whole_mant_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=value_count * 3,
    )
    mid15_raw_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=value_count * 2,
    )
    mid15_low_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=value_count,
    )
    mid15_high_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=value_count,
    )
    side_raw_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=value_count,
    )
    sign_packer = BitPacker(bitorder="little")

    max_run = 0
    run_count = 0
    first_run_length: int | None = None
    values = 0
    current_sign: int | None = None
    current_run = 0

    def consume_completed_runs(runs: np.ndarray) -> None:
        nonlocal first_run_length, max_run, run_count
        if runs.size == 0:
            return
        if first_run_length is None:
            first_run_length = int(runs[0])
        max_run = max(max_run, int(runs.max()))
        run_count += int(runs.size)

    for _, u32 in iter_u32_chunks(manifest, blocks, args.chunk_values, args.limit_values):
        values += int(u32.size)
        signs = ((u32 >> 31) & 1).astype(np.uint8, copy=False)
        packed_signs = sign_packer.pack(signs)
        sign_zstd.write(packed_signs)
        sign_lzma.write(packed_signs)

        if signs.size:
            first_sign, lengths = sign_run_lengths(signs)
            completed, current_sign, current_run = merge_run_lengths(first_sign, lengths, current_sign, current_run)
            consume_completed_runs(completed)

        mants = (u32 & 0x7FFFFF).astype(np.uint32, copy=False)
        whole_mant_zstd.write(packed_mantissa23(mants))

        mid15 = ((mants >> 4) & 0x7FFF).astype(np.uint32, copy=False)
        mid15_raw_zstd.write(mid15.astype(np.uint16, copy=False).tobytes())
        mid15_low_zstd.write((mid15 & 0xFF).astype(np.uint8, copy=False).tobytes())
        mid15_high_zstd.write((mid15 >> 8).astype(np.uint8, copy=False).tobytes())

        lo4 = (mants & 0xF).astype(np.uint8, copy=False)
        hi4 = ((mants >> 19) & 0xF).astype(np.uint8, copy=False)
        side_raw_zstd.write(((hi4 << 4) | lo4).astype(np.uint8, copy=False).tobytes())

    tail = sign_packer.flush()
    sign_zstd.write(tail)
    sign_lzma.write(tail)
    if current_sign is not None:
        consume_completed_runs(np.asarray([current_run], dtype=np.int64))

    sign_bitpack_zstd_bytes = int(sign_zstd.close())
    sign_bitpack_lzma_bytes = int(sign_lzma.close())
    whole_mantissa_zstd_bytes = int(whole_mant_zstd.close())
    mid15_raw_u16_zstd_bytes = int(mid15_raw_zstd.close())
    mid15_bytesplit_zstd_bytes = int(mid15_low_zstd.close() + mid15_high_zstd.close())
    side_hi4lo4_raw_zstd_bytes = int(side_raw_zstd.close())
    pass_wall_seconds = time.perf_counter() - pass_start
    compress_seconds = float(
        sign_zstd.total_seconds
        + sign_lzma.total_seconds
        + whole_mant_zstd.total_seconds
        + mid15_raw_zstd.total_seconds
        + mid15_low_zstd.total_seconds
        + mid15_high_zstd.total_seconds
        + side_raw_zstd.total_seconds
    )

    return {
        "value_count": int(values),
        "sign_bitpack_zstd_bytes": sign_bitpack_zstd_bytes,
        "sign_bitpack_lzma_bytes": sign_bitpack_lzma_bytes,
        "whole_mantissa_zstd_bytes": whole_mantissa_zstd_bytes,
        "mid15_raw_u16_zstd_bytes": mid15_raw_u16_zstd_bytes,
        "mid15_bytesplit_zstd_bytes": mid15_bytesplit_zstd_bytes,
        "side_hi4lo4_raw_zstd_bytes": side_hi4lo4_raw_zstd_bytes,
        "sign_bitpack_zstd_seconds": float(sign_zstd.total_seconds),
        "sign_bitpack_lzma_seconds": float(sign_lzma.total_seconds),
        "whole_mantissa_zstd_seconds": float(whole_mant_zstd.total_seconds),
        "mid15_raw_u16_zstd_seconds": float(mid15_raw_zstd.total_seconds),
        "mid15_bytesplit_zstd_seconds": float(mid15_low_zstd.total_seconds + mid15_high_zstd.total_seconds),
        "side_hi4lo4_raw_zstd_seconds": float(side_raw_zstd.total_seconds),
        "timing": {
            "pass1_wall_seconds": float(pass_wall_seconds),
            "pass1_compressor_seconds": compress_seconds,
            "pass1_preprocess_overhead_seconds": float(max(0.0, pass_wall_seconds - compress_seconds)),
        },
        "sign_rle": {
            "run_count": int(run_count),
            "max_run": int(max_run),
            "first_run_length": None if first_run_length is None else int(first_run_length),
            "length_dtype": "uint8" if max_run < 256 else "uint16",
        },
    }


def sign_rle_compression_pass(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    args: argparse.Namespace,
    length_dtype: str,
    run_count: int,
) -> Dict[str, Any]:
    pass_start = time.perf_counter()
    # The caller already selected the historical dtype rule: uint8 if max_run < 256, else uint16.
    # Compute the known zstd input sizes from the first pass to match one-shot zstd.compress framing.
    length_bytes = 1 if length_dtype == "uint8" else 2
    lengths_zstd = StreamCounter(
        "zstd",
        zstd_level=args.sign_rle_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=run_count * length_bytes,
    )
    lengths_lzma = StreamCounter("lzma", zstd_level=args.zstd_level, zstd_threads=args.zstd_threads, lzma_preset=args.lzma_preset)
    deltas_zstd = StreamCounter(
        "zstd",
        zstd_level=args.sign_rle_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=run_count * 2,
    )
    deltas_lzma = StreamCounter("lzma", zstd_level=args.zstd_level, zstd_threads=args.zstd_threads, lzma_preset=args.lzma_preset)

    previous_run: int | None = None
    out_dtype = np.uint8 if length_dtype == "uint8" else np.uint16

    def emit_runs(run_lengths: np.ndarray) -> None:
        nonlocal previous_run
        if run_lengths.size == 0:
            return
        run_arr = run_lengths.astype(np.uint64, copy=False).astype(out_dtype)
        if previous_run is None:
            deltas = np.diff(run_lengths, prepend=run_lengths[0])
        else:
            deltas = np.diff(run_lengths, prepend=previous_run)
        delta_arr = deltas.astype(np.int64, copy=False).astype(np.int16)
        lengths_zstd.write(run_arr.tobytes())
        lengths_lzma.write(run_arr.tobytes())
        deltas_zstd.write(delta_arr.tobytes())
        deltas_lzma.write(delta_arr.tobytes())
        previous_run = int(run_lengths[-1])

    for run_lengths in sign_run_length_batches(manifest, blocks, args.chunk_values, args.limit_values):
        emit_runs(run_lengths)

    rle_lengths_zstd_bytes = int(lengths_zstd.close())
    rle_lengths_lzma_bytes = int(lengths_lzma.close())
    rle_deltas_zstd_bytes = int(deltas_zstd.close())
    rle_deltas_lzma_bytes = int(deltas_lzma.close())
    pass_wall_seconds = time.perf_counter() - pass_start
    compress_seconds = float(lengths_zstd.total_seconds + lengths_lzma.total_seconds + deltas_zstd.total_seconds + deltas_lzma.total_seconds)

    return {
        "rle_lengths_zstd_bytes": rle_lengths_zstd_bytes,
        "rle_lengths_lzma_bytes": rle_lengths_lzma_bytes,
        "rle_deltas_zstd_bytes": rle_deltas_zstd_bytes,
        "rle_deltas_lzma_bytes": rle_deltas_lzma_bytes,
        "rle_lengths_zstd_seconds": float(lengths_zstd.total_seconds),
        "rle_lengths_lzma_seconds": float(lengths_lzma.total_seconds),
        "rle_deltas_zstd_seconds": float(deltas_zstd.total_seconds),
        "rle_deltas_lzma_seconds": float(deltas_lzma.total_seconds),
        "timing": {
            "pass2_wall_seconds": float(pass_wall_seconds),
            "pass2_compressor_seconds": compress_seconds,
            "pass2_preprocess_overhead_seconds": float(max(0.0, pass_wall_seconds - compress_seconds)),
        },
    }


def bitshuffle_pass(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pass_start = time.perf_counter()
    value_count = total_selected_values(blocks, args.limit_values)
    mid15_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=math.ceil(value_count * 15 / 8),
    )
    side_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=math.ceil(value_count * 8 / 8),
    )
    mid15_packer = BitPacker(bitorder="big")
    side_packer = BitPacker(bitorder="big")

    for bit in range(15):
        for _, u32 in iter_u32_chunks(manifest, blocks, args.chunk_values, args.limit_values):
            mants = (u32 & 0x7FFFFF).astype(np.uint32, copy=False)
            mid15 = ((mants >> 4) & 0x7FFF).astype(np.uint32, copy=False)
            mid15_zstd.write(mid15_packer.pack((mid15 >> bit) & 1))
            if bit < 8:
                lo4 = (mants & 0xF).astype(np.uint8, copy=False)
                hi4 = ((mants >> 19) & 0xF).astype(np.uint8, copy=False)
                side = ((hi4 << 4) | lo4).astype(np.uint8, copy=False)
                side_zstd.write(side_packer.pack((side >> bit) & 1))
    mid15_zstd.write(mid15_packer.flush())
    side_zstd.write(side_packer.flush())
    mid15_bytes = int(mid15_zstd.close())
    side_bytes = int(side_zstd.close())
    pass_wall_seconds = time.perf_counter() - pass_start
    compress_seconds = float(mid15_zstd.total_seconds + side_zstd.total_seconds)
    return {
        "mid15_bitshuffle_zstd_bytes": mid15_bytes,
        "side_hi4lo4_bitshuffle_zstd_bytes": side_bytes,
        "mid15_bitshuffle_zstd_seconds": float(mid15_zstd.total_seconds),
        "side_hi4lo4_bitshuffle_zstd_seconds": float(side_zstd.total_seconds),
        "timing": {
            "pass3_wall_seconds": float(pass_wall_seconds),
            "pass3_compressor_seconds": compress_seconds,
            "pass3_preprocess_overhead_seconds": float(max(0.0, pass_wall_seconds - compress_seconds)),
        },
    }


def write_stream_file_to_counter(path: Path, counter: StreamCounter) -> None:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            counter.write(chunk)


def bitshuffle_temp_pass(
    manifest: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pass_start = time.perf_counter()
    value_count = total_selected_values(blocks, args.limit_values)
    if value_count % 8 != 0:
        raise ValueError("Temporary bitshuffle mode requires selected value count divisible by 8.")

    temp_root = Path(args.temp_dir) if args.temp_dir else Path(args.output_json).resolve().parent / "bitshuffle_tmp"
    if temp_root.exists() and not args.keep_temp:
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    mid_paths = [temp_root / f"mid15_bit{bit:02d}.bin" for bit in range(15)]
    side_paths = [temp_root / f"hi4lo4_bit{bit:02d}.bin" for bit in range(8)]
    mid_files = [path.open("wb") for path in mid_paths]
    side_files = [path.open("wb") for path in side_paths]
    mid_packers = [BitPacker(bitorder="big") for _ in range(15)]
    side_packers = [BitPacker(bitorder="big") for _ in range(8)]

    try:
        temp_write_start = time.perf_counter()
        for _, u32 in iter_u32_chunks(manifest, blocks, args.chunk_values, args.limit_values):
            mants = (u32 & 0x7FFFFF).astype(np.uint32, copy=False)
            mid15 = ((mants >> 4) & 0x7FFF).astype(np.uint32, copy=False)
            lo4 = (mants & 0xF).astype(np.uint8, copy=False)
            hi4 = ((mants >> 19) & 0xF).astype(np.uint8, copy=False)
            side = ((hi4 << 4) | lo4).astype(np.uint8, copy=False)
            for bit in range(15):
                mid_files[bit].write(mid_packers[bit].pack((mid15 >> bit) & 1))
                if bit < 8:
                    side_files[bit].write(side_packers[bit].pack((side >> bit) & 1))
        for bit in range(15):
            mid_files[bit].write(mid_packers[bit].flush())
            if bit < 8:
                side_files[bit].write(side_packers[bit].flush())
        temp_write_seconds = time.perf_counter() - temp_write_start
    finally:
        for handle in mid_files + side_files:
            handle.close()

    mid15_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=math.ceil(value_count * 15 / 8),
    )
    side_zstd = StreamCounter(
        "zstd",
        zstd_level=args.zstd_level,
        zstd_threads=args.zstd_threads,
        lzma_preset=args.lzma_preset,
        known_size=math.ceil(value_count * 8 / 8),
    )
    for path in mid_paths:
        write_stream_file_to_counter(path, mid15_zstd)
    for path in side_paths:
        write_stream_file_to_counter(path, side_zstd)

    mid15_bytes = int(mid15_zstd.close())
    side_bytes = int(side_zstd.close())
    pass_wall_seconds = time.perf_counter() - pass_start
    compress_seconds = float(mid15_zstd.total_seconds + side_zstd.total_seconds)
    result = {
        "mid15_bitshuffle_zstd_bytes": mid15_bytes,
        "side_hi4lo4_bitshuffle_zstd_bytes": side_bytes,
        "mid15_bitshuffle_zstd_seconds": float(mid15_zstd.total_seconds),
        "side_hi4lo4_bitshuffle_zstd_seconds": float(side_zstd.total_seconds),
        "timing": {
            "pass3_wall_seconds": float(pass_wall_seconds),
            "pass3_temp_write_seconds": float(temp_write_seconds),
            "pass3_compressor_seconds": compress_seconds,
            "pass3_preprocess_overhead_seconds": float(max(0.0, pass_wall_seconds - compress_seconds)),
        },
    }
    if not args.keep_temp:
        shutil.rmtree(temp_root)
    return result


def best_method(candidates: Dict[str, int]) -> Tuple[int, str]:
    method, size = min(candidates.items(), key=lambda item: item[1])
    return int(size), method


def build_summary(raw: Dict[str, Any], rle: Dict[str, Any], bitshuffle: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    sign_candidates = {
        "bitpack+zstd": raw["sign_bitpack_zstd_bytes"],
        "bitpack+lzma": raw["sign_bitpack_lzma_bytes"],
        "rle_lengths+zstd": rle["rle_lengths_zstd_bytes"],
        "rle_lengths+lzma": rle["rle_lengths_lzma_bytes"],
        "rle_deltas+zstd": rle["rle_deltas_zstd_bytes"],
        "rle_deltas+lzma": rle["rle_deltas_lzma_bytes"],
    }
    sign_candidate_seconds = {
        "bitpack+zstd": raw["sign_bitpack_zstd_seconds"],
        "bitpack+lzma": raw["sign_bitpack_lzma_seconds"],
        "rle_lengths+zstd": rle["rle_lengths_zstd_seconds"],
        "rle_lengths+lzma": rle["rle_lengths_lzma_seconds"],
        "rle_deltas+zstd": rle["rle_deltas_zstd_seconds"],
        "rle_deltas+lzma": rle["rle_deltas_lzma_seconds"],
    }
    sign_bytes, sign_method = best_method(sign_candidates)
    sign_seconds = float(sign_candidate_seconds[sign_method])

    mid15_candidates = {
        "mid15_raw_u16_zstd": raw["mid15_raw_u16_zstd_bytes"],
        "mid15_bytesplit_zstd": raw["mid15_bytesplit_zstd_bytes"],
    }
    mid15_candidate_seconds = {
        "mid15_raw_u16_zstd": raw["mid15_raw_u16_zstd_seconds"],
        "mid15_bytesplit_zstd": raw["mid15_bytesplit_zstd_seconds"],
    }
    side_candidates = {
        "hi4lo4_raw_zstd": raw["side_hi4lo4_raw_zstd_bytes"],
    }
    side_candidate_seconds = {
        "hi4lo4_raw_zstd": raw["side_hi4lo4_raw_zstd_seconds"],
    }
    if bitshuffle:
        mid15_candidates["mid15_bitshuffle_zstd"] = bitshuffle["mid15_bitshuffle_zstd_bytes"]
        side_candidates["hi4lo4_bitshuffle_zstd"] = bitshuffle["side_hi4lo4_bitshuffle_zstd_bytes"]
        mid15_candidate_seconds["mid15_bitshuffle_zstd"] = bitshuffle["mid15_bitshuffle_zstd_seconds"]
        side_candidate_seconds["hi4lo4_bitshuffle_zstd"] = bitshuffle["side_hi4lo4_bitshuffle_zstd_seconds"]

    mid15_bytes, mid15_method = best_method(mid15_candidates)
    side_bytes, side_method = best_method(side_candidates)
    mid15_seconds = float(mid15_candidate_seconds[mid15_method])
    side_seconds = float(side_candidate_seconds[side_method])
    path_l_bytes = int(mid15_bytes + side_bytes)
    path_l_seconds = float(mid15_seconds + side_seconds)
    whole_zstd_bytes = int(raw["whole_mantissa_zstd_bytes"])
    whole_zstd_seconds = float(raw["whole_mantissa_zstd_seconds"])

    count = int(raw["value_count"])
    raw_float32_bytes = int(count * 4)
    exp_bytes = None
    exp_source = None
    if args.exp_bytes is not None:
        exp_bytes = float(args.exp_bytes)
        exp_source = "provided_exp_bytes"
    elif args.average_nll_bits is not None:
        exp_bytes = float(count * args.average_nll_bits / 8.0)
        exp_source = "average_nll_bits"
    exp_seconds = None if getattr(args, "exp_seconds", None) is None else float(args.exp_seconds)

    schemes: List[Dict[str, Any]] = []
    if exp_bytes is not None:
        for name, mant_bytes, mant_method, mant_seconds in [
            ("TUI predicted exp + sign-best + PathL-lite", path_l_bytes, f"{mid15_method} + {side_method}", path_l_seconds),
            ("TUI predicted exp + sign-best + mantissa-whole-zstd", whole_zstd_bytes, "mantissa23_packed+zstd", whole_zstd_seconds),
        ]:
            total = float(sign_bytes + exp_bytes + mant_bytes)
            aux_seconds = float(sign_seconds + mant_seconds)
            schemes.append(
                {
                    "scheme": name,
                    "sign_bytes": int(sign_bytes),
                    "sign_method": sign_method,
                    "sign_seconds": sign_seconds,
                    "exp_bytes": exp_bytes,
                    "exp_source": exp_source,
                    "exp_seconds": exp_seconds,
                    "mant_bytes": int(mant_bytes),
                    "mant_method": mant_method,
                    "mant_seconds": float(mant_seconds),
                    "total_bytes": total,
                    "aux_seconds": aux_seconds,
                    "total_seconds": None if exp_seconds is None else float(exp_seconds + aux_seconds),
                    "bits_per_value": float(total * 8.0 / count),
                    "compression_ratio_vs_float32": float(raw_float32_bytes / total),
                    "space_saving_ratio": float(1.0 - total / raw_float32_bytes),
                }
            )

    return {
        "value_count": count,
        "raw_float32_bytes": raw_float32_bytes,
        "settings": {
            "zstd_level": int(args.zstd_level),
            "zstd_threads": int(args.zstd_threads),
            "sign_rle_level": int(args.sign_rle_level),
            "lzma_preset": int(args.lzma_preset),
            "chunk_values": int(args.chunk_values),
            "limit_blocks": int(args.limit_blocks),
            "limit_values": int(args.limit_values),
            "skip_bitshuffle": bool(args.skip_bitshuffle),
            "bitshuffle_mode": str(args.bitshuffle_mode),
            "keep_temp": bool(args.keep_temp),
        },
        "sign": {
            "best_bytes": int(sign_bytes),
            "best_method": sign_method,
            "best_seconds": sign_seconds,
            "candidates": {key: int(value) for key, value in sign_candidates.items()},
            "candidate_seconds": {key: float(value) for key, value in sign_candidate_seconds.items()},
            "rle": raw["sign_rle"],
        },
        "mantissa": {
            "whole_zstd_bytes": whole_zstd_bytes,
            "whole_zstd_seconds": whole_zstd_seconds,
            "whole_zstd_method": "mantissa23_packed+zstd",
            "pathL_lite_bytes": path_l_bytes,
            "pathL_lite_seconds": path_l_seconds,
            "pathL_lite_method": f"{mid15_method} + {side_method}",
            "pathL_lite_detail": {
                "mid_bytes": int(mid15_bytes),
                "mid_method": mid15_method,
                "mid_seconds": mid15_seconds,
                "mid_candidates": {key: int(value) for key, value in mid15_candidates.items()},
                "mid_candidate_seconds": {key: float(value) for key, value in mid15_candidate_seconds.items()},
                "side_bytes": int(side_bytes),
                "side_method": side_method,
                "side_seconds": side_seconds,
                "side_candidates": {key: int(value) for key, value in side_candidates.items()},
                "side_candidate_seconds": {key: float(value) for key, value in side_candidate_seconds.items()},
            },
        },
        "timing": {
            **raw.get("timing", {}),
            **rle.get("timing", {}),
            **bitshuffle.get("timing", {}),
        },
        "combined_schemes": schemes,
    }


def print_progress(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def main() -> int:
    args = parse_args()
    if args.chunk_values <= 0:
        raise ValueError("--chunk-values must be positive.")

    manifest = load_json(args.manifest)
    blocks = selected_blocks(manifest, args.limit_blocks)
    count = total_selected_values(blocks, args.limit_values)
    if count <= 0:
        raise ValueError("No TUI values selected.")

    print_progress(f"[Info] Selected {len(blocks)} blocks, {count} float32 values.", args.quiet)
    print_progress("[Info] Pass 1: sign bitpack, RLE stats, mantissa whole/raw PathL streams.", args.quiet)
    raw = sign_pack_and_raw_mantissa_pass(manifest, blocks, args)

    print_progress("[Info] Pass 2: sign RLE compressed candidates.", args.quiet)
    rle = sign_rle_compression_pass(manifest, blocks, args, raw["sign_rle"]["length_dtype"], int(raw["sign_rle"]["run_count"]))

    bitshuffle: Dict[str, int] = {}
    if not args.skip_bitshuffle:
        mode = args.bitshuffle_mode
        if mode == "auto":
            mode = "temp" if count % 8 == 0 else "multipass"
        if mode == "temp":
            print_progress("[Info] Pass 3: PathL bitshuffle candidates via temporary bit-plane streams.", args.quiet)
            bitshuffle = bitshuffle_temp_pass(manifest, blocks, args)
        else:
            print_progress("[Info] Pass 3: PathL bitshuffle candidates. This scans TUI once per bit plane.", args.quiet)
            bitshuffle = bitshuffle_pass(manifest, blocks, args)

    summary = build_summary(raw, rle, bitshuffle, args)
    save_json(args.output_json, summary)
    print_progress(f"[OK] Saved exact TUI auxiliary-bit summary to {args.output_json}", args.quiet)

    sign = summary["sign"]
    mant = summary["mantissa"]
    print_progress(
        f"[OK] sign best: {sign['best_bytes']} bytes ({sign['best_method']}); "
        f"PathL-lite: {mant['pathL_lite_bytes']} bytes ({mant['pathL_lite_method']}); "
        f"whole-zstd: {mant['whole_zstd_bytes']} bytes",
        args.quiet,
    )
    for row in summary["combined_schemes"]:
        print_progress(
            "[OK] {scheme}: total={total:.0f} bytes, bps={bps:.6f}, ratio={ratio:.6f}x".format(
                scheme=row["scheme"],
                total=row["total_bytes"],
                bps=row["bits_per_value"],
                ratio=row["compression_ratio_vs_float32"],
            ),
            args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
