#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


FLOAT32_BYTES = 4


@dataclass(frozen=True)
class TuiRow:
    subline: int
    xline_start: int
    xline_end: int
    trace_count: int
    trace_offset: int


@dataclass(frozen=True)
class TuiBlock:
    block_id: str
    subline_start: int
    subline_end: int
    xline_start: int
    xline_end: int
    profiles: int
    traces_per_profile: int
    samples_per_trace: int
    trace_offset: int
    trace_count: int
    value_offset: int
    value_count: int
    byte_offset: int
    byte_count: int
    dat_path: str | None = None

    @property
    def shape(self) -> List[int]:
        return [self.profiles, self.traces_per_profile, self.samples_per_trace]

    def to_manifest_entry(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = self.shape
        return payload


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def flatten_tui_rows(meta: Dict[str, Any]) -> List[TuiRow]:
    rows: List[TuiRow] = []
    trace_offset = 0
    for file_entry in meta.get("files", []):
        for item in file_entry.get("subline_ranges", []):
            count = int(item["trace_count"])
            rows.append(
                TuiRow(
                    subline=int(item["subline"]),
                    xline_start=int(item["xline_start"]),
                    xline_end=int(item["xline_end"]),
                    trace_count=count,
                    trace_offset=trace_offset,
                )
            )
            trace_offset += count
    expected = int(meta.get("trace_count", trace_offset))
    if trace_offset != expected:
        raise ValueError(f"Trace count mismatch: rows={trace_offset}, metadata={expected}")
    return rows


def group_regular_blocks(rows: Iterable[TuiRow], samples_per_trace: int) -> List[TuiBlock]:
    sorted_rows = sorted(rows, key=lambda item: item.trace_offset)
    blocks: List[TuiBlock] = []
    current: List[TuiRow] = []

    def flush() -> None:
        if not current:
            return
        first = current[0]
        last = current[-1]
        profiles = len(current)
        traces = first.trace_count
        trace_count = profiles * traces
        value_count = trace_count * samples_per_trace
        block_id = f"b{len(blocks):04d}_s{first.subline}_{last.subline}_x{first.xline_start}_{first.xline_end}"
        blocks.append(
            TuiBlock(
                block_id=block_id,
                subline_start=first.subline,
                subline_end=last.subline,
                xline_start=first.xline_start,
                xline_end=first.xline_end,
                profiles=profiles,
                traces_per_profile=traces,
                samples_per_trace=samples_per_trace,
                trace_offset=first.trace_offset,
                trace_count=trace_count,
                value_offset=first.trace_offset * samples_per_trace,
                value_count=value_count,
                byte_offset=first.trace_offset * samples_per_trace * FLOAT32_BYTES,
                byte_count=value_count * FLOAT32_BYTES,
            )
        )

    for row in sorted_rows:
        if not current:
            current = [row]
            continue
        prev = current[-1]
        same_shape = (
            row.subline == prev.subline + 1
            and row.xline_start == prev.xline_start
            and row.xline_end == prev.xline_end
            and row.trace_count == prev.trace_count
            and row.trace_offset == prev.trace_offset + prev.trace_count
        )
        if same_shape:
            current.append(row)
        else:
            flush()
            current = [row]
    flush()
    return blocks


def copy_byte_range(source_path: Path, dest_path: Path, byte_offset: int, byte_count: int, overwrite: bool) -> None:
    if dest_path.exists() and not overwrite:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as src, dest_path.open("wb") as dst:
        src.seek(byte_offset)
        remaining = byte_count
        while remaining > 0:
            chunk = src.read(min(64 * 1024 * 1024, remaining))
            if not chunk:
                raise IOError(f"Unexpected EOF while copying {source_path}")
            dst.write(chunk)
            remaining -= len(chunk)


def extract_blocks(source_dat_path: str, blocks: List[TuiBlock], block_dir: str, overwrite: bool) -> List[TuiBlock]:
    source = Path(source_dat_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"TUI source dat not found: {source}")
    extracted: List[TuiBlock] = []
    for block in blocks:
        out_path = Path(block_dir).expanduser().resolve() / f"{block.block_id}.dat"
        copy_byte_range(source, out_path, block.byte_offset, block.byte_count, overwrite=overwrite)
        if out_path.stat().st_size != block.byte_count:
            raise ValueError(f"Block size mismatch for {block.block_id}: {out_path}")
        extracted.append(TuiBlock(**{**asdict(block), "dat_path": str(out_path)}))
    return extracted


def build_manifest(
    source_meta_path: str,
    output_path: str,
    block_dir: str | None = None,
    extract: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    meta_path = Path(source_meta_path).expanduser().resolve()
    meta = load_json(meta_path)
    samples_per_trace = int(meta["samples_per_trace"])
    source_dat_path = str(Path(meta["dat_path"]).expanduser().resolve())
    rows = flatten_tui_rows(meta)
    blocks = group_regular_blocks(rows, samples_per_trace=samples_per_trace)
    if extract:
        if block_dir is None:
            raise ValueError("block_dir is required when extract=True")
        blocks = extract_blocks(source_dat_path, blocks, block_dir, overwrite=overwrite)

    total_traces = sum(block.trace_count for block in blocks)
    total_values = sum(block.value_count for block in blocks)
    payload = {
        "schema": "tui_regular_blocks_v1",
        "source_meta_path": str(meta_path),
        "source_dat_path": source_dat_path,
        "dtype": "float32",
        "storage_order": "block_trace_major",
        "samples_per_trace": samples_per_trace,
        "trace_count": total_traces,
        "value_count": total_values,
        "byte_count": total_values * FLOAT32_BYTES,
        "block_count": len(blocks),
        "extracted": bool(extract),
        "block_dir": None if block_dir is None else str(Path(block_dir).expanduser().resolve()),
        "assumption": "Source dat rows follow metadata subline_ranges order; original TUI converter writes trace-major existing traces only.",
        "blocks": [block.to_manifest_entry() for block in blocks],
    }
    if int(meta.get("trace_count", total_traces)) != total_traces:
        raise ValueError("Manifest trace total does not match TUI metadata.")
    if int(meta.get("total_values", total_values)) != total_values:
        raise ValueError("Manifest value total does not match TUI metadata.")
    save_json(output_path, payload)
    return payload


def manifest_block_by_id(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(block["block_id"]): block for block in manifest.get("blocks", [])}


def ensure_manifest_blocks_extracted(manifest: Dict[str, Any]) -> None:
    missing = []
    for block in manifest.get("blocks", []):
        dat_path = block.get("dat_path")
        if not dat_path or not Path(dat_path).exists():
            missing.append(str(block.get("block_id")))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Manifest has blocks without extracted dat files: {preview}")


def copy_manifest_file(source: str, dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
