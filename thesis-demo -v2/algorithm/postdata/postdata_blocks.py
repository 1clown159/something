#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


FLOAT32_BYTES = 4


@dataclass(frozen=True)
class PostDataBlock:
    block_id: str
    profile_start: int
    profile_end: int
    trace_start: int
    trace_end: int
    profiles: int
    traces_per_profile: int
    samples_per_trace: int
    trace_offset: int
    trace_count: int
    value_offset: int
    value_count: int
    byte_offset: int
    byte_count: int
    dat_path: str

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


def _positive_int(meta: Dict[str, Any], key: str) -> int:
    value = int(meta[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _make_blocks(
    dat_path: str,
    profile_count: int,
    traces_per_profile: int,
    samples_per_trace: int,
    profiles_per_block: int,
) -> List[PostDataBlock]:
    block_span = profile_count if profiles_per_block <= 0 else int(profiles_per_block)
    blocks: List[PostDataBlock] = []
    for profile_start in range(0, profile_count, block_span):
        profiles = min(block_span, profile_count - profile_start)
        profile_end = profile_start + profiles - 1
        trace_offset = profile_start * traces_per_profile
        trace_count = profiles * traces_per_profile
        value_offset = trace_offset * samples_per_trace
        value_count = trace_count * samples_per_trace
        byte_offset = value_offset * FLOAT32_BYTES
        byte_count = value_count * FLOAT32_BYTES
        block_id = f"b{len(blocks):04d}_p{profile_start:04d}_{profile_end:04d}_t0000_{traces_per_profile - 1:04d}"
        blocks.append(
            PostDataBlock(
                block_id=block_id,
                profile_start=profile_start,
                profile_end=profile_end,
                trace_start=0,
                trace_end=traces_per_profile - 1,
                profiles=profiles,
                traces_per_profile=traces_per_profile,
                samples_per_trace=samples_per_trace,
                trace_offset=trace_offset,
                trace_count=trace_count,
                value_offset=value_offset,
                value_count=value_count,
                byte_offset=byte_offset,
                byte_count=byte_count,
                dat_path=dat_path,
            )
        )
    return blocks


def build_manifest(
    source_meta_path: str,
    output_path: str,
    profiles_per_block: int = 0,
) -> Dict[str, Any]:
    meta_path = Path(source_meta_path).expanduser().resolve()
    meta = load_json(meta_path)
    source_dat_path = str(Path(meta["dat_path"]).expanduser().resolve())
    source_path = Path(source_dat_path)
    if not source_path.exists():
        raise FileNotFoundError(f"PostData source dat not found: {source_path}")

    profile_count = _positive_int(meta, "profile_count")
    traces_per_profile = _positive_int(meta, "traces_per_profile")
    samples_per_trace = _positive_int(meta, "samples_per_trace")
    trace_count = profile_count * traces_per_profile
    value_count = trace_count * samples_per_trace
    byte_count = value_count * FLOAT32_BYTES

    expected_trace_count = int(meta.get("trace_count", trace_count))
    expected_total_samples = int(meta.get("total_samples", value_count))
    actual_bytes = int(source_path.stat().st_size)
    if expected_trace_count != trace_count:
        raise ValueError(f"Trace count mismatch: metadata={expected_trace_count}, derived={trace_count}")
    if expected_total_samples != value_count:
        raise ValueError(f"Value count mismatch: metadata={expected_total_samples}, derived={value_count}")
    if actual_bytes != byte_count:
        raise ValueError(f"Byte count mismatch: file={actual_bytes}, derived={byte_count}")

    blocks = _make_blocks(
        dat_path=source_dat_path,
        profile_count=profile_count,
        traces_per_profile=traces_per_profile,
        samples_per_trace=samples_per_trace,
        profiles_per_block=profiles_per_block,
    )
    payload = {
        "schema": "postdata_regular_blocks_v1",
        "source_meta_path": str(meta_path),
        "source_dat_path": source_dat_path,
        "dtype": "float32",
        "storage_order": str(meta.get("storage_order", "trace_major_row_contiguous")),
        "profile_source": str(meta.get("profile_source", "header_inline")),
        "profile_count": profile_count,
        "traces_per_profile": traces_per_profile,
        "samples_per_trace": samples_per_trace,
        "trace_count": trace_count,
        "value_count": value_count,
        "byte_count": byte_count,
        "block_count": len(blocks),
        "profiles_per_block": int(profiles_per_block),
        "default_split": {
            "train_profiles": [0, 279],
            "val_profiles": [280, 314],
            "test_profiles": [315, 349],
        },
        "assumption": "PostData is a row-contiguous regular float32 volume ordered as profiles x traces x samples.",
        "blocks": [block.to_manifest_entry() for block in blocks],
    }
    save_json(output_path, payload)
    return payload


def manifest_block_by_id(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(block["block_id"]): block for block in manifest.get("blocks", [])}


def ensure_manifest_blocks_available(manifest: Dict[str, Any]) -> None:
    missing = []
    invalid = []
    for block in manifest.get("blocks", []):
        dat_path = block.get("dat_path") or manifest.get("source_dat_path")
        if not dat_path or not Path(dat_path).exists():
            missing.append(str(block.get("block_id")))
            continue
        need = int(block.get("byte_offset", 0)) + int(block["byte_count"])
        if Path(dat_path).stat().st_size < need:
            invalid.append(str(block.get("block_id")))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Manifest has blocks with missing dat files: {preview}")
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(f"Manifest has blocks whose byte range exceeds dat file size: {preview}")
