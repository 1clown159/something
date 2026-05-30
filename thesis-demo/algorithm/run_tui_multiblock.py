#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from .codec import Stage4GlobalDiagonalRangeCodec, Stage4RangeCodec, Stage4TileRangeCodec
    from .common import (
        ExperimentConfig,
        checkpoint_path,
        extract_float_exponents,
        resolve_device,
        save_json,
        set_seed,
        to_serializable_config,
    )
    from .stage4 import (
        Small2DCNN,
        build_stage4_features,
        build_single_stage4_feature,
        build_loader as build_tensor_loader,
        feature_mode_to_in_channels,
        load_stage4_model,
        target_symbol_for_coord,
        train_classifier,
    )
    from .tui_blocks import build_manifest, ensure_manifest_blocks_extracted, load_json
except ImportError:
    from codec import Stage4GlobalDiagonalRangeCodec, Stage4RangeCodec, Stage4TileRangeCodec
    from common import (
        ExperimentConfig,
        checkpoint_path,
        extract_float_exponents,
        resolve_device,
        save_json,
        set_seed,
        to_serializable_config,
    )
    from stage4 import (
        Small2DCNN,
        build_stage4_features,
        build_single_stage4_feature,
        build_loader as build_tensor_loader,
        feature_mode_to_in_channels,
        load_stage4_model,
        target_symbol_for_coord,
        train_classifier,
    )
    from tui_blocks import build_manifest, ensure_manifest_blocks_extracted, load_json


DEFAULT_TUI_META = r"E:\code\thesis\experiments\dat\TUI_trace_major.dat.json"
DEFAULT_OUTPUT_DIR = r"E:\code\thesis\20260420\outputs_tui_multiblock"
DEFAULT_MANIFEST = r"E:\code\thesis\20260420\tui_blocks_manifest.json"
DEFAULT_BLOCK_DIR = r"E:\code\thesis\20260420\tui_blocks"

DEFAULT_VAL_RANGES = [(3898, 3912), (4075, 4089), (4273, 4287)]
DEFAULT_TEST_RANGES = [(3913, 3927), (4090, 4104), (4288, 4302)]
VALID_FEATURE_MODES = {"strict", "causal_edge", "diagonal_causal_edge"}
VALID_TARGET_MODES = {"raw", "residual"}


@dataclass(frozen=True)
class Segment:
    segment_id: str
    split: str
    block_id: str
    local_profile_start: int
    profiles: int
    subline_start: int
    subline_end: int
    shape: Tuple[int, int, int]
    dat_path: str
    block_shape: Tuple[int, int, int]

    @property
    def voxel_count(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])


class BlockStore:
    def __init__(self, manifest: Dict[str, Any]) -> None:
        self.blocks = {str(block["block_id"]): block for block in manifest["blocks"]}
        self.cache: Dict[str, np.ndarray] = {}

    def get(self, block_id: str) -> np.ndarray:
        cached = self.cache.get(block_id)
        if cached is not None:
            return cached
        block = self.blocks[block_id]
        dat_path = str(block["dat_path"])
        shape = tuple(int(v) for v in block["shape"])
        floats = np.memmap(dat_path, dtype=np.float32, mode="r", shape=shape)
        exps = extract_float_exponents(floats).reshape(shape)
        self.cache[block_id] = exps
        return exps


class MultiBlockPatchDataset(Dataset):
    def __init__(
        self,
        samples: np.ndarray,
        segments: Sequence[Segment],
        block_store: BlockStore,
        config: ExperimentConfig,
    ) -> None:
        self.samples = samples.astype(np.int64, copy=False)
        self.segments = list(segments)
        self.block_store = block_store
        self.patch_shape = config.patch_shape
        self.feature_mode = config.feature_mode
        self.target_mode = config.target_mode

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int):
        segment_index, local_p, t, s = self.samples[index].tolist()
        segment = self.segments[int(segment_index)]
        volume = self.block_store.get(segment.block_id)
        p = int(segment.local_profile_start + local_p)
        coord = (p, int(t), int(s))
        feature = build_single_stage4_feature(
            volume,
            coord,
            self.patch_shape,
            feature_mode=self.feature_mode,
            target_mode=self.target_mode,
        )[0]
        label = target_symbol_for_coord(volume, coord, self.target_mode)
        return feature, torch.tensor(label, dtype=torch.long)


def parse_ranges(items: Sequence[str]) -> List[Tuple[int, int]]:
    ranges = []
    for item in items:
        left, right = item.split("-", 1)
        start = int(left)
        end = int(right)
        if end < start:
            raise ValueError(f"Invalid range: {item}")
        ranges.append((start, end))
    return ranges


def in_any_range(value: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(start <= value <= end for start, end in ranges)


def split_for_subline(subline: int, split_mode: str, val_ranges: Sequence[Tuple[int, int]], test_ranges: Sequence[Tuple[int, int]]) -> str:
    if split_mode == "full_train":
        return "train"
    if in_any_range(subline, test_ranges):
        return "test"
    if in_any_range(subline, val_ranges):
        return "val"
    return "train"


def make_segments(
    manifest: Dict[str, Any],
    split_mode: str,
    val_ranges: Sequence[Tuple[int, int]],
    test_ranges: Sequence[Tuple[int, int]],
) -> List[Segment]:
    segments: List[Segment] = []
    for block in manifest["blocks"]:
        block_id = str(block["block_id"])
        block_shape = tuple(int(v) for v in block["shape"])
        start = int(block["subline_start"])
        profiles = int(block["profiles"])
        traces = int(block["traces_per_profile"])
        samples = int(block["samples_per_trace"])
        current_split = None
        current_start = 0
        current_count = 0

        def flush() -> None:
            nonlocal current_split, current_start, current_count
            if current_split is None or current_count <= 0:
                return
            subline_start = start + current_start
            subline_end = subline_start + current_count - 1
            segment_id = f"{block_id}_{current_split}_s{subline_start}_{subline_end}"
            segments.append(
                Segment(
                    segment_id=segment_id,
                    split=current_split,
                    block_id=block_id,
                    local_profile_start=current_start,
                    profiles=current_count,
                    subline_start=subline_start,
                    subline_end=subline_end,
                    shape=(current_count, traces, samples),
                    dat_path=str(block["dat_path"]),
                    block_shape=block_shape,
                )
            )

        for offset in range(profiles):
            subline = start + offset
            label = split_for_subline(subline, split_mode, val_ranges, test_ranges)
            if current_split is None:
                current_split = label
                current_start = offset
                current_count = 1
            elif label == current_split:
                current_count += 1
            else:
                flush()
                current_split = label
                current_start = offset
                current_count = 1
        flush()
    return segments


def allocate_samples(
    segments: Sequence[Segment],
    total_samples: int,
    min_samples_per_segment: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not segments or total_samples <= 0:
        return np.empty((0, 4), dtype=np.int64), []
    total_voxels = sum(segment.voxel_count for segment in segments)
    rng = np.random.default_rng(seed)
    all_samples = []
    allocation = []
    for segment_index, segment in enumerate(segments):
        weighted = int(round(total_samples * segment.voxel_count / max(total_voxels, 1)))
        count = max(int(min_samples_per_segment), weighted)
        count = min(count, segment.voxel_count)
        linear = rng.choice(segment.voxel_count, size=count, replace=False)
        profile_span = segment.shape[1] * segment.shape[2]
        p = linear // profile_span
        rem = linear % profile_span
        t = rem // segment.shape[2]
        s = rem % segment.shape[2]
        segment_ids = np.full(count, segment_index, dtype=np.int64)
        all_samples.append(np.stack([segment_ids, p, t, s], axis=1))
        allocation.append(
            {
                "segment_id": segment.segment_id,
                "split": segment.split,
                "shape": list(segment.shape),
                "voxel_count": segment.voxel_count,
                "sample_count": int(count),
            }
        )
    samples = np.concatenate(all_samples, axis=0) if all_samples else np.empty((0, 4), dtype=np.int64)
    rng.shuffle(samples)
    return samples, allocation


def build_loader(samples: np.ndarray, segments: Sequence[Segment], block_store: BlockStore, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    dataset = MultiBlockPatchDataset(samples, segments, block_store, config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def benchmark_classifier(model: torch.nn.Module, loader: DataLoader, device: str, total_voxels: int) -> Dict[str, Any]:
    metrics = evaluate_loader(model, loader, device=device, topk=(1, 5))
    evaluated = int(len(loader.dataset))
    avg_bits = None if evaluated == 0 else float(metrics["loss"] / math.log(2.0))
    return {
        "lossless_compatible": True,
        "evaluated_voxels": evaluated,
        "total_voxels": int(total_voxels),
        "average_nll_bits": avg_bits,
        "perplexity": None if avg_bits is None else float(2.0 ** avg_bits),
        "ideal_code_length_bits": 0.0 if avg_bits is None else float(avg_bits * evaluated),
        "proxy_size_bytes": None if avg_bits is None else int(math.ceil(avg_bits * max(total_voxels, 1) / 8.0)),
        "argmax_error_rate": None if evaluated == 0 else float(1.0 - metrics["accuracy_top1"]),
        "accuracy_top1": None if evaluated == 0 else float(metrics["accuracy_top1"]),
        "accuracy_top5": None if evaluated == 0 else float(metrics["accuracy_top5"]),
    }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str, topk: Tuple[int, ...]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_hits = {k: 0.0 for k in topk}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            total_loss += float(loss.item()) * inputs.size(0)
            total_items += int(inputs.size(0))
            max_k = max(topk)
            pred = logits.topk(max_k, dim=1).indices
            for k in topk:
                total_hits[k] += float((pred[:, :k] == targets[:, None]).any(dim=1).float().sum().item())
    metrics = {"loss": total_loss / max(total_items, 1)}
    for k in topk:
        metrics[f"accuracy_top{k}"] = total_hits[k] / max(total_items, 1)
    return metrics


def materialize_samples(
    samples: np.ndarray,
    segments: Sequence[Segment],
    block_store: BlockStore,
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(samples.shape[0]) == 0:
        in_channels = feature_mode_to_in_channels(config.feature_mode, config.target_mode)
        empty_x = torch.empty((0, in_channels, *config.patch_shape), dtype=torch.float32)
        empty_y = torch.empty((0,), dtype=torch.long)
        return empty_x, empty_y

    feature_chunks = []
    label_chunks = []
    for segment_index, segment in enumerate(segments):
        mask = samples[:, 0] == segment_index
        segment_samples = samples[mask]
        if int(segment_samples.shape[0]) == 0:
            continue
        volume = block_store.get(segment.block_id)
        coords = segment_samples[:, 1:4].copy()
        coords[:, 0] += int(segment.local_profile_start)
        features, labels = build_stage4_features(
            volume,
            coords,
            config.patch_shape,
            feature_mode=config.feature_mode,
            target_mode=config.target_mode,
        )
        feature_chunks.append(features)
        label_chunks.append(labels)
    if not feature_chunks:
        in_channels = feature_mode_to_in_channels(config.feature_mode, config.target_mode)
        return (
            torch.empty((0, in_channels, *config.patch_shape), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def configure(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        codec_device=args.codec_device,
        feature_mode=args.feature_mode,
        target_mode=args.target_mode,
        stage4_base_channels=args.base_channels,
        tile_shape=(args.tile_h, args.tile_w),
        valid_region_mode="none",
    )
    cfg.epochs_stage4 = args.epochs_stage4
    cfg.stage4_train_samples = args.train_samples
    cfg.stage4_val_samples = args.val_samples
    return cfg


def split_summary(segments: Sequence[Segment]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        selected = [segment for segment in segments if segment.split == split]
        payload[split] = {
            "segment_count": len(selected),
            "profiles": int(sum(segment.profiles for segment in selected)),
            "voxels": int(sum(segment.voxel_count for segment in selected)),
            "segments": [asdict(segment) for segment in selected],
        }
    return payload


def train_multiblock(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_json(args.manifest)
    ensure_manifest_blocks_extracted(manifest)
    config = configure(args)
    if config.feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode for training: {config.feature_mode}")
    if config.target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {config.target_mode}")
    set_seed(config.seed)
    val_ranges = parse_ranges(args.val_range) if args.val_range else DEFAULT_VAL_RANGES
    test_ranges = parse_ranges(args.test_range) if args.test_range else DEFAULT_TEST_RANGES
    segments = make_segments(manifest, args.split_mode, val_ranges, test_ranges)
    train_segments = [segment for segment in segments if segment.split == "train"]
    val_segments = [segment for segment in segments if segment.split == "val"]
    benchmark_segments = [segment for segment in segments if segment.split == args.benchmark_split]
    if not train_segments:
        raise ValueError("No train segments were selected.")

    train_samples, train_allocation = allocate_samples(
        train_segments,
        config.stage4_train_samples,
        args.min_samples_per_block,
        config.seed + 21,
    )
    val_samples, val_allocation = allocate_samples(
        val_segments,
        config.stage4_val_samples,
        0,
        config.seed + 22,
    )
    block_store = BlockStore(manifest)
    materialized_shapes = None
    if args.materialize_features:
        print("[Info] Materializing train features...")
        train_x, train_y = materialize_samples(train_samples, train_segments, block_store, config)
        print("[Info] Materializing val features...")
        val_x, val_y = materialize_samples(val_samples, val_segments, block_store, config)
        train_loader = build_tensor_loader(train_x, train_y, config, shuffle=True)
        val_loader = build_tensor_loader(val_x, val_y, config, shuffle=False)
        materialized_shapes = {
            "train_x": list(train_x.shape),
            "train_y": list(train_y.shape),
            "val_x": list(val_x.shape),
            "val_y": list(val_y.shape),
            "train_feature_bytes": int(train_x.numel() * train_x.element_size()),
            "val_feature_bytes": int(val_x.numel() * val_x.element_size()),
        }
    else:
        train_loader = build_loader(train_samples, train_segments, block_store, config, shuffle=True)
        val_loader = build_loader(val_samples, val_segments, block_store, config, shuffle=False)

    in_channels = feature_mode_to_in_channels(config.feature_mode, config.target_mode)
    base_channels = max(1, int(config.stage4_base_channels))
    model = Small2DCNN(out_dim=256, base_channels=base_channels, in_channels=in_channels)
    device = resolve_device(args.device)
    training = train_classifier(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=config.epochs_stage4,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        topk=(1, 5),
    )
    benchmark = None
    benchmark_allocation = []
    if args.benchmark_samples > 0 and benchmark_segments:
        benchmark_samples, benchmark_allocation = allocate_samples(
            benchmark_segments,
            args.benchmark_samples,
            0,
            config.seed + 23,
        )
        benchmark_loader = build_loader(benchmark_samples, benchmark_segments, block_store, config, shuffle=False)
        benchmark = benchmark_classifier(
            model,
            benchmark_loader,
            device=device,
            total_voxels=sum(segment.voxel_count for segment in benchmark_segments),
        )
        benchmark["split"] = args.benchmark_split
        benchmark["target_mode"] = config.target_mode
        benchmark["feature_mode"] = config.feature_mode
        benchmark["benchmark_sample_count"] = int(benchmark_samples.shape[0])
    ckpt_path = checkpoint_path(args.output_dir, "stage4", "causal")
    torch.save(
        {
            "stage": "stage4",
            "mode": "causal",
            "feature_mode": config.feature_mode,
            "target_mode": config.target_mode,
            "model_family": "2d_cnn",
            "predictor": "loco_i_2d" if config.target_mode == "residual" else None,
            "config": {
                **asdict(config),
                "feature_mode": config.feature_mode,
                "target_mode": config.target_mode,
                "in_channels": in_channels,
                "base_channels": base_channels,
                "training_source": "tui_multiblock",
                "split_mode": args.split_mode,
            },
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    metrics = {
        "stage": "stage4",
        "mode": "tui_multiblock",
        "device": device,
        "manifest": os.path.abspath(args.manifest),
        "split_mode": args.split_mode,
        "val_ranges": val_ranges,
        "test_ranges": test_ranges,
        "config": to_serializable_config(config),
        "split_summary": split_summary(segments),
        "train_sample_count": int(train_samples.shape[0]),
        "val_sample_count": int(val_samples.shape[0]),
        "materialize_features": bool(args.materialize_features),
        "materialized_shapes": materialized_shapes,
        "train_allocation": train_allocation,
        "val_allocation": val_allocation,
        "training": training,
        "benchmark": benchmark,
        "benchmark_allocation": benchmark_allocation,
        "checkpoint_path": ckpt_path,
    }
    json_path = args.save_json or os.path.join(args.output_dir, "tui_multiblock_train_summary.json")
    save_json(json_path, metrics)
    print("[OK] Saved training summary to", json_path)
    print("[OK] Saved checkpoint to", ckpt_path)
    return metrics


def benchmark_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_json(args.manifest)
    ensure_manifest_blocks_extracted(manifest)
    config = configure(args)
    checkpoint = args.checkpoint_path or checkpoint_path(args.output_dir, "stage4", "causal")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    val_ranges = parse_ranges(args.val_range) if args.val_range else DEFAULT_VAL_RANGES
    test_ranges = parse_ranges(args.test_range) if args.test_range else DEFAULT_TEST_RANGES
    segments = make_segments(manifest, args.split_mode, val_ranges, test_ranges)
    benchmark_segments = [segment for segment in segments if segment.split == args.benchmark_split]
    if not benchmark_segments:
        raise ValueError(f"No segments selected for benchmark split={args.benchmark_split}")
    samples, allocation = allocate_samples(
        benchmark_segments,
        args.benchmark_samples,
        0,
        config.seed + 23,
    )
    device = resolve_device(args.device)
    model, checkpoint_payload = load_stage4_model(checkpoint, device=device)
    ckpt_feature_mode = checkpoint_payload.get("feature_mode") or checkpoint_payload.get("config", {}).get("feature_mode")
    ckpt_target_mode = checkpoint_payload.get("target_mode") or checkpoint_payload.get("config", {}).get("target_mode")
    if ckpt_feature_mode:
        config.feature_mode = str(ckpt_feature_mode)
    if ckpt_target_mode:
        config.target_mode = str(ckpt_target_mode)
    block_store = BlockStore(manifest)
    loader = build_loader(samples, benchmark_segments, block_store, config, shuffle=False)
    benchmark = benchmark_classifier(
        model,
        loader,
        device=device,
        total_voxels=sum(segment.voxel_count for segment in benchmark_segments),
    )
    benchmark.update(
        {
            "split": args.benchmark_split,
            "feature_mode": config.feature_mode,
            "target_mode": config.target_mode,
            "benchmark_sample_count": int(samples.shape[0]),
            "checkpoint_path": os.path.abspath(checkpoint),
            "allocation": allocation,
        }
    )
    json_path = args.save_json or os.path.join(args.output_dir, f"tui_{args.benchmark_split}_benchmark.json")
    save_json(json_path, benchmark)
    print("[OK] Saved benchmark summary to", json_path)
    return benchmark


def select_codec_class(layout: str):
    if layout == "tile64":
        return Stage4TileRangeCodec
    if layout == "global_diag":
        return Stage4GlobalDiagonalRangeCodec
    return Stage4RangeCodec


def segment_volume(block_store: BlockStore, segment: Segment) -> np.ndarray:
    volume = block_store.get(segment.block_id)
    start = segment.local_profile_start
    end = start + segment.profiles
    return volume[start:end]


def apply_volume_limits(volume: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    profiles = args.limit_profiles if args.limit_profiles > 0 else volume.shape[0]
    traces = args.limit_traces if args.limit_traces > 0 else volume.shape[1]
    samples = args.limit_samples if args.limit_samples > 0 else volume.shape[2]
    return volume[:profiles, :traces, :samples]


def complete_block_segments(manifest: Dict[str, Any]) -> List[Segment]:
    segments = []
    for block in manifest["blocks"]:
        shape = tuple(int(v) for v in block["shape"])
        segments.append(
            Segment(
                segment_id=f"{block['block_id']}_all",
                split="all",
                block_id=str(block["block_id"]),
                local_profile_start=0,
                profiles=shape[0],
                subline_start=int(block["subline_start"]),
                subline_end=int(block["subline_end"]),
                shape=shape,
                dat_path=str(block["dat_path"]),
                block_shape=shape,
            )
        )
    return segments


def roundtrip_segments(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_json(args.manifest)
    ensure_manifest_blocks_extracted(manifest)
    config = configure(args)
    checkpoint = args.checkpoint_path or checkpoint_path(args.output_dir, "stage4", "causal")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.codec_layout in {"tile64", "global_diag"} and config.valid_region_mode != "none":
        raise ValueError("tile64/global_diag require valid_region_mode none.")

    val_ranges = parse_ranges(args.val_range) if args.val_range else DEFAULT_VAL_RANGES
    test_ranges = parse_ranges(args.test_range) if args.test_range else DEFAULT_TEST_RANGES
    if args.roundtrip_split == "all":
        segments = complete_block_segments(manifest)
    else:
        all_segments = make_segments(manifest, args.split_mode, val_ranges, test_ranges)
        segments = [segment for segment in all_segments if segment.split == args.roundtrip_split]
    if args.limit_segments > 0:
        segments = segments[: args.limit_segments]
    if not segments:
        raise ValueError(f"No segments selected for split={args.roundtrip_split}")

    codec_cls = select_codec_class(args.codec_layout)
    codec_kwargs = {
        "checkpoint_path": checkpoint,
        "config": config,
        "device": args.codec_device,
        "feature_mode": args.feature_mode_for_codec,
        "target_mode": args.target_mode_for_codec,
        "profile_timing": args.profile_timing,
        "inference_batch": args.inference_batch,
    }
    if codec_cls is Stage4TileRangeCodec:
        codec_kwargs["tile_shape"] = config.tile_shape
    codec = codec_cls(**codec_kwargs)
    block_store = BlockStore(manifest)
    bitstream_dir = Path(args.output_dir) / f"tui_{args.roundtrip_split}_{args.codec_layout}_bitstreams"
    bitstream_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total_bytes = 0
    total_voxels = 0
    ok_all = True
    for index, segment in enumerate(segments):
        full_volume = segment_volume(block_store, segment)
        volume = apply_volume_limits(full_volume, args)
        bitstream_path = bitstream_dir / f"{index:04d}_{segment.segment_id}.s4rc"
        result = codec.roundtrip(volume, str(bitstream_path))
        ok_all = ok_all and bool(result["ok"])
        encoded_bytes = int(result["encode"]["total_bytes"])
        voxels = int(np.prod(volume.shape))
        total_bytes += encoded_bytes
        total_voxels += voxels
        results.append(
            {
                "segment": asdict(segment),
                "ok": bool(result["ok"]),
                "shape": list(volume.shape),
                "source_segment_shape": list(full_volume.shape),
                "encoded_bytes": encoded_bytes,
                "bits_per_voxel": 8.0 * encoded_bytes / max(voxels, 1),
                "encode": result["encode"],
                "header": result["header"],
            }
        )
        print(f"[OK] {index + 1}/{len(segments)} {segment.segment_id}: ok={result['ok']}")
    summary = {
        "manifest": os.path.abspath(args.manifest),
        "checkpoint_path": os.path.abspath(checkpoint),
        "split_mode": args.split_mode,
        "roundtrip_split": args.roundtrip_split,
        "codec_layout": args.codec_layout,
        "limits": {
            "limit_profiles": int(args.limit_profiles),
            "limit_traces": int(args.limit_traces),
            "limit_samples": int(args.limit_samples),
        },
        "segment_count": len(results),
        "ok": bool(ok_all),
        "total_encoded_bytes": int(total_bytes),
        "total_voxels": int(total_voxels),
        "bits_per_voxel": 8.0 * total_bytes / max(total_voxels, 1),
        "results": results,
    }
    json_path = args.save_json or os.path.join(args.output_dir, f"tui_{args.roundtrip_split}_{args.codec_layout}_roundtrip.json")
    save_json(json_path, summary)
    print("[OK] Saved roundtrip summary to", json_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TUI variable-size block training and block-wise Stage4 range coding.")
    parser.add_argument("--action", choices=["prepare", "train", "roundtrip", "benchmark"], required=True)
    parser.add_argument("--tui-meta", default=DEFAULT_TUI_META)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--block-dir", default=DEFAULT_BLOCK_DIR)
    parser.add_argument("--extract-blocks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-json", default="")
    parser.add_argument("--split-mode", choices=["heldout", "full_train"], default="heldout")
    parser.add_argument("--val-range", action="append", default=[], help="Inclusive subline range like 3898-3912. Can repeat.")
    parser.add_argument("--test-range", action="append", default=[], help="Inclusive subline range like 3913-3927. Can repeat.")
    parser.add_argument("--roundtrip-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--benchmark-split", choices=["val", "test"], default="test")
    parser.add_argument("--limit-segments", type=int, default=0)
    parser.add_argument("--limit-profiles", type=int, default=0)
    parser.add_argument("--limit-traces", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--codec-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--feature-mode", choices=["strict", "causal_edge", "diagonal_causal_edge"], default="diagonal_causal_edge")
    parser.add_argument("--target-mode", choices=["raw", "residual"], default="residual")
    parser.add_argument("--feature-mode-for-codec", choices=["auto", "strict", "causal_edge", "diagonal_causal_edge"], default="auto")
    parser.add_argument("--target-mode-for-codec", choices=["auto", "raw", "residual"], default="auto")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=500000)
    parser.add_argument("--val-samples", type=int, default=50000)
    parser.add_argument("--benchmark-samples", type=int, default=50000)
    parser.add_argument("--min-samples-per-block", type=int, default=5000)
    parser.add_argument("--materialize-features", action="store_true")
    parser.add_argument("--epochs-stage4", type=int, default=120)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--codec-layout", choices=["raster", "tile64", "global_diag"], default="raster")
    parser.add_argument("--tile-h", type=int, default=64)
    parser.add_argument("--tile-w", type=int, default=64)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.action == "prepare":
        manifest = build_manifest(
            source_meta_path=args.tui_meta,
            output_path=args.manifest,
            block_dir=args.block_dir,
            extract=args.extract_blocks,
            overwrite=args.overwrite,
        )
        print("[OK] Saved manifest to", args.manifest)
        print("[OK] Blocks:", manifest["block_count"])
        print("[OK] Values:", manifest["value_count"])
        return 0
    if args.action == "train":
        train_multiblock(args)
        return 0
    if args.action == "roundtrip":
        roundtrip_segments(args)
        return 0
    if args.action == "benchmark":
        benchmark_checkpoint(args)
        return 0
    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
