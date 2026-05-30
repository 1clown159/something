#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from .codec import Stage4GlobalDiagonalRangeCodec, Stage4RangeCodec, Stage4TileRangeCodec
    from .common import (
        DEFAULT_BIN_PATH,
        DEFAULT_META_PATH,
        DEFAULT_OUTPUT_DIR,
        ExperimentConfig,
        SplitConfig,
        VolumeData,
        VolumeShape,
        environment_report,
        infer_shape_from_file,
        resolve_device,
        save_json,
        to_serializable_config,
        write_sidecar_meta,
    )
    from .hybrid_codec import Stage4HybridROIzstdCodec
    from .stage4 import run_stage4_training
except ImportError:
    from codec import Stage4GlobalDiagonalRangeCodec, Stage4RangeCodec, Stage4TileRangeCodec
    from common import (
        DEFAULT_BIN_PATH,
        DEFAULT_META_PATH,
        DEFAULT_OUTPUT_DIR,
        ExperimentConfig,
        SplitConfig,
        VolumeData,
        VolumeShape,
        environment_report,
        infer_shape_from_file,
        resolve_device,
        save_json,
        to_serializable_config,
        write_sidecar_meta,
    )
    from hybrid_codec import Stage4HybridROIzstdCodec
    from stage4 import run_stage4_training


TRAINABLE_FEATURE_MODES = {"strict", "causal_edge", "diagonal_causal_edge"}
TRAINABLE_TARGET_MODES = {"raw", "residual"}


def build_split_config(args: argparse.Namespace) -> SplitConfig:
    return SplitConfig(
        profile_offset=args.profile_offset,
        train_profiles=args.train_profiles,
        val_profiles=args.val_profiles,
        test_profiles=args.test_profiles,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="20260416 Stage 4 diagonal causal range coding with optional valid-region rectangles")
    parser.add_argument(
        "--action",
        choices=["train", "roundtrip", "encode", "decode", "hybrid_encode", "hybrid_decode", "hybrid_roundtrip"],
        default="train",
    )
    parser.add_argument("--bin-path", default=DEFAULT_BIN_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--shape", nargs=3, type=int, default=[10, 600, 2001])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--codec-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--feature-mode",
        choices=["auto", "strict", "legacy", "causal_edge", "diagonal_causal_edge"],
        default="diagonal_causal_edge",
    )
    parser.add_argument("--target-mode", choices=["auto", "raw", "residual"], default="raw")
    parser.add_argument("--profile-timing", action="store_true", help="Collect detailed codec timing breakdown during encode/decode")
    parser.add_argument("--inference-batch", type=int, default=1, help="Mini-batch size for encode-side model inference; use 1 for verified lossless parity")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=0, help="Override stage4_train_samples when > 0")
    parser.add_argument("--val-samples", type=int, default=0, help="Override stage4_val_samples when > 0")
    parser.add_argument("--eval-trace-stride", type=int, default=0, help="Override eval_trace_stride when > 0")
    parser.add_argument("--eval-sample-stride", type=int, default=0, help="Override eval_sample_stride when > 0")
    parser.add_argument("--epochs-stage4", type=int, default=200)
    parser.add_argument("--base-channels", type=int, default=16, help="Base width for the narrowed 2D CNN backbone")
    parser.add_argument("--valid-region-mode", choices=["none", "auto_rect"], default="none")
    parser.add_argument(
        "--valid-region-min-nonzero-ratio",
        type=float,
        default=0.0,
        help="Trace is considered valid when its non-zero ratio exceeds this threshold. Keep 0.0 for lossless-safe edge trimming.",
    )
    parser.add_argument("--valid-region-margin-traces", type=int, default=0)
    parser.add_argument("--valid-region-group-size", type=int, default=1)
    parser.add_argument("--hybrid-zstd-level", type=int, default=9)
    parser.add_argument("--hybrid-zstd-threads", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--codec-layout", choices=["raster", "tile64", "global_diag"], default="raster")
    parser.add_argument("--tile-h", type=int, default=64)
    parser.add_argument("--tile-w", type=int, default=64)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--bitstream-path", default="")
    parser.add_argument("--decoded-output", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "full"], default="test")
    parser.add_argument("--profile-offset", type=int, default=0, help="Starting profile index for train/val/test split slicing")
    parser.add_argument("--train-profiles", type=int, default=8, help="Number of profiles in the train split")
    parser.add_argument("--val-profiles", type=int, default=1, help="Number of profiles in the val split")
    parser.add_argument("--test-profiles", type=int, default=1, help="Number of profiles in the test split")
    parser.add_argument("--limit-profiles", type=int, default=0)
    parser.add_argument("--limit-traces", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        codec_device=args.codec_device,
        feature_mode=("diagonal_causal_edge" if args.feature_mode == "auto" else args.feature_mode),
        target_mode=("raw" if args.target_mode == "auto" else args.target_mode),
        stage4_base_channels=args.base_channels,
        tile_shape=(args.tile_h, args.tile_w),
        valid_region_mode=args.valid_region_mode,
        valid_region_min_nonzero_ratio=args.valid_region_min_nonzero_ratio,
        valid_region_margin_traces=args.valid_region_margin_traces,
        valid_region_group_size=args.valid_region_group_size,
    )
    if args.smoke:
        cfg.stage4_train_samples = 4000
        cfg.stage4_val_samples = 1000
        cfg.epochs_stage4 = 1
        cfg.eval_trace_stride = 12
        cfg.eval_sample_stride = 32
    else:
        cfg.epochs_stage4 = args.epochs_stage4
        if args.train_samples > 0:
            cfg.stage4_train_samples = args.train_samples
        if args.val_samples > 0:
            cfg.stage4_val_samples = args.val_samples
        if args.eval_trace_stride > 0:
            cfg.eval_trace_stride = args.eval_trace_stride
        if args.eval_sample_stride > 0:
            cfg.eval_sample_stride = args.eval_sample_stride
    return cfg


def apply_limits(volume: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    p = args.limit_profiles if args.limit_profiles > 0 else volume.shape[0]
    t = args.limit_traces if args.limit_traces > 0 else volume.shape[1]
    s = args.limit_samples if args.limit_samples > 0 else volume.shape[2]
    return volume[:p, :t, :s]


def pick_volume(volume_data: VolumeData, split: str, args: argparse.Namespace) -> np.ndarray:
    if split == "full":
        volume = volume_data.exps
    else:
        volume = volume_data.get_split(split)
    return apply_limits(volume, args)


def default_checkpoint(output_dir: str) -> str:
    return os.path.join(output_dir, "stage4", "causal", "checkpoint.pt")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    shape = infer_shape_from_file(args.bin_path, VolumeShape(*args.shape))
    write_sidecar_meta(args.meta_path, args.bin_path, shape)
    split_config = build_split_config(args)
    volume_data = VolumeData(args.bin_path, shape, split_config=split_config)
    config = configure(args)
    train_device = resolve_device(args.device)

    if args.action == "train":
        if split_config.train_profiles <= 0:
            raise ValueError("Training requires --train-profiles > 0.")
        if config.feature_mode not in TRAINABLE_FEATURE_MODES:
            raise ValueError("Training supports --feature-mode strict, causal_edge, or diagonal_causal_edge only.")
        if config.target_mode not in TRAINABLE_TARGET_MODES:
            raise ValueError("Training supports --target-mode raw or residual only.")
        metrics = run_stage4_training(volume_data, config, train_device, args.output_dir)
        summary = {
            "environment": environment_report(),
            "device": train_device,
            "config": to_serializable_config(config),
            "volume_metadata": volume_data.to_metadata(),
            "stage4": metrics,
        }
        json_path = args.save_json or os.path.join(args.output_dir, "stage4_summary.json")
        save_json(json_path, summary)
        print("[OK] Saved training summary to", json_path)
        raise SystemExit(0)

    checkpoint_path = args.checkpoint_path or default_checkpoint(args.output_dir)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    is_hybrid = args.action.startswith("hybrid_")
    base_action = args.action[len("hybrid_") :] if is_hybrid else args.action
    if is_hybrid:
        if args.codec_layout != "raster":
            raise ValueError("Hybrid ROI+zstd prototype currently supports --codec-layout raster only.")
        codec = Stage4HybridROIzstdCodec(
            checkpoint_path=checkpoint_path,
            config=config,
            device=args.codec_device,
            feature_mode=args.feature_mode,
            target_mode=args.target_mode,
            profile_timing=args.profile_timing,
            inference_batch=args.inference_batch,
            zstd_level=args.hybrid_zstd_level,
            zstd_threads=args.hybrid_zstd_threads,
        )
    else:
        if args.codec_layout == "tile64":
            codec_cls = Stage4TileRangeCodec
        elif args.codec_layout == "global_diag":
            codec_cls = Stage4GlobalDiagonalRangeCodec
        else:
            codec_cls = Stage4RangeCodec
        codec_kwargs = {
            "checkpoint_path": checkpoint_path,
            "config": config,
            "device": args.codec_device,
            "feature_mode": args.feature_mode,
            "target_mode": args.target_mode,
            "profile_timing": args.profile_timing,
            "inference_batch": args.inference_batch,
        }
        if codec_cls is Stage4TileRangeCodec:
            codec_kwargs["tile_shape"] = config.tile_shape
        codec = codec_cls(**codec_kwargs)

    default_ext = ".s4hz" if is_hybrid else ".s4rc"
    action_stem = f"stage4_hybrid_{args.split}" if is_hybrid else f"stage4_{args.split}"

    if base_action == "encode":
        volume = pick_volume(volume_data, args.split, args)
        bitstream_path = args.bitstream_path or os.path.join(args.output_dir, action_stem + default_ext)
        metrics = codec.encode_exponents(volume, bitstream_path)
        json_path = args.save_json or os.path.join(args.output_dir, f"{action_stem}_encode.json")
        save_json(
            json_path,
            {
                "split": args.split,
                "volume_shape": list(volume.shape),
                "feature_mode": getattr(codec, "feature_mode", getattr(codec, "roi_codec").feature_mode if hasattr(codec, "roi_codec") else None),
                "target_mode": getattr(codec, "target_mode", getattr(codec, "roi_codec").target_mode if hasattr(codec, "roi_codec") else None),
                "codec_layout": args.codec_layout,
                "tile_shape": list(config.tile_shape),
                "is_hybrid": bool(is_hybrid),
                "hybrid_zstd_level": int(args.hybrid_zstd_level),
                "hybrid_zstd_threads": int(args.hybrid_zstd_threads),
                "profile_timing": bool(args.profile_timing),
                "inference_batch": int(args.inference_batch),
                "metrics": metrics,
            },
        )
        print("[OK] Saved bitstream to", bitstream_path)
        raise SystemExit(0)

    if base_action == "decode":
        bitstream_path = args.bitstream_path or os.path.join(args.output_dir, action_stem + default_ext)
        decoded, header = codec.decode_exponents(bitstream_path)
        decoded_output = args.decoded_output or os.path.join(args.output_dir, f"{action_stem}_decoded.npy")
        np.save(decoded_output, decoded)
        json_path = args.save_json or os.path.join(args.output_dir, f"{action_stem}_decode.json")
        save_json(
            json_path,
            {
                "header": header,
                "decoded_output": os.path.abspath(decoded_output),
                "codec_layout": args.codec_layout,
                "tile_shape": list(config.tile_shape),
                "is_hybrid": bool(is_hybrid),
                "profile_timing": bool(args.profile_timing),
                "inference_batch": int(args.inference_batch),
            },
        )
        print("[OK] Saved decoded exponents to", decoded_output)
        raise SystemExit(0)

    volume = pick_volume(volume_data, args.split, args)
    bitstream_path = args.bitstream_path or os.path.join(args.output_dir, action_stem + default_ext)
    result = codec.roundtrip(volume, bitstream_path)
    decoded, decode_header = codec.decode_exponents(bitstream_path)
    reference = np.asarray(volume, dtype=np.uint8)
    result["volume_shape"] = list(volume.shape)
    result["codec_layout"] = args.codec_layout
    result["tile_shape"] = list(config.tile_shape)
    result["is_hybrid"] = bool(is_hybrid)
    result["profile_timing"] = bool(args.profile_timing)
    result["inference_batch"] = int(args.inference_batch)
    result["target_mode"] = getattr(codec, "target_mode", getattr(codec, "roi_codec").target_mode if hasattr(codec, "roi_codec") else None)
    result["decode_header"] = decode_header
    result["max_abs_diff"] = int(np.max(np.abs(decoded.astype(np.int16) - reference.astype(np.int16))))
    json_path = args.save_json or os.path.join(args.output_dir, f"{action_stem}_roundtrip.json")
    save_json(json_path, result)
    print("[OK] Roundtrip", "passed" if result["ok"] else "failed")
    print("[OK] Saved roundtrip report to", json_path)
