#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .common import (
        ExperimentConfig,
        VolumeData,
        checkpoint_path,
        choose_random_indices,
        lexicographic_causal_mask_2d,
        make_regular_grid_indices,
        metrics_path,
        normalize_uint8_patch,
        save_json,
        set_seed,
    )
    from .roi import (
        choose_random_indices_in_rectangles,
        detect_profile_rectangles,
        make_regular_grid_indices_in_rectangles,
        rectangles_to_metadata,
        region_enabled,
    )
except ImportError:
    from common import (
        ExperimentConfig,
        VolumeData,
        checkpoint_path,
        choose_random_indices,
        lexicographic_causal_mask_2d,
        make_regular_grid_indices,
        metrics_path,
        normalize_uint8_patch,
        save_json,
        set_seed,
    )
    from roi import (
        choose_random_indices_in_rectangles,
        detect_profile_rectangles,
        make_regular_grid_indices_in_rectangles,
        rectangles_to_metadata,
        region_enabled,
    )


VALID_FEATURE_MODES = {"strict", "legacy", "causal_edge", "diagonal_causal_edge"}
VALID_TARGET_MODES = {"raw", "residual"}
CAUSAL_EDGE_FEATURE_MODES = {"causal_edge", "diagonal_causal_edge"}
_CAUSAL_EDGE_PLAN_CACHE: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}


class Small2DCNN(nn.Module):
    def __init__(self, out_dim: int = 256, base_channels: int = 16, in_channels: int = 2) -> None:
        super().__init__()
        hidden_channels = base_channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden_channels, out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.net(inputs).flatten(1)
        return self.head(hidden)


def predictor_loco_i_2d(plane: np.ndarray, t: int, s: int) -> int:
    if t <= 0 and s <= 0:
        return 0
    if t <= 0:
        return int(plane[t, s - 1])
    if s <= 0:
        return int(plane[t - 1, s])
    left = int(plane[t, s - 1])
    up = int(plane[t - 1, s])
    up_left = int(plane[t - 1, s - 1])
    return int(np.clip(left + up - up_left, 0, 255))


def predictor_for_coord(volume: np.ndarray, coord: Tuple[int, int, int]) -> int:
    p, t, s = coord
    return predictor_loco_i_2d(volume[p], t, s)


def residual_symbol(exp_value: int, pred_value: int) -> int:
    return int((int(exp_value) - int(pred_value)) & 0xFF)


def reconstruct_exp_from_symbol(symbol: int, pred_value: int, target_mode: str) -> int:
    if target_mode == "raw":
        return int(symbol)
    if target_mode == "residual":
        return int((int(pred_value) + int(symbol)) & 0xFF)
    raise ValueError(f"Unsupported target mode: {target_mode}")


def target_symbol_for_coord(volume: np.ndarray, coord: Tuple[int, int, int], target_mode: str) -> int:
    p, t, s = coord
    exp_value = int(volume[p, t, s])
    if target_mode == "raw":
        return exp_value
    if target_mode == "residual":
        pred = predictor_for_coord(volume, coord)
        return residual_symbol(exp_value, pred)
    raise ValueError(f"Unsupported target mode: {target_mode}")


def feature_mode_to_in_channels(feature_mode: str, target_mode: str) -> int:
    if feature_mode in CAUSAL_EDGE_FEATURE_MODES:
        return 6 if target_mode == "residual" else 4
    if feature_mode in {"strict", "legacy"}:
        if target_mode != "raw":
            raise ValueError("Residual target mode is only supported with causal edge feature modes.")
        return 2
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def feature_mode_to_pad_mode(feature_mode: str) -> str:
    feature_mode = feature_mode.lower()
    if feature_mode == "legacy":
        return "edge"
    if feature_mode == "strict":
        return "constant"
    raise ValueError(f"Unsupported pad mode for feature mode: {feature_mode}")


def _pad_plane(plane: np.ndarray, patch_shape: Tuple[int, int], feature_mode: str) -> np.ndarray:
    t_half, s_half = (dim // 2 for dim in patch_shape)
    pad_mode = feature_mode_to_pad_mode(feature_mode)
    if pad_mode == "edge":
        return np.pad(plane, ((t_half, t_half), (s_half, s_half)), mode="edge")
    return np.pad(plane, ((t_half, t_half), (s_half, s_half)), mode="constant", constant_values=0)


def _get_causal_edge_plan(patch_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    plan = _CAUSAL_EDGE_PLAN_CACHE.get(patch_shape)
    if plan is not None:
        return plan
    t_half, s_half = (dim // 2 for dim in patch_shape)
    jj, kk = np.indices(patch_shape, dtype=np.int32)
    dt = (jj - t_half).reshape(-1)
    ds = (kk - s_half).reshape(-1)
    plan = {"dt": dt, "ds": ds}
    _CAUSAL_EDGE_PLAN_CACHE[patch_shape] = plan
    return plan


def _visible_context_mask(tt: np.ndarray, ss: np.ndarray, t: int, s: int, feature_mode: str) -> np.ndarray:
    if feature_mode == "causal_edge":
        return (tt < t) | ((tt == t) & (ss < s))
    if feature_mode == "diagonal_causal_edge":
        return (tt + ss) < (t + s)
    raise ValueError(f"Unsupported causal edge feature mode: {feature_mode}")


def _augment_feature_for_target_mode(feature4: np.ndarray, target_pred: int, target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return feature4
    if target_mode != "residual":
        raise ValueError(f"Unsupported target mode: {target_mode}")

    patch_shape = feature4.shape[1:]
    pred_value = np.full(patch_shape, float(target_pred) / 255.0, dtype=np.float32)
    residual = np.zeros(patch_shape, dtype=np.float32)
    usable = feature4[1] > 0.5
    if np.any(usable):
        values_u8 = np.rint(feature4[0] * 255.0).astype(np.uint8)
        residual_u8 = ((values_u8.astype(np.int16) - int(target_pred)) & 0xFF).astype(np.uint8)
        residual[usable] = residual_u8[usable].astype(np.float32) / 255.0
    return np.concatenate([feature4, pred_value[None, ...], residual[None, ...]], axis=0)


def _build_causal_edge_feature_array_2d(
    plane: np.ndarray,
    coord: Tuple[int, int],
    patch_shape: Tuple[int, int],
    target_mode: str,
    feature_mode: str = "causal_edge",
    bos_value: int = 0,
) -> np.ndarray:
    t, s = coord
    max_t, max_s = plane.shape
    plan = _get_causal_edge_plan(patch_shape)
    dt = plan["dt"]
    ds = plan["ds"]

    tt = t + dt
    ss = s + ds

    inb = (tt >= 0) & (tt < max_t) & (ss >= 0) & (ss < max_s)
    real = inb & _visible_context_mask(tt, ss, t, s, feature_mode)

    flat = np.zeros((4, dt.size), dtype=np.float32)

    if np.any(real):
        flat[0, real] = plane[tt[real], ss[real]].astype(np.float32) / 255.0
        flat[1, real] = 1.0
        flat[2, real] = 1.0

    remaining = ~real
    if np.any(remaining):
        ct = np.clip(tt, 0, max_t - 1)
        cs = np.clip(ss, 0, max_s - 1)

        mapped_valid = np.zeros(dt.size, dtype=bool)
        mt = np.zeros(dt.size, dtype=np.int32)
        ms = np.zeros(dt.size, dtype=np.int32)

        if not (t == 0 and s == 0):
            if s > 0:
                same_trace = remaining & (ct == t)
                mapped_valid |= same_trace
                mt[same_trace] = t
                ms[same_trace] = s - 1

            if t > 0:
                prev_trace = remaining & (~mapped_valid)
                if feature_mode == "diagonal_causal_edge":
                    prev_trace &= ((np.minimum(ct, t - 1) + cs) < (t + s))
                mapped_valid |= prev_trace
                mt[prev_trace] = np.minimum(ct[prev_trace], t - 1)
                ms[prev_trace] = cs[prev_trace]

        if np.any(mapped_valid):
            flat[0, mapped_valid] = plane[mt[mapped_valid], ms[mapped_valid]].astype(np.float32) / 255.0
            flat[1, mapped_valid] = 1.0
            flat[3, mapped_valid] = 1.0

        bos = remaining & (~mapped_valid)
        if np.any(bos) and bos_value != 0:
            flat[0, bos] = float(bos_value) / 255.0

    feature4 = flat.reshape((4, *patch_shape))
    target_pred = predictor_loco_i_2d(plane, t, s)
    return _augment_feature_for_target_mode(feature4, target_pred, target_mode)


def build_single_stage4_feature_causal_edge(
    volume: np.ndarray,
    coord: Tuple[int, int, int],
    patch_shape: Tuple[int, int],
    target_mode: str,
    feature_mode: str = "causal_edge",
    bos_value: int = 0,
) -> torch.Tensor:
    p, t, s = coord
    feature = _build_causal_edge_feature_array_2d(
        volume[p],
        (t, s),
        patch_shape,
        target_mode=target_mode,
        feature_mode=feature_mode,
        bos_value=bos_value,
    )
    return torch.from_numpy(feature[None, ...])


def build_stage4_features_causal_edge(
    volume: np.ndarray,
    coords: np.ndarray,
    patch_shape: Tuple[int, int],
    target_mode: str,
    feature_mode: str = "causal_edge",
    bos_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_channels = feature_mode_to_in_channels(feature_mode, target_mode)
    features = np.zeros((len(coords), in_channels, *patch_shape), dtype=np.float32)
    labels = np.zeros(len(coords), dtype=np.int64)
    for idx, (p, t, s) in enumerate(coords.tolist()):
        features[idx] = _build_causal_edge_feature_array_2d(
            volume[p],
            (t, s),
            patch_shape,
            target_mode=target_mode,
            feature_mode=feature_mode,
            bos_value=bos_value,
        )
        labels[idx] = target_symbol_for_coord(volume, (p, t, s), target_mode)
    return torch.from_numpy(features), torch.from_numpy(labels)


def build_stage4_features(
    volume: np.ndarray,
    coords: np.ndarray,
    patch_shape: Tuple[int, int],
    feature_mode: str = "strict",
    target_mode: str = "raw",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    if feature_mode in CAUSAL_EDGE_FEATURE_MODES:
        return build_stage4_features_causal_edge(
            volume,
            coords,
            patch_shape,
            target_mode=target_mode,
            feature_mode=feature_mode,
        )
    if target_mode != "raw":
        raise ValueError("Residual target mode is only supported with causal edge feature modes.")

    mask = lexicographic_causal_mask_2d(patch_shape)
    features = np.zeros((len(coords), 2, *patch_shape), dtype=np.float32)
    labels = np.zeros(len(coords), dtype=np.int64)
    for idx, (p, t, s) in enumerate(coords.tolist()):
        padded = _pad_plane(volume[p], patch_shape, feature_mode)
        patch = padded[t : t + patch_shape[0], s : s + patch_shape[1]]
        masked = patch * mask
        features[idx, 0] = normalize_uint8_patch(masked)
        features[idx, 1] = mask
        labels[idx] = int(volume[p, t, s])
    return torch.from_numpy(features), torch.from_numpy(labels)


def build_single_stage4_feature(
    volume: np.ndarray,
    coord: Tuple[int, int, int],
    patch_shape: Tuple[int, int],
    mask: np.ndarray | None = None,
    feature_mode: str = "strict",
    target_mode: str = "raw",
) -> torch.Tensor:
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    if feature_mode in CAUSAL_EDGE_FEATURE_MODES:
        return build_single_stage4_feature_causal_edge(
            volume,
            coord,
            patch_shape,
            target_mode=target_mode,
            feature_mode=feature_mode,
        )
    if target_mode != "raw":
        raise ValueError("Residual target mode is only supported with causal edge feature modes.")

    p, t, s = coord
    mask = lexicographic_causal_mask_2d(patch_shape) if mask is None else mask
    padded = _pad_plane(volume[p], patch_shape, feature_mode)
    patch = padded[t : t + patch_shape[0], s : s + patch_shape[1]]
    masked = patch * mask
    feature = np.stack([normalize_uint8_patch(masked), mask], axis=0).astype(np.float32)
    return torch.from_numpy(feature[None, ...])


def build_loader(features: torch.Tensor, labels: torch.Tensor, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=False,
    )


def loader_item_count(loader: DataLoader) -> int:
    return int(len(loader.dataset))


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: str, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_hits = {k: 0.0 for k in topk}
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            total_loss += float(loss.item()) * inputs.size(0)
            total_items += inputs.size(0)
            max_k = max(topk)
            pred = logits.topk(max_k, dim=1).indices
            for k in topk:
                total_hits[k] += float((pred[:, :k] == targets[:, None]).any(dim=1).float().sum().item())
    metrics = {"loss": total_loss / max(total_items, 1)}
    for k in topk:
        metrics[f"accuracy_top{k}"] = total_hits[k] / max(total_items, 1)
    return metrics


def predict_logits(model: nn.Module, features: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    if int(features.shape[0]) == 0:
        return torch.empty((0, 256), dtype=torch.float32)
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    model.eval()
    chunks = []
    with torch.no_grad():
        for (inputs,) in loader:
            logits = model(inputs.to(device)).cpu()
            chunks.append(logits)
    if not chunks:
        return torch.empty((0, 256), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_state = None
    best_val_loss = None
    has_val = loader_item_count(val_loader) > 0
    history = []
    train_loss = 0.0
    train_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        total_hits = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * inputs.size(0)
            total_items += inputs.size(0)
            total_hits += float((logits.argmax(dim=1) == targets).float().sum().item())
        train_loss = total_loss / max(total_items, 1)
        train_acc = total_hits / max(total_items, 1)
        val_metrics = evaluate_classifier(model, val_loader, device=device, topk=topk) if has_val else None
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": None if val_metrics is None else val_metrics["loss"],
                "val_accuracy": None if val_metrics is None else val_metrics["accuracy_top1"],
            }
        )
        criterion = train_loss if val_metrics is None else float(val_metrics["loss"])
        is_best = best_val_loss is None or criterion < best_val_loss
        if best_val_loss is None or criterion < best_val_loss:
            best_val_loss = criterion
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if val_metrics is None:
            print(
                f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} train_top1={train_acc:.6f} "
                f"best_loss={float(best_val_loss):.6f}{' *' if is_best else ''}",
                flush=True,
            )
        else:
            print(
                f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"train_loss={train_loss:.6f} train_top1={train_acc:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_top1={val_metrics['accuracy_top1']:.6f} "
                f"val_top5={val_metrics.get('accuracy_top5', 0.0):.6f} "
                f"best_loss={float(best_val_loss):.6f}{' *' if is_best else ''}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "train_metrics": {"loss": train_loss, "accuracy_top1": train_acc},
        "val_metrics": None if not has_val else evaluate_classifier(model, val_loader, device=device, topk=topk),
    }


def stage4_benchmark(volume: np.ndarray, coords: np.ndarray, logits: np.ndarray, labels: np.ndarray, feature_mode: str, target_mode: str) -> Dict[str, Any]:
    if len(labels) == 0:
        total_voxels = int(np.prod(volume.shape))
        return {
            "lossless_compatible": feature_mode in {"strict", *CAUSAL_EDGE_FEATURE_MODES},
            "target_mode": target_mode,
            "predictor": "loco_i_2d" if target_mode == "residual" else None,
            "evaluated_voxels": 0,
            "total_voxels": total_voxels,
            "average_nll_bits": None,
            "perplexity": None,
            "ideal_code_length_bits": 0.0,
            "proxy_size_bytes": None,
            "argmax_error_rate": None,
            "accuracy_top1": None,
            "accuracy_top5": None,
        }
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1.0)
    ideal_bits = -np.log2(true_probs)
    avg_bits = float(np.mean(ideal_bits))
    total_voxels = int(np.prod(volume.shape))
    estimated_total_bits = avg_bits * total_voxels
    top1 = np.argmax(logits, axis=1)
    top5 = np.argsort(logits, axis=1)[:, -5:]
    top5_hit = np.any(top5 == labels[:, None], axis=1)
    return {
        "lossless_compatible": feature_mode in {"strict", *CAUSAL_EDGE_FEATURE_MODES},
        "target_mode": target_mode,
        "predictor": "loco_i_2d" if target_mode == "residual" else None,
        "evaluated_voxels": int(len(coords)),
        "total_voxels": total_voxels,
        "average_nll_bits": avg_bits,
        "perplexity": float(2 ** avg_bits),
        "ideal_code_length_bits": float(np.sum(ideal_bits)),
        "proxy_size_bytes": int(math.ceil(estimated_total_bits / 8.0)),
        "argmax_error_rate": float(np.mean(top1 != labels)),
        "accuracy_top1": float(np.mean(top1 == labels)),
        "accuracy_top5": float(np.mean(top5_hit)),
    }


def checkpoint_in_channels(checkpoint: Dict[str, Any]) -> int:
    config = checkpoint.get("config", {})
    if "in_channels" in config:
        return int(config["in_channels"])
    weight = checkpoint["state_dict"]["net.0.weight"]
    return int(weight.shape[1])


def checkpoint_base_channels(checkpoint: Dict[str, Any]) -> int:
    config = checkpoint.get("config", {})
    if "stage4_base_channels" in config:
        return int(config["stage4_base_channels"])
    if "base_channels" in config:
        return int(config["base_channels"])
    weight = checkpoint["state_dict"]["net.0.weight"]
    return int(weight.shape[0])


def load_stage4_model(checkpoint_file: str, device: str) -> Tuple[Small2DCNN, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    in_channels = checkpoint_in_channels(checkpoint)
    base_channels = checkpoint_base_channels(checkpoint)
    model = Small2DCNN(out_dim=256, base_channels=base_channels, in_channels=in_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def resolve_feature_mode(feature_mode: str, checkpoint: Dict[str, Any]) -> str:
    if feature_mode != "auto":
        return feature_mode
    ckpt_mode = checkpoint.get("feature_mode") or checkpoint.get("config", {}).get("feature_mode")
    if ckpt_mode in VALID_FEATURE_MODES:
        return ckpt_mode
    return "strict"


def resolve_target_mode(target_mode: str, checkpoint: Dict[str, Any]) -> str:
    if target_mode != "auto":
        return target_mode
    ckpt_mode = checkpoint.get("target_mode") or checkpoint.get("config", {}).get("target_mode")
    if ckpt_mode in VALID_TARGET_MODES:
        return ckpt_mode
    return "raw"


def run_stage4_training(volume_data: VolumeData, config: ExperimentConfig, device: str, output_root: str) -> Dict[str, Any]:
    set_seed(config.seed)
    feature_mode = config.feature_mode
    target_mode = config.target_mode
    if feature_mode not in VALID_FEATURE_MODES or feature_mode == "legacy":
        raise ValueError("Training supports strict, causal_edge, or diagonal_causal_edge feature modes only.")
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError("Training supports target_mode raw or residual only.")
    if target_mode == "residual" and feature_mode not in CAUSAL_EDGE_FEATURE_MODES:
        raise ValueError("Residual target mode currently requires a causal edge feature mode.")

    volume_train = volume_data.get_split("train")
    volume_val = volume_data.get_split("val")
    volume_test = volume_data.get_split("test")
    if int(volume_train.shape[0]) <= 0:
        raise ValueError("Training requires a non-empty train split.")
    has_val = int(volume_val.shape[0]) > 0
    has_test = int(volume_test.shape[0]) > 0

    valid_region = {
        "train": None,
        "val": None,
        "test": None,
    }
    if region_enabled(config.valid_region_mode):
        valid_region["train"] = detect_profile_rectangles(
            volume_train,
            min_nonzero_ratio=config.valid_region_min_nonzero_ratio,
            margin_traces=config.valid_region_margin_traces,
            group_size=config.valid_region_group_size,
        )
        if has_val:
            valid_region["val"] = detect_profile_rectangles(
                volume_val,
                min_nonzero_ratio=config.valid_region_min_nonzero_ratio,
                margin_traces=config.valid_region_margin_traces,
                group_size=config.valid_region_group_size,
            )
        if has_test:
            valid_region["test"] = detect_profile_rectangles(
                volume_test,
                min_nonzero_ratio=config.valid_region_min_nonzero_ratio,
                margin_traces=config.valid_region_margin_traces,
                group_size=config.valid_region_group_size,
            )

    if valid_region["train"] is None:
        train_coords = choose_random_indices(volume_train.shape, config.stage4_train_samples, config.seed + 21)
    else:
        train_coords = choose_random_indices_in_rectangles(
            volume_train.shape,
            valid_region["train"],
            config.stage4_train_samples,
            config.seed + 21,
        )
    if has_val:
        if valid_region["val"] is None:
            val_coords = choose_random_indices(volume_val.shape, config.stage4_val_samples, config.seed + 22)
        else:
            val_coords = choose_random_indices_in_rectangles(
                volume_val.shape,
                valid_region["val"],
                config.stage4_val_samples,
                config.seed + 22,
            )
    else:
        val_coords = np.empty((0, 3), dtype=np.int64)
    if has_test:
        if valid_region["test"] is None:
            eval_coords = make_regular_grid_indices(volume_test.shape, config.eval_trace_stride, config.eval_sample_stride)
        else:
            eval_coords = make_regular_grid_indices_in_rectangles(
                volume_test.shape,
                valid_region["test"],
                config.eval_trace_stride,
                config.eval_sample_stride,
            )
    else:
        eval_coords = np.empty((0, 3), dtype=np.int64)

    train_x, train_y = build_stage4_features(volume_train, train_coords, config.patch_shape, feature_mode=feature_mode, target_mode=target_mode)
    val_x, val_y = build_stage4_features(volume_val, val_coords, config.patch_shape, feature_mode=feature_mode, target_mode=target_mode)
    eval_x, eval_y = build_stage4_features(volume_test, eval_coords, config.patch_shape, feature_mode=feature_mode, target_mode=target_mode)

    in_channels = feature_mode_to_in_channels(feature_mode, target_mode)
    base_channels = max(1, int(config.stage4_base_channels))
    model = Small2DCNN(out_dim=256, base_channels=base_channels, in_channels=in_channels)
    training = train_classifier(
        model,
        build_loader(train_x, train_y, config, shuffle=True),
        build_loader(val_x, val_y, config, shuffle=False),
        device=device,
        epochs=config.epochs_stage4,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        topk=(1, 5),
    )
    logits = predict_logits(model.to(device), eval_x, device=device, batch_size=config.batch_size).numpy()
    benchmark = stage4_benchmark(volume_test, eval_coords, logits, eval_y.numpy(), feature_mode=feature_mode, target_mode=target_mode)
    ckpt_path = checkpoint_path(output_root, "stage4", "causal")
    torch.save(
        {
            "stage": "stage4",
            "mode": "causal",
            "feature_mode": feature_mode,
            "target_mode": target_mode,
            "model_family": "2d_cnn",
            "predictor": "loco_i_2d" if target_mode == "residual" else None,
            "config": {
                **asdict(config),
                "feature_mode": feature_mode,
                "target_mode": target_mode,
                "in_channels": in_channels,
                "base_channels": base_channels,
            },
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    metrics = {
        "stage": "stage4",
        "mode": "causal",
        "feature_mode": feature_mode,
        "target_mode": target_mode,
        "model_family": "2d_cnn",
        "predictor": "loco_i_2d" if target_mode == "residual" else None,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "train_only": bool(not has_val and not has_test),
        "has_validation_split": bool(has_val),
        "has_test_split": bool(has_test),
        "train_sample_count": int(train_x.shape[0]),
        "val_sample_count": int(val_x.shape[0]),
        "eval_sample_count": int(eval_x.shape[0]),
        "valid_region": {
            "train": rectangles_to_metadata(
                valid_region["train"],
                volume_train.shape,
                config.valid_region_mode,
                config.valid_region_min_nonzero_ratio,
                config.valid_region_margin_traces,
                config.valid_region_group_size,
            ),
            "val": rectangles_to_metadata(
                valid_region["val"],
                volume_val.shape,
                config.valid_region_mode,
                config.valid_region_min_nonzero_ratio,
                config.valid_region_margin_traces,
                config.valid_region_group_size,
            ) if has_val else None,
            "test": rectangles_to_metadata(
                valid_region["test"],
                volume_test.shape,
                config.valid_region_mode,
                config.valid_region_min_nonzero_ratio,
                config.valid_region_margin_traces,
                config.valid_region_group_size,
            ) if has_test else None,
        },
        "training": training,
        "test_metrics": None if not has_test else evaluate_classifier(model.to(device), build_loader(eval_x, eval_y, config, shuffle=False), device=device, topk=(1, 5)),
        "benchmark": benchmark,
        "checkpoint_path": ckpt_path,
    }
    save_json(metrics_path(output_root, "stage4", "causal"), metrics)
    return metrics
