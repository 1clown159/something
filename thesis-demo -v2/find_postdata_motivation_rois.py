from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
FULL_DATA_PATH = ROOT / "experiments" / "dat" / "PostData-for-training.sgy.dat"
TRAIN_DATA_PATH = ROOT / "experiments" / "dat" / "postdata_8_1_1" / "PostData-for-training.train.dat"
DATA_PATH = FULL_DATA_PATH if FULL_DATA_PATH.exists() else TRAIN_DATA_PATH
OUT_DIR = Path(__file__).resolve().parent

SHAPE = (350, 600, 2001)
STORAGE_SHAPE = SHAPE if DATA_PATH == FULL_DATA_PATH else (280, 600, 2001)
PROFILE_INDEX = SHAPE[0] // 2
ROI = 128
STRIDE = 16
MIN_ACTIVE_RATIO = 0.80
MANTISSA_STREAM_VALUES = 40
MANTISSA_STREAM_STRIDE = 4
DISPLAY_HIGH_BITS = tuple(range(22, 14, -1))
DISPLAY_LOW_BITS = tuple(range(7, -1, -1))
MANAGED_HIGH_NIBBLE = tuple(range(22, 18, -1))
MANAGED_LOW_NIBBLE = tuple(range(3, -1, -1))
MANTISSA_BIT_CMAP = ListedColormap(["#FFFFFF", "#E5E5E5", "#FCD69A"])
EXPONENT_SAMPLE_LINE_OFFSETS = (-3, 0, 3)
EXPONENT_SAMPLE_LINE_COLORS = ("#E53935", "#00B7FF", "#FF8F00")


def load_middle_profile() -> np.ndarray:
    volume = np.memmap(DATA_PATH, dtype=np.float32, mode="r", shape=STORAGE_SHAPE)
    return np.asarray(volume[PROFILE_INDEX], dtype=np.float32)


def float_components(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u32 = np.ascontiguousarray(profile, dtype=np.float32).view(np.uint32)
    signs = ((u32 >> 31) & 0x1).astype(np.uint8)
    exponents = ((u32 >> 23) & 0xFF).astype(np.uint8)
    mantissas = (u32 & 0x7FFFFF).astype(np.uint32)
    return signs, exponents, mantissas


def candidate_starts(limit: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, limit - size + 1, stride))
    last = limit - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    av = af - af.mean()
    bv = bf - bf.mean()
    denom = np.sqrt(np.dot(av, av) * np.dot(bv, bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def sign_score(block: np.ndarray) -> tuple[float, dict[str, float]]:
    p = float(block.mean())
    bias = abs(p - 0.5) * 2.0
    same_x = float((block[:, 1:] == block[:, :-1]).mean())
    same_y = float((block[1:, :] == block[:-1, :]).mean())
    same = 0.5 * (same_x + same_y)
    nontrivial = max(0.0, 1.0 - max(0.0, bias - 0.82) / 0.18)
    score = 1.35 * bias * nontrivial + 0.35 * same
    metrics = {
        "p_sign_1": p,
        "bias_abs_from_half": abs(p - 0.5),
        "neighbor_same": same,
        "score": score,
    }
    return score, metrics


def exponent_score(block: np.ndarray) -> tuple[float, dict[str, float]]:
    e = block.astype(np.float64)
    std = float(e.std())
    span = float(e.max() - e.min())
    corr_x = corr2(e[:, :-1], e[:, 1:])
    corr_y = corr2(e[:-1, :], e[1:, :])
    mean_abs_diff = 0.5 * (
        float(np.abs(np.diff(e, axis=1)).mean())
        + float(np.abs(np.diff(e, axis=0)).mean())
    )
    structure = min(std / 2.0, 1.0) + min(span / 8.0, 1.0)
    score = 1.8 * (corr_x + corr_y) * 0.5 + 0.45 * structure - 0.12 * mean_abs_diff
    metrics = {
        "std": std,
        "range": span,
        "corr_x": corr_x,
        "corr_y": corr_y,
        "mean_abs_neighbor_diff": mean_abs_diff,
        "score": score,
    }
    return score, metrics


def nibble_entropy(values: np.ndarray) -> tuple[float, float]:
    hist = np.bincount(values.ravel(), minlength=16).astype(np.float64)
    prob = hist / hist.sum()
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    kl_from_uniform = 4.0 - entropy
    return entropy, kl_from_uniform


def nibble_neighbor_same(values: np.ndarray) -> float:
    same_x = float((values[:, 1:] == values[:, :-1]).mean())
    same_y = float((values[1:, :] == values[:-1, :]).mean())
    return 0.5 * (same_x + same_y)


def mantissa_score(block: np.ndarray) -> tuple[float, dict[str, float]]:
    nibbles = {
        "m22_m19": ((block >> 19) & 0xF).astype(np.uint8),
        "m18_m15": ((block >> 15) & 0xF).astype(np.uint8),
        "m7_m4": ((block >> 4) & 0xF).astype(np.uint8),
        "m3_m0": (block & 0xF).astype(np.uint8),
    }
    entropy_terms = []
    same_terms = []
    metrics: dict[str, float] = {}
    for name, values in nibbles.items():
        ent, kl = nibble_entropy(values)
        same = nibble_neighbor_same(values)
        entropy_terms.append(kl * min(ent / 2.5, 1.0))
        same_terms.append(same)
        metrics[f"{name}_entropy"] = ent
        metrics[f"{name}_kl_from_uniform"] = kl
        metrics[f"{name}_neighbor_same"] = same
    score = float(np.mean(entropy_terms) + 1.8 * np.mean(same_terms))
    metrics["score"] = score
    return score, metrics


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def mantissa_stream_score(stream: np.ndarray) -> tuple[float, dict[str, float]]:
    high = mantissa_stream_bits(stream, list(range(22, 14, -1))).astype(np.float64)
    low = mantissa_stream_bits(stream, list(range(7, -1, -1))).astype(np.float64)

    high_same = (high[1:] == high[:-1]).mean(axis=0)
    high_p = high.mean(axis=0)
    high_entropy = np.array([binary_entropy(float(p)) for p in high_p])
    high_balance = 4.0 * high_p * (1.0 - high_p)
    high_useful = high_same * high_balance

    low_same = (low[1:] == low[:-1]).mean(axis=0)
    low_p = low.mean(axis=0)
    low_entropy = np.array([binary_entropy(float(p)) for p in low_p])
    low_balance = 4.0 * low_p * (1.0 - low_p)

    high_score = float(0.70 * high_useful.mean() + 0.30 * np.sort(high_useful)[-4:].mean())
    low_score = float((low_same * low_balance).mean())
    score = 1.15 * high_score + 0.25 * low_score
    metrics = {
        "score": score,
        "high_score": high_score,
        "low_score": low_score,
        "high_neighbor_same_mean": float(high_same.mean()),
        "high_entropy_mean": float(high_entropy.mean()),
        "high_balance_mean": float(high_balance.mean()),
        "low_neighbor_same_mean": float(low_same.mean()),
        "low_entropy_mean": float(low_entropy.mean()),
        "low_balance_mean": float(low_balance.mean()),
    }
    for idx, bit in enumerate(range(22, 14, -1)):
        metrics[f"m{bit}_same"] = float(high_same[idx])
        metrics[f"m{bit}_p1"] = float(high_p[idx])
    return score, metrics


def find_best(
    component: np.ndarray,
    score_fn,
    starts_trace: list[int],
    starts_sample: list[int],
    activity: np.ndarray,
    activity_threshold: float,
    global_std: float,
    anchor_center: tuple[float, float] | None = None,
    max_anchor_distance: float | None = None,
) -> dict:
    best: dict | None = None
    for t0 in starts_trace:
        for s0 in starts_sample:
            activity_block = activity[t0 : t0 + ROI, s0 : s0 + ROI]
            raw_active_ratio = float((activity_block > activity_threshold).mean())
            raw_std = float(activity_block.std())
            if raw_active_ratio < MIN_ACTIVE_RATIO or raw_std < 0.15 * global_std:
                continue
            block = component[t0 : t0 + ROI, s0 : s0 + ROI]
            score, metrics = score_fn(block)
            activity_weight = min(1.25, 0.75 + 0.5 * raw_active_ratio)
            proximity_weight = 1.0
            if anchor_center is not None and max_anchor_distance is not None:
                center = (t0 + ROI / 2.0, s0 + ROI / 2.0)
                dist = float(np.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1]))
                if dist > max_anchor_distance:
                    continue
                proximity_weight = 0.65 + 0.35 * (1.0 - dist / max_anchor_distance)
                metrics["anchor_distance"] = dist
                metrics["proximity_weight"] = proximity_weight
            weighted_score = score * activity_weight
            weighted_score *= proximity_weight
            metrics = dict(metrics)
            metrics["raw_active_ratio"] = raw_active_ratio
            metrics["raw_std"] = raw_std
            metrics["weighted_score"] = weighted_score
            if best is None or weighted_score > best["score"]:
                best = {
                    "trace_start": int(t0),
                    "sample_start": int(s0),
                    "trace_end": int(t0 + ROI),
                    "sample_end": int(s0 + ROI),
                    "roi_size": ROI,
                    "score": float(weighted_score),
                    "component_score": float(score),
                    "metrics": metrics,
                }
    assert best is not None
    return best


def roi_from_stream(trace: int, sample_start: int, shape: tuple[int, int]) -> dict:
    trace_start = max(0, min(trace - ROI // 2, shape[0] - ROI))
    sample_anchor = sample_start + MANTISSA_STREAM_VALUES // 2
    sample_start_roi = max(0, min(sample_anchor - ROI // 2, shape[1] - ROI))
    return {
        "trace_start": int(trace_start),
        "sample_start": int(sample_start_roi),
        "trace_end": int(trace_start + ROI),
        "sample_end": int(sample_start_roi + ROI),
        "roi_size": ROI,
    }


def find_best_mantissa_stream(
    mantissas: np.ndarray,
    profile: np.ndarray,
    activity_threshold: float,
    global_std: float,
    anchor_center: tuple[float, float],
    max_anchor_distance: float,
) -> dict:
    best: dict | None = None
    max_sample_start = mantissas.shape[1] - MANTISSA_STREAM_VALUES
    for trace in range(mantissas.shape[0]):
        for sample_start in range(0, max_sample_start + 1, MANTISSA_STREAM_STRIDE):
            center = (float(trace), float(sample_start + MANTISSA_STREAM_VALUES / 2.0))
            dist = float(np.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1]))
            if dist > max_anchor_distance:
                continue
            raw_segment = np.abs(profile[trace, sample_start : sample_start + MANTISSA_STREAM_VALUES])
            raw_active_ratio = float((raw_segment > activity_threshold).mean())
            raw_std = float(raw_segment.std())
            if raw_active_ratio < MIN_ACTIVE_RATIO or raw_std < 0.08 * global_std:
                continue
            stream = mantissas[trace, sample_start : sample_start + MANTISSA_STREAM_VALUES]
            component_score, metrics = mantissa_stream_score(stream)
            proximity_weight = 0.75 + 0.25 * (1.0 - dist / max_anchor_distance)
            activity_weight = min(1.20, 0.80 + 0.40 * raw_active_ratio)
            weighted_score = component_score * proximity_weight * activity_weight
            metrics = dict(metrics)
            metrics.update(
                {
                    "anchor_distance": dist,
                    "proximity_weight": proximity_weight,
                    "raw_active_ratio": raw_active_ratio,
                    "raw_std": raw_std,
                    "weighted_score": weighted_score,
                }
            )
            if best is None or weighted_score > best["score"]:
                roi = roi_from_stream(trace, sample_start, mantissas.shape)
                best = {
                    **roi,
                    "score": float(weighted_score),
                    "component_score": float(component_score),
                    "metrics": metrics,
                    "stream_view": {
                        "values": MANTISSA_STREAM_VALUES,
                        "trace": int(trace),
                        "sample_start": int(sample_start),
                        "sample_end": int(sample_start + MANTISSA_STREAM_VALUES),
                        "shown_bits": "m22:m15 ... m7:m0",
                    },
                }
    if best is None:
        raise RuntimeError("No valid mantissa stream candidates.")
    return best


def find_best_sign_exponent_joint(
    signs: np.ndarray,
    exponents: np.ndarray,
    starts_trace: list[int],
    starts_sample: list[int],
    activity: np.ndarray,
    activity_threshold: float,
    global_std: float,
) -> dict:
    candidates = []
    for t0 in starts_trace:
        for s0 in starts_sample:
            activity_block = activity[t0 : t0 + ROI, s0 : s0 + ROI]
            raw_active_ratio = float((activity_block > activity_threshold).mean())
            raw_std = float(activity_block.std())
            if raw_active_ratio < MIN_ACTIVE_RATIO or raw_std < 0.15 * global_std:
                continue
            sign_component_score, sign_metrics = sign_score(signs[t0 : t0 + ROI, s0 : s0 + ROI])
            exp_component_score, exp_metrics = exponent_score(exponents[t0 : t0 + ROI, s0 : s0 + ROI])
            candidates.append(
                {
                    "trace_start": int(t0),
                    "sample_start": int(s0),
                    "trace_end": int(t0 + ROI),
                    "sample_end": int(s0 + ROI),
                    "roi_size": ROI,
                    "sign_component_score": float(sign_component_score),
                    "exponent_component_score": float(exp_component_score),
                    "metrics": {
                        "sign": sign_metrics,
                        "exponent": exp_metrics,
                        "raw_active_ratio": raw_active_ratio,
                        "raw_std": raw_std,
                    },
                }
            )
    if not candidates:
        raise RuntimeError("No valid sign/exponent joint ROI candidates.")
    max_sign = max(item["sign_component_score"] for item in candidates)
    max_exp = max(item["exponent_component_score"] for item in candidates)
    best = None
    for item in candidates:
        sign_norm = item["sign_component_score"] / max(max_sign, 1e-12)
        exp_norm = item["exponent_component_score"] / max(max_exp, 1e-12)
        activity_weight = min(1.25, 0.75 + 0.5 * item["metrics"]["raw_active_ratio"])
        joint_score = (0.42 * sign_norm + 0.58 * exp_norm) * activity_weight
        item["score"] = float(joint_score)
        item["metrics"]["joint_score"] = float(joint_score)
        item["metrics"]["sign_normalized"] = float(sign_norm)
        item["metrics"]["exponent_normalized"] = float(exp_norm)
        if best is None or item["score"] > best["score"]:
            best = item
    assert best is not None
    return best


def percentile_clip(profile: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    lo, hi = np.percentile(profile[np.isfinite(profile)], [low, high])
    if lo == hi:
        lo, hi = float(profile.min()), float(profile.max())
    return float(lo), float(hi)


def add_roi_rect(ax, roi: dict, color: str, label: str) -> None:
    rect = patches.Rectangle(
        (roi["trace_start"], roi["sample_start"]),
        ROI,
        ROI,
        linewidth=2.2,
        edgecolor=color,
        facecolor="none",
        label=label,
    )
    ax.add_patch(rect)


def add_dashed_roi_rect(ax, roi: dict, color: str, label: str) -> None:
    rect = patches.Rectangle(
        (roi["trace_start"], roi["sample_start"]),
        ROI,
        ROI,
        linewidth=2.8,
        linestyle=(0, (5, 4)),
        edgecolor=color,
        facecolor="none",
        label=label,
    )
    ax.add_patch(rect)


def save_profile_marker_variants(profile: np.ndarray, roi: dict) -> None:
    vmin, vmax = percentile_clip(profile)

    fig, ax = plt.subplots(figsize=(7.2, 9.2), constrained_layout=True)
    ax.imshow(profile.T, cmap="gray", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    add_dashed_roi_rect(ax, roi, "#FFD400", "Sign/Exponent ROI")
    ax.set_title("PostData middle profile 175")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.savefig(OUT_DIR / "postdata_middle_profile_175_sign_exponent_marker.png", dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 9.2), constrained_layout=True)
    ax.imshow(profile.T, cmap="gray", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title("PostData middle profile 175")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.savefig(OUT_DIR / "postdata_middle_profile_175_clean.png", dpi=260)
    plt.close(fig)


def save_overview(profile: np.ndarray, rois: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    vmin, vmax = percentile_clip(profile)
    ax.imshow(
        profile.T,
        cmap="gray",
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    add_roi_rect(ax, rois["sign_exponent"], "#0B6FAE", "Sign/Exponent ROI")
    add_roi_rect(ax, rois["mantissa"], "#6D2A8E", "Mantissa ROI")
    ax.set_title(f"PostData middle profile {PROFILE_INDEX}: selected 128x128 ROIs")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(OUT_DIR / "postdata_middle_profile_roi_overview.png", dpi=220)
    plt.close(fig)


def roi_block(component: np.ndarray, roi: dict) -> np.ndarray:
    return component[
        roi["trace_start"] : roi["trace_end"],
        roi["sample_start"] : roi["sample_end"],
    ]


def add_sign_corner_legend(ax) -> None:
    handles = [
        patches.Patch(facecolor="#F4F4F2", edgecolor="#30363D", linewidth=0.4, label="0"),
        patches.Patch(facecolor="#0B79B4", edgecolor="#30363D", linewidth=0.4, label="1"),
    ]
    legend = ax.legend(
        handles=handles,
        title=None,
        loc="upper right",
        fontsize=7.4,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.05,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#B0B8C0")
    legend.get_frame().set_linewidth(0.6)


def exponent_line_specs(n_cols: int) -> list[tuple[int, str]]:
    center = int(round(0.5 * (n_cols - 1)))
    specs: list[tuple[int, str]] = []
    for offset, color in zip(EXPONENT_SAMPLE_LINE_OFFSETS, EXPONENT_SAMPLE_LINE_COLORS):
        col = min(max(center + int(offset), 0), n_cols - 1)
        if all(existing_col != col for existing_col, _ in specs):
            specs.append((col, color))
    return specs


def draw_exponent_sample_lines(ax, n_rows: int, n_cols: int) -> list[tuple[int, str]]:
    specs = exponent_line_specs(n_cols)
    for col, color in specs:
        ax.vlines(
            x=col,
            ymin=-0.5,
            ymax=n_rows - 0.5,
            colors=color,
            linestyles=(0, (5, 3)),
            linewidth=1.4,
            alpha=0.95,
        )
    return specs


def add_exponent_corner_legend(ax, line_specs: list[tuple[int, str]]) -> None:
    handles = [
        Line2D([0], [0], color=color, linestyle=(0, (5, 3)), linewidth=1.6, label=f"L{idx}")
        for idx, (_, color) in enumerate(line_specs, start=1)
    ]
    legend = ax.legend(
        handles=handles,
        title=None,
        loc="upper right",
        fontsize=7.4,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.7,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#B0B8C0")
    legend.get_frame().set_linewidth(0.6)


def draw_corresponding_amplitude_traces(ax, data_block: np.ndarray, line_specs: list[tuple[int, str]], y_label: str = "Amplitude") -> None:
    x = np.arange(data_block.shape[0], dtype=np.float64)
    for idx, (col, color) in enumerate(line_specs, start=1):
        trace = data_block[:, col].astype(np.float64)
        ax.plot(x, trace, color=color, linewidth=1.3, label=f"L{idx}")

    ax.set_title("Corresponding amplitude profiles", fontsize=10)
    ax.set_xlabel("Sample within ROI")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color="#888888")


def save_sign(signs: np.ndarray, roi: dict) -> None:
    block = roi_block(signs, roi).T
    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    cmap = ListedColormap(["#F4F4F2", "#0B79B4"])
    ax.imshow(block, cmap=cmap, interpolation="nearest", origin="upper", vmin=0, vmax=1)
    ax.set_title("Sign stream: biased binary map")
    ax.set_xlabel("Trace within ROI")
    ax.set_ylabel("Sample within ROI")
    ax.set_xticks([])
    ax.set_yticks([])
    add_sign_corner_legend(ax)
    fig.savefig(OUT_DIR / "postdata_sign_roi.png", dpi=260)
    plt.close(fig)


def save_exponent(profile: np.ndarray, exponents: np.ndarray, roi: dict) -> None:
    block = roi_block(exponents, roi).T
    profile_block = roi_block(profile, roi).T
    fig = plt.figure(figsize=(5.8, 6.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.55])
    ax = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[1, 0])
    im = ax.imshow(block, cmap="viridis", interpolation="nearest", origin="upper")
    line_specs = draw_exponent_sample_lines(ax, n_rows=block.shape[0], n_cols=block.shape[1])
    add_exponent_corner_legend(ax, line_specs)
    ax.set_title("Exponent stream: spatially coherent scale field")
    ax.set_xlabel("")
    ax.set_ylabel("Sample within ROI")
    ax.set_xticks([])
    ax.set_yticks([])
    draw_corresponding_amplitude_traces(ax_amp, profile_block, line_specs, y_label="Amplitude")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Biased exponent")
    fig.savefig(OUT_DIR / "postdata_exponent_roi.png", dpi=260)
    plt.close(fig)


def save_sign_exponent_source_region(
    profile: np.ndarray,
    signs: np.ndarray,
    exponents: np.ndarray,
    roi: dict,
) -> None:
    fig = plt.figure(figsize=(13.2, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1, 1, 1])
    vmin, vmax = percentile_clip(profile)

    ax_profile = fig.add_subplot(gs[0, 0])
    ax_profile.imshow(profile.T, cmap="gray", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    add_dashed_roi_rect(ax_profile, roi, "#FFD400", "Shared ROI")
    ax_profile.set_title("Middle profile 175")
    ax_profile.set_xlabel("Trace")
    ax_profile.set_ylabel("Sample")
    ax_profile.legend(loc="lower right", fontsize=8)

    ax_raw = fig.add_subplot(gs[0, 1])
    raw_block = roi_block(profile, roi).T
    rvmin, rvmax = percentile_clip(raw_block)
    ax_raw.imshow(raw_block, cmap="gray", interpolation="nearest", origin="upper", vmin=rvmin, vmax=rvmax)
    ax_raw.set_title("Selected input ROI")

    ax_sign = fig.add_subplot(gs[0, 2])
    ax_sign.imshow(
        roi_block(signs, roi).T,
        cmap=ListedColormap(["#F4F4F2", "#0B79B4"]),
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
    )
    ax_sign.set_title("Sign field")
    add_sign_corner_legend(ax_sign)

    ax_exp = fig.add_subplot(gs[0, 3])
    exp_block = roi_block(exponents, roi).T
    im = ax_exp.imshow(exp_block, cmap="viridis", interpolation="nearest", origin="upper")
    line_specs = draw_exponent_sample_lines(ax_exp, n_rows=exp_block.shape[0], n_cols=exp_block.shape[1])
    add_exponent_corner_legend(ax_exp, line_specs)
    ax_exp.set_title("Exponent field")
    fig.colorbar(im, ax=ax_exp, fraction=0.048, pad=0.03)

    for ax in [ax_raw, ax_sign, ax_exp]:
        ax.set_xticks([])
        ax.set_yticks([])
    coord = (
        f"Shared ROI: trace [{roi['trace_start']}, {roi['trace_end']}), "
        f"sample [{roi['sample_start']}, {roi['sample_end']})"
    )
    fig.suptitle(coord, fontsize=14)
    fig.savefig(OUT_DIR / "postdata_sign_exponent_source_region.png", dpi=260)
    plt.close(fig)


def save_mantissa(mantissas: np.ndarray, roi: dict) -> None:
    stream, _ = mantissa_stream(mantissas, roi)
    fig = plt.figure(figsize=(6.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[8, 0.75, 8, 0.65])
    high_ax = fig.add_subplot(gs[0, 0])
    fold_ax = fig.add_subplot(gs[0, 1])
    low_ax = fig.add_subplot(gs[0, 2])
    legend_ax = fig.add_subplot(gs[0, 3])
    high_ax.imshow(mantissa_display_values(stream, DISPLAY_HIGH_BITS, MANAGED_HIGH_NIBBLE), cmap=MANTISSA_BIT_CMAP, vmin=0, vmax=2, interpolation="nearest", origin="upper", aspect="equal")
    low_ax.imshow(mantissa_display_values(stream, DISPLAY_LOW_BITS, MANAGED_LOW_NIBBLE), cmap=MANTISSA_BIT_CMAP, vmin=0, vmax=2, interpolation="nearest", origin="upper", aspect="equal")
    high_ax.set_title("High 8 bits\nmanaged m22:m19")
    low_ax.set_title("Low 8 bits\nmanaged m3:m0")
    for ax in [high_ax, low_ax]:
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"m{b}" for b in (DISPLAY_HIGH_BITS if ax is high_ax else DISPLAY_LOW_BITS)], rotation=90, fontsize=8)
        ax.set_ylabel("Consecutive float values")
        ax.set_yticks([0, MANTISSA_STREAM_VALUES // 2, MANTISSA_STREAM_VALUES - 1])
        add_cell_grid(ax, cols=8, rows=stream.shape[0])
    add_torn_edge(high_ax, "right", rows=stream.shape[0])
    add_torn_edge(low_ax, "left", rows=stream.shape[0])
    fold_ax.axis("off")
    fold_ax.text(0.5, 0.5, "...\nfold\nm14:m8", ha="center", va="center", fontsize=11)
    legend_ax.imshow(np.array([[0], [1], [2]], dtype=np.uint8), cmap=MANTISSA_BIT_CMAP, vmin=0, vmax=2, interpolation="nearest")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([0, 1, 2])
    legend_ax.set_yticklabels(["0", "other 1", "managed 1"], fontsize=7)
    legend_ax.set_title("bit", fontsize=9)
    fig.suptitle(f"Nibble-managed mantissa stream ({MANTISSA_STREAM_VALUES} values)")
    fig.savefig(OUT_DIR / "postdata_mantissa_folded_roi.png", dpi=260)
    fig.savefig(OUT_DIR / "postdata_mantissa_folded_roi_v2.png", dpi=260)
    plt.close(fig)


def mantissa_stream(mantissas: np.ndarray, roi: dict) -> tuple[np.ndarray, tuple[int, int]]:
    if "stream_view" in roi:
        trace = int(roi["stream_view"]["trace"])
        sample_start = int(roi["stream_view"]["sample_start"])
        values = int(roi["stream_view"].get("values", MANTISSA_STREAM_VALUES))
        stream = mantissas[trace, sample_start : sample_start + values]
        return stream.astype(np.uint32), (trace, sample_start)
    trace = int((roi["trace_start"] + roi["trace_end"]) // 2)
    sample_start = int((roi["sample_start"] + roi["sample_end"] - MANTISSA_STREAM_VALUES) // 2)
    sample_start = max(roi["sample_start"], min(sample_start, roi["sample_end"] - MANTISSA_STREAM_VALUES))
    stream = mantissas[trace, sample_start : sample_start + MANTISSA_STREAM_VALUES]
    return stream.astype(np.uint32), (trace, sample_start)


def mantissa_stream_bits(stream: np.ndarray, bits: list[int]) -> np.ndarray:
    columns = [((stream >> bit) & 0x1).astype(np.uint8) for bit in bits]
    return np.stack(columns, axis=1)


def mantissa_display_values(
    stream: np.ndarray,
    bits: tuple[int, ...],
    managed_bits: tuple[int, ...],
) -> np.ndarray:
    display = mantissa_stream_bits(stream, list(bits))
    managed_columns = np.isin(np.asarray(bits), np.asarray(managed_bits))
    display[:, managed_columns] *= 2
    return display


def add_cell_grid(ax, cols: int, rows: int) -> None:
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)


def add_torn_edge(ax, side: str, rows: int, cols: int = 8) -> None:
    y = np.linspace(-0.5, rows - 0.5, 25)
    # Use complementary offsets for left/right so the torn edges visually match.
    phase = np.arange(y.size) % 2
    offset = 0.22 + 0.22 * phase
    if side == "right":
        edge = cols - 0.5
        jagged_x = edge - offset
        poly_x = np.concatenate([[edge, edge], jagged_x[::-1]])
        poly_y = np.concatenate([[-0.5, rows - 0.5], y[::-1]])
    elif side == "left":
        edge = -0.5
        jagged_x = edge + (0.66 - offset)
        poly_x = np.concatenate([[edge, edge], jagged_x[::-1]])
        poly_y = np.concatenate([[-0.5, rows - 0.5], y[::-1]])
    else:
        raise ValueError("side must be 'left' or 'right'")
    ax.fill(poly_x, poly_y, color="white", zorder=5, clip_on=False)
    ax.plot(jagged_x, y, color="black", linewidth=1.1, zorder=6, clip_on=False)


def save_combined(profile: np.ndarray, signs: np.ndarray, exponents: np.ndarray, mantissas: np.ndarray, rois: dict[str, dict]) -> None:
    fig = plt.figure(figsize=(12.8, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.45], height_ratios=[1, 1])
    ax_profile = fig.add_subplot(gs[:, 0])
    vmin, vmax = percentile_clip(profile)
    ax_profile.imshow(profile.T, cmap="gray", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    add_roi_rect(ax_profile, rois["sign_exponent"], "#0B6FAE", "Sign/Exponent")
    add_roi_rect(ax_profile, rois["mantissa"], "#6D2A8E", "Mantissa")
    ax_profile.set_title(f"Middle profile {PROFILE_INDEX}")
    ax_profile.set_xlabel("Trace")
    ax_profile.set_ylabel("Sample")
    ax_profile.legend(loc="lower right", fontsize=8)

    mid = gs[:, 1].subgridspec(3, 1, height_ratios=[1.0, 1.1, 0.6], hspace=0.08)
    ax_sign = fig.add_subplot(mid[0, 0])
    ax_sign.imshow(roi_block(signs, rois["sign_exponent"]).T, cmap=ListedColormap(["#F4F4F2", "#0B79B4"]), interpolation="nearest", origin="upper", vmin=0, vmax=1)
    ax_sign.set_title("Sign: biased binary")
    ax_sign.set_xticks([])
    ax_sign.set_yticks([])
    add_sign_corner_legend(ax_sign)

    ax_exp = fig.add_subplot(mid[1, 0])
    exp_block = roi_block(exponents, rois["sign_exponent"]).T
    exp_im = ax_exp.imshow(exp_block, cmap="viridis", interpolation="nearest", origin="upper")
    line_specs = draw_exponent_sample_lines(ax_exp, n_rows=exp_block.shape[0], n_cols=exp_block.shape[1])
    add_exponent_corner_legend(ax_exp, line_specs)
    ax_exp.set_title("Exponent: spatial coherence")
    ax_exp.set_xticks([])
    ax_exp.set_yticks([])
    fig.colorbar(exp_im, ax=ax_exp, fraction=0.046, pad=0.03)
    ax_amp = fig.add_subplot(mid[2, 0])
    draw_corresponding_amplitude_traces(ax_amp, roi_block(profile, rois["sign_exponent"]).T, line_specs, y_label="Amplitude")

    sub = gs[:, 2].subgridspec(1, 3, width_ratios=[8, 0.7, 8])
    stream, _ = mantissa_stream(mantissas, rois["mantissa"])
    high_ax = fig.add_subplot(sub[0, 0])
    high_ax.imshow(mantissa_display_values(stream, DISPLAY_HIGH_BITS, MANAGED_HIGH_NIBBLE), cmap=MANTISSA_BIT_CMAP, vmin=0, vmax=2, interpolation="nearest", origin="upper", aspect="equal")
    high_ax.set_title("Mantissa high 8 bits\nmanaged m22:m19")
    high_ax.set_xticks(range(8))
    high_ax.set_xticklabels([f"m{b}" for b in DISPLAY_HIGH_BITS], rotation=90, fontsize=8)
    high_ax.set_yticks([])
    add_cell_grid(high_ax, cols=8, rows=stream.shape[0])
    add_torn_edge(high_ax, "right", rows=stream.shape[0])
    ax_fold = fig.add_subplot(sub[0, 1])
    ax_fold.axis("off")
    ax_fold.text(0.5, 0.5, "...\nfold\nm14:m8", ha="center", va="center", fontsize=10)
    low_ax = fig.add_subplot(sub[0, 2])
    low_ax.imshow(mantissa_display_values(stream, DISPLAY_LOW_BITS, MANAGED_LOW_NIBBLE), cmap=MANTISSA_BIT_CMAP, vmin=0, vmax=2, interpolation="nearest", origin="upper", aspect="equal")
    low_ax.set_title("Mantissa low 8 bits\nmanaged m3:m0")
    low_ax.set_xticks(range(8))
    low_ax.set_xticklabels([f"m{b}" for b in DISPLAY_LOW_BITS], rotation=90, fontsize=8)
    low_ax.set_yticks([])
    add_cell_grid(low_ax, cols=8, rows=stream.shape[0])
    add_torn_edge(low_ax, "left", rows=stream.shape[0])
    fig.suptitle("PostData float32 nibble-managed bit-stream structure from the middle profile")
    fig.savefig(OUT_DIR / "postdata_bitstream_roi_combined.png", dpi=240)
    fig.savefig(OUT_DIR / "postdata_bitstream_roi_combined_v2.png", dpi=240)
    plt.close(fig)


def main() -> None:
    profile = load_middle_profile()
    signs, exponents, mantissas = float_components(profile)
    starts_trace = candidate_starts(profile.shape[0], ROI, STRIDE)
    starts_sample = candidate_starts(profile.shape[1], ROI, STRIDE)
    activity = np.abs(profile)
    activity_threshold = float(np.percentile(activity, 20))
    global_std = float(profile.std())

    sign_exponent_roi = find_best_sign_exponent_joint(
        signs,
        exponents,
        starts_trace,
        starts_sample,
        activity,
        activity_threshold,
        global_std,
    )
    anchor_center = (
        0.5 * (sign_exponent_roi["trace_start"] + sign_exponent_roi["trace_end"]),
        0.5 * (sign_exponent_roi["sample_start"] + sign_exponent_roi["sample_end"]),
    )
    rois = {
        "sign_exponent": sign_exponent_roi,
        "sign": sign_exponent_roi,
        "exponent": sign_exponent_roi,
    }
    rois["mantissa"] = find_best_mantissa_stream(
        mantissas,
        profile,
        activity_threshold,
        global_std,
        anchor_center,
        max_anchor_distance=320.0,
    )
    mantissa_stream_values, mantissa_stream_start = mantissa_stream(mantissas, rois["mantissa"])
    summary_rois = {
        "sign_exponent": sign_exponent_roi,
        "mantissa": rois["mantissa"],
    }
    summary = {
        "data_path": str(DATA_PATH),
        "shape": SHAPE,
        "profile_index": PROFILE_INDEX,
        "profile_shape": list(profile.shape),
        "roi_size": ROI,
        "stride": STRIDE,
        "activity_filter": {
            "abs_amplitude_percentile_20": activity_threshold,
            "min_active_ratio": MIN_ACTIVE_RATIO,
            "min_raw_std_fraction_of_profile_std": 0.15,
            "profile_std": global_std,
        },
        "rois": summary_rois,
    }
    (OUT_DIR / "postdata_middle_profile_roi_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    save_overview(profile, rois)
    save_profile_marker_variants(profile, rois["sign_exponent"])
    save_sign(signs, rois["sign_exponent"])
    save_exponent(profile, exponents, rois["sign_exponent"])
    save_sign_exponent_source_region(profile, signs, exponents, rois["sign_exponent"])
    save_mantissa(mantissas, rois["mantissa"])
    save_combined(profile, signs, exponents, mantissas, rois)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
