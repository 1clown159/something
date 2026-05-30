"""
SEG-Y 元数据解析器 — 全文件扫描，自动推断 inline/crossline 范围
"""

import struct
import os
from pathlib import Path

FMT_BYTES = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 8: 1}
FMT_NAMES = {1: "IBM 浮点 (4字节)", 2: "32位整数", 3: "16位整数",
             4: "32位定点", 5: "IEEE 浮点 (4字节)", 8: "8位整数"}


def parse_sgy(file_path):
    path = Path(file_path)
    file_size = path.stat().st_size

    with open(file_path, 'rb') as f:
        # === 文本头 + 二进制头 ===
        text_raw = f.read(3200)
        bin_raw = f.read(400)
        format_code = struct.unpack_from('>h', bin_raw, 24)[0]
        sample_count = struct.unpack_from('>h', bin_raw, 20)[0]
        sample_interval = struct.unpack_from('>h', bin_raw, 16)[0]
        bps = FMT_BYTES.get(format_code, 4)
        trace_data_size = sample_count * bps
        trace_total = 240 + trace_data_size
        trace_count = (file_size - 3600) // trace_total if trace_total > 0 else 0
        remainder = (file_size - 3600) % trace_total

        # === 全文件扫描: 收集 inline / crossline / source_x / source_y ===
        inline_trace_counts = {}   # inline_val → trace count
        crossline_set = set()
        src_x_values = []
        src_y_values = []

        f.seek(3600)
        for _ in range(trace_count):
            hdr = f.read(240)
            if len(hdr) < 240:
                break

            il = struct.unpack_from('>i', hdr, 188)[0]
            xl = struct.unpack_from('>i', hdr, 192)[0]
            sx = struct.unpack_from('>i', hdr, 72)[0]
            sy = struct.unpack_from('>i', hdr, 76)[0]

            if il != 0:
                inline_trace_counts[il] = inline_trace_counts.get(il, 0) + 1
            if xl != 0:
                crossline_set.add(xl)
            src_x_values.append(sx)
            src_y_values.append(sy)

            f.seek(trace_data_size, os.SEEK_CUR)

    # === 推断维度 ===
    inline_available = len(inline_trace_counts) > 0
    crossline_available = len(crossline_set) > 0
    src_available = (len(src_x_values) > 0 and any(v != 0 for v in src_x_values))

    profile_count = None
    traces_per_profile = None

    if inline_available:
        # 有 inline_3d → profile_count = 唯一 inline 数
        profile_count = len(inline_trace_counts)
        counts = list(inline_trace_counts.values())
        # traces_per_profile = 每个 inline 中的道数（取最多见的或中位数）
        if len(set(counts)) == 1:
            traces_per_profile = counts[0]
        else:
            from statistics import median
            traces_per_profile = int(median(counts))

        il_vals = sorted(inline_trace_counts.keys())
        il_min, il_max = il_vals[0], il_vals[-1]

        # crossline: 如果道头中没有，backfill 从每个剖面道数取
        if not crossline_available and traces_per_profile and traces_per_profile > 0:
            xl_min, xl_max = 1, traces_per_profile
            crossline_set = set(range(1, traces_per_profile + 1))
        else:
            xl_vals = sorted(crossline_set) if crossline_set else [1, trace_count]
            xl_min, xl_max = xl_vals[0], xl_vals[-1]

    elif src_available:
        # inline/crossline 都没有 → 从 SourceX/SourceY 坐标跳变推断
        sx_arr = [sx for sx in src_x_values]
        sy_arr = [sy for sy in src_y_values]

        # 检测 SourceY 跳变 → 找到剖面边界
        # SourceY 在道内渐变，在剖面边界处跳变
        jumps = []
        threshold = 50  # Y 坐标跳变阈值
        for i in range(1, len(sy_arr)):
            if abs(sy_arr[i] - sy_arr[i-1]) > threshold:
                jumps.append(i)

        if jumps:
            segment_lens = []
            prev = 0
            for j in jumps:
                segment_lens.append(j - prev)
                prev = j
            segment_lens.append(len(sy_arr) - prev)

            from collections import Counter
            freq = Counter(segment_lens)
            traces_per_profile = freq.most_common(1)[0][0]
            profile_count = trace_count // traces_per_profile
        else:
            # 无跳变：SourceY 单调变化 → 整个文件是单剖面
            profile_count = 1
            traces_per_profile = trace_count

        # inline/crossline 范围从坐标推断
        if src_available and profile_count and traces_per_profile:
            il_min, il_max = 1, profile_count
            xl_min, xl_max = 1, traces_per_profile
            crossline_set = set(range(1, traces_per_profile + 1)) if traces_per_profile else set()
            inline_trace_counts = {i+1: traces_per_profile for i in range(profile_count)}
        else:
            il_min = il_max = None
            xl_min = xl_max = None

    else:
        # 没有任何可用信息
        il_min = il_max = None
        xl_min = xl_max = None

    # === 推断维度字符串 ===
    dims = f"{profile_count or '?'} × {traces_per_profile or '?'} × {sample_count}"

    return {
        "file_name":             path.name,
        "file_size_mb":          round(file_size / (1024 * 1024), 2),
        "file_size_bytes":       file_size,
        "text_header_bytes":     3200,
        "binary_header_bytes":   400,
        "sample_count":          sample_count,
        "sample_interval_us":    sample_interval,
        "format_code":           format_code,
        "format_name":           FMT_NAMES.get(format_code, f"未知({format_code})"),
        "bytes_per_sample":      bps,
        "trace_total_bytes":     trace_total,
        "trace_header_bytes":    240,
        "trace_data_bytes":      trace_data_size,
        "trace_count":           trace_count,
        "remainder_bytes":       remainder,
        "inline_min":            il_min,
        "inline_max":            il_max,
        "crossline_min":         xl_min,
        "crossline_max":         xl_max,
        "profile_count":         profile_count,
        "traces_per_profile":    traces_per_profile,
        "unique_inlines":        len(inline_trace_counts),
        "unique_crosslines":     len(crossline_set),
        "inferred_dimensions":   dims,
        "total_samples":         trace_count * sample_count,
        "inline_stored":         inline_available,
        "crossline_stored":      crossline_available,
        "inferred_from":         "inline_3d" if inline_available else ("source_coords" if src_available else "none"),
    }
