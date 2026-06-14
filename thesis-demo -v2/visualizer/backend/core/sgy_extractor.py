"""
SEG-Y 数据提取与重建模块 (向量化高性能版)
支持从 SEG-Y 文件中提取浮点数据用于压缩，以及从压缩数据重建 SEG-Y 文件
"""

import os
import struct
import math
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

FMT_BYTES = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 8: 1}
FMT_NAMES = {1: "IBM 浮点 (4字节)", 2: "32位整数", 3: "16位整数",
             4: "32位定点", 5: "IEEE 浮点 (4字节)", 8: "8位整数"}


def _ibm_to_ieee_vectorized(u32_be: np.ndarray) -> np.ndarray:
    """向量化 IBM float32 → IEEE float32 转换 (批量处理)"""
    sign = (u32_be >> 31) & 1
    exp = ((u32_be >> 24) & 0x7F).astype(np.int32)
    mant = (u32_be & 0x00FFFFFF).astype(np.float64)

    f64 = mant * np.power(16.0, exp.astype(np.float64) - 64 - 6)
    mask = sign == 1
    if np.any(mask):
        f64 = f64.copy()
        f64[mask] = -f64[mask]
    return f64.astype(np.float32)


def _ibm_to_ieee(raw4: bytes) -> float:
    bits = int.from_bytes(raw4, 'big')
    sign = -1 if (bits >> 31) & 1 else 1
    exp = (bits >> 24) & 0x7F
    mant = bits & 0x00FFFFFF
    if exp == 0 and mant == 0:
        return 0.0
    return sign * (mant / (1 << 24)) * (16.0 ** (exp - 64))


def _ieee_to_ibm_vectorized(f32: np.ndarray) -> np.ndarray:
    """向量化 IEEE float32 → IBM float32 转换为 big-endian bytes"""
    n = len(f32)
    result = np.zeros(n, dtype=np.uint32)
    nonzero = (f32 != 0.0)

    if np.any(nonzero):
        f = f32[nonzero].astype(np.float64)
        sign_bits = np.where(f < 0, np.uint32(0x80000000), np.uint32(0))
        f = np.abs(f)

        log16 = np.floor(np.log2(f) / 4.0).astype(np.int64)
        exp = log16 + 64
        shift = (24 - 4 * (exp - 64)).astype(np.float64)
        mant = (f * np.power(2.0, shift)).astype(np.uint64)

        overflow = mant >= (1 << 24)
        if np.any(overflow):
            exp = np.where(overflow, exp + 1, exp)
            shift = (24 - 4 * (exp - 64)).astype(np.float64)
            mant = (f * np.power(2.0, shift)).astype(np.uint64)

        exp = np.clip(exp, 0, 127).astype(np.uint32)
        mant = np.minimum(mant, (1 << 24) - 1).astype(np.uint32)
        result[nonzero] = sign_bits | (exp << 24) | mant

    return result.astype('>u4').view('>V4')


def _ieee_to_ibm(value: float) -> bytes:
    if value == 0.0:
        return b'\x00\x00\x00\x00'
    if math.isnan(value) or math.isinf(value):
        value = 0.0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    exp = max(0, min(127, int(math.log(value, 16)) + 64))
    mant = int((value / (16.0 ** (exp - 64))) * (1 << 24))
    if mant >= (1 << 24):
        exp += 1
        mant = int((value / (16.0 ** (exp - 64))) * (1 << 24))
    if mant >= (1 << 24):
        mant = (1 << 24) - 1
    bits = sign | (exp << 24) | mant
    return struct.pack('>I', bits)


def _ieee_to_big_endian_float32(value: float) -> bytes:
    return struct.pack('>f', float(value))


def extract_sgy_headers(file_path: str) -> Dict[str, Any]:
    """
    提取 SEG-Y 全部头信息，并推断三维维度 (向量化扫描)。

    Returns:
        headers dict: {
            "text_header": bytes (3200),
            "binary_header": bytes (400),
            "trace_headers": [bytes (240), ...],
            "meta": { format_code, sample_count, trace_count, bps,
                      profile_count, traces_per_profile, ... }
        }
    """
    path = Path(file_path)
    file_size = path.stat().st_size

    with open(file_path, 'rb') as f:
        text_header = f.read(3200)
        binary_header = f.read(400)

        format_code = struct.unpack_from('>h', binary_header, 24)[0]
        sample_count = struct.unpack_from('>h', binary_header, 20)[0]
        sample_interval = struct.unpack_from('>h', binary_header, 16)[0]
        bps = FMT_BYTES.get(format_code, 4)
        trace_data_bytes = sample_count * bps
        trace_total = 240 + trace_data_bytes
        trace_count = (file_size - 3600) // trace_total if trace_total > 0 else 0

        # 批量读取所有道头 (仅 240B×trace_count，不读取道数据)
        trace_headers_raw = np.zeros((trace_count, 240), dtype=np.uint8)
        f.seek(3600)
        for t in range(trace_count):
            hdr = f.read(240)
            if len(hdr) < 240:
                trace_count = t
                trace_headers_raw = trace_headers_raw[:t]
                break
            trace_headers_raw[t] = np.frombuffer(hdr, dtype=np.uint8)
            f.seek(trace_data_bytes, os.SEEK_CUR)

        trace_headers = [bytes(row) for row in trace_headers_raw]

        # 向量化提取 inline/crossline/source_y 字段
        hdr_u32 = trace_headers_raw[:, 188:196].view('>i4').reshape(-1, 2)
        inline_vals = hdr_u32[:, 0].astype(np.int32)
        xl_vals = hdr_u32[:, 1].astype(np.int32)

        # SourceY at offset 76 (4 bytes big-endian)
        sy_raw = trace_headers_raw[:, 76:80].view('>i4').flatten().astype(np.int32)

        # 统计 inline/crossline
        inline_counts = {}
        for il in inline_vals:
            if il != 0:
                inline_counts[int(il)] = inline_counts.get(int(il), 0) + 1

        crossline_set = set(int(xl) for xl in xl_vals if xl != 0)
        src_y_values = sy_raw.tolist()

    profile_count = None
    traces_per_profile = None

    if len(inline_counts) > 0:
        profile_count = len(inline_counts)
        counts = list(inline_counts.values())
        if len(set(counts)) == 1:
            traces_per_profile = counts[0]
        else:
            from statistics import median
            traces_per_profile = int(median(counts))
    elif len(src_y_values) > 0 and any(v != 0 for v in src_y_values):
        threshold = 50
        jumps = []
        for i in range(1, len(src_y_values)):
            if abs(src_y_values[i] - src_y_values[i - 1]) > threshold:
                jumps.append(i)
        if jumps:
            segment_lens = []
            prev = 0
            for j in jumps:
                segment_lens.append(j - prev)
                prev = j
            segment_lens.append(len(src_y_values) - prev)
            from collections import Counter
            freq = Counter(segment_lens)
            traces_per_profile = freq.most_common(1)[0][0]
            profile_count = trace_count // traces_per_profile
        else:
            profile_count = 1
            traces_per_profile = trace_count

    return {
        "text_header": text_header,
        "binary_header": binary_header,
        "trace_headers": trace_headers,
        "meta": {
            "format_code": format_code,
            "format_name": FMT_NAMES.get(format_code, f"未知({format_code})"),
            "sample_count": sample_count,
            "sample_interval_us": sample_interval,
            "bytes_per_sample": bps,
            "trace_total_bytes": trace_total,
            "trace_data_bytes": trace_data_bytes,
            "trace_count": trace_count,
            "file_size_bytes": file_size,
            "profile_count": profile_count,
            "traces_per_profile": traces_per_profile,
        }
    }


def extract_sgy_float32(
    file_path: str,
    meta: Optional[Dict[str, Any]] = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """
    从 SEG-Y 文件中提取所有采样数据为 float32 numpy 数组 (向量化, 内存高效)。
    返回形状为 (trace_count, sample_count) 的二维数组。
    """
    if meta is None:
        meta = extract_sgy_headers(file_path)["meta"]

    sample_count = meta["sample_count"]
    trace_count = meta["trace_count"]
    bps = meta["bytes_per_sample"]
    format_code = meta["format_code"]
    trace_data_bytes = sample_count * bps
    trace_total = 240 + trace_data_bytes

    data = np.zeros((trace_count, sample_count), dtype=dtype)

    with open(file_path, 'rb') as f:
        for t in range(trace_count):
            f.seek(3600 + t * trace_total + 240)
            raw = f.read(trace_data_bytes)
            if len(raw) < trace_data_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8)

            if format_code == 1:
                u32 = arr.view('>u4')
                data[t] = _ibm_to_ieee_vectorized(u32).astype(dtype)
            elif format_code in (5,):
                data[t] = arr.view('>f4').astype(dtype)
            elif format_code == 2:
                data[t] = arr.view('>i4').astype(dtype)
            else:
                data[t] = arr.astype(dtype)

    return data


def extract_sgy_components(
    file_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    从 SEG-Y 文件提取浮点数据的三个分量。

    Returns:
        (headers_dict, exponents, signs, mants)
    """
    headers = extract_sgy_headers(file_path)
    meta = headers["meta"]
    data = extract_sgy_float32(file_path, meta=meta, dtype=np.float32)

    u32 = data.view(np.uint32)
    signs = ((u32 >> 31) & 0x1).astype(np.uint8)
    exps = ((u32 >> 23) & 0xFF).astype(np.uint8)
    mants = (u32 & 0x7FFFFF).astype(np.uint32)

    return headers, exps, signs, mants


def reconstruct_sgy(
    headers: Dict[str, Any],
    float32_data: np.ndarray,
    output_path: str,
) -> None:
    """
    从头信息和 float32 数据重建 SEG-Y 文件 (向量化批量写入)。

    Args:
        headers: extract_sgy_headers 返回的头信息字典
        float32_data: shape (trace_count, sample_count) 的 float32 数组
        output_path: 输出 SEG-Y 文件路径
    """
    meta = headers["meta"]
    format_code = meta["format_code"]
    trace_count = min(meta["trace_count"], float32_data.shape[0])
    sample_count = meta["sample_count"]
    trace_data_bytes = sample_count * meta["bytes_per_sample"]
    trace_total = 240 + trace_data_bytes
    trace_headers = [bytes(h) for h in headers["trace_headers"][:trace_count]]

    # 向量化：将所有采样数据一次性转换为目标字节格式
    if format_code == 1:
        sample_bytes = _ieee_to_ibm_vectorized(float32_data[:trace_count].ravel()).tobytes()
    elif format_code in (5,):
        sample_bytes = float32_data[:trace_count].astype('>f4').tobytes()
    else:
        sample_bytes = float32_data[:trace_count].astype('>i4').tobytes()

    # 预计算总大小并一次写入
    total_size = 3200 + 400 + trace_count * (240 + trace_data_bytes)
    buf = bytearray(total_size)
    buf[:3200] = headers["text_header"]
    buf[3200:3600] = headers["binary_header"]

    pos = 3600
    sp = 0
    for t in range(trace_count):
        buf[pos:pos + 240] = trace_headers[t]
        pos += 240
        buf[pos:pos + trace_data_bytes] = sample_bytes[sp:sp + trace_data_bytes]
        pos += trace_data_bytes
        sp += trace_data_bytes

    with open(output_path, 'wb') as f:
        f.write(buf)


def reconstruct_float32_from_components(
    signs: np.ndarray,
    exps: np.ndarray,
    mants: np.ndarray,
) -> np.ndarray:
    """
    从符号位、指数、尾数分量重建 float32 数组。

    Args:
        signs: uint8 (0 或 1), shape 任意
        exps: uint8 (0-255), shape 与 signs 相同
        mants: uint32 (低 23 位有效), shape 与 signs 相同

    Returns:
        float32 数组，shape 与输入相同
    """
    u32 = (
        (signs.astype(np.uint32) << 31)
        | (exps.astype(np.uint32) << 23)
        | (mants.astype(np.uint32) & 0x7FFFFF)
    )
    return u32.view(np.float32)
