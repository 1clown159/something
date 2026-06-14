#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _numba_write_bit(out, out_len, current_byte, num_bits_filled, bit):
        current_byte = (current_byte << 1) | (bit & 1)
        num_bits_filled += 1
        if num_bits_filled == 8:
            if out_len >= out.shape[0]:
                raise RuntimeError("Numba range encoder output buffer overflow.")
            out[out_len] = current_byte
            out_len += 1
            current_byte = 0
            num_bits_filled = 0
        return out_len, current_byte, num_bits_filled


    @njit(cache=True)
    def _numba_encode_symbols(
        cdfs,
        symbols,
        low,
        high,
        num_underflow,
        current_byte,
        num_bits_filled,
        max_output_bytes,
    ):
        num_state_bits = 32
        full_range = 1 << num_state_bits
        half_range = full_range >> 1
        quarter_range = half_range >> 1
        state_mask = full_range - 1
        out = np.empty(max_output_bytes, dtype=np.uint8)
        out_len = 0

        for i in range(symbols.shape[0]):
            symbol = int(symbols[i])
            total = int(cdfs[i, 256])
            sym_low = int(cdfs[i, symbol])
            sym_high = int(cdfs[i, symbol + 1])
            if not (0 <= sym_low < sym_high <= total):
                raise RuntimeError("Invalid CDF interval.")

            rng = high - low + 1
            new_low = low + sym_low * rng // total
            new_high = low + sym_high * rng // total - 1
            low = new_low
            high = new_high

            while ((low ^ high) & half_range) == 0:
                bit = low >> (num_state_bits - 1)
                out_len, current_byte, num_bits_filled = _numba_write_bit(
                    out, out_len, current_byte, num_bits_filled, bit
                )
                for _ in range(num_underflow):
                    out_len, current_byte, num_bits_filled = _numba_write_bit(
                        out, out_len, current_byte, num_bits_filled, bit ^ 1
                    )
                num_underflow = 0
                low = (low << 1) & state_mask
                high = ((high << 1) & state_mask) | 1

            while (low & ~high & quarter_range) != 0:
                num_underflow += 1
                low = (low << 1) ^ half_range
                high = ((high ^ half_range) << 1) | half_range | 1

        return out[:out_len], low, high, num_underflow, current_byte, num_bits_filled


    @njit(cache=True)
    def _numba_encode_boundaries(
        sym_low_arr,
        sym_high_arr,
        total,
        low,
        high,
        num_underflow,
        current_byte,
        num_bits_filled,
        max_output_bytes,
    ):
        num_state_bits = 32
        full_range = 1 << num_state_bits
        half_range = full_range >> 1
        quarter_range = half_range >> 1
        state_mask = full_range - 1
        out = np.empty(max_output_bytes, dtype=np.uint8)
        out_len = 0

        for i in range(sym_low_arr.shape[0]):
            sl = int(sym_low_arr[i])
            sh = int(sym_high_arr[i])
            if not (0 <= sl < sh <= total):
                raise RuntimeError("Invalid boundary interval.")

            rng = high - low + 1
            new_low = low + sl * rng // total
            new_high = low + sh * rng // total - 1
            low = new_low
            high = new_high

            while ((low ^ high) & half_range) == 0:
                bit = low >> (num_state_bits - 1)
                out_len, current_byte, num_bits_filled = _numba_write_bit(
                    out, out_len, current_byte, num_bits_filled, bit
                )
                for _ in range(num_underflow):
                    out_len, current_byte, num_bits_filled = _numba_write_bit(
                        out, out_len, current_byte, num_bits_filled, bit ^ 1
                    )
                num_underflow = 0
                low = (low << 1) & state_mask
                high = ((high << 1) & state_mask) | 1

            while (low & ~high & quarter_range) != 0:
                num_underflow += 1
                low = (low << 1) ^ half_range
                high = ((high ^ half_range) << 1) | half_range | 1

        return out[:out_len], low, high, num_underflow, current_byte, num_bits_filled


    @njit(cache=True)
    def _numba_read_bit(data, byte_index, current_byte, num_bits_remaining):
        if num_bits_remaining == 0:
            if byte_index >= data.shape[0]:
                return 0, byte_index, current_byte, num_bits_remaining
            current_byte = int(data[byte_index])
            byte_index += 1
            num_bits_remaining = 8
        num_bits_remaining -= 1
        bit = (current_byte >> num_bits_remaining) & 1
        return bit, byte_index, current_byte, num_bits_remaining


    @njit(cache=True)
    def _numba_bisect_right_cdf(cdf, value):
        lo = 0
        hi = 257
        while lo < hi:
            mid = (lo + hi) >> 1
            if value < int(cdf[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo


    @njit(cache=True)
    def _numba_decode_symbols(
        cdfs,
        data,
        low,
        high,
        code,
        byte_index,
        current_byte,
        num_bits_remaining,
    ):
        num_state_bits = 32
        full_range = 1 << num_state_bits
        half_range = full_range >> 1
        quarter_range = half_range >> 1
        state_mask = full_range - 1
        symbols = np.empty(cdfs.shape[0], dtype=np.uint8)

        for i in range(cdfs.shape[0]):
            total = int(cdfs[i, 256])
            rng = high - low + 1
            offset = code - low
            value = ((offset + 1) * total - 1) // rng
            symbol = _numba_bisect_right_cdf(cdfs[i], int(value)) - 1
            if symbol < 0 or symbol + 1 >= 257:
                raise RuntimeError("Decoded symbol outside CDF range.")
            symbols[i] = symbol

            sym_low = int(cdfs[i, symbol])
            sym_high = int(cdfs[i, symbol + 1])
            if not (0 <= sym_low < sym_high <= total):
                raise RuntimeError("Invalid CDF interval.")

            new_low = low + sym_low * rng // total
            new_high = low + sym_high * rng // total - 1
            low = new_low
            high = new_high

            while ((low ^ high) & half_range) == 0:
                bit, byte_index, current_byte, num_bits_remaining = _numba_read_bit(
                    data, byte_index, current_byte, num_bits_remaining
                )
                code = ((code << 1) & state_mask) | bit
                low = (low << 1) & state_mask
                high = ((high << 1) & state_mask) | 1

            while (low & ~high & quarter_range) != 0:
                bit, byte_index, current_byte, num_bits_remaining = _numba_read_bit(
                    data, byte_index, current_byte, num_bits_remaining
                )
                code = (code & half_range) | ((code << 1) & (state_mask >> 1)) | bit
                low = (low << 1) ^ half_range
                high = ((high ^ half_range) << 1) | half_range | 1

        return symbols, low, high, code, byte_index, current_byte, num_bits_remaining


class BitOutputStream:
    def __init__(self) -> None:
        self.bytes = bytearray()
        self.current_byte = 0
        self.num_bits_filled = 0

    def write(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            self.bytes.append(self.current_byte)
            self.current_byte = 0
            self.num_bits_filled = 0

    def finish(self) -> bytes:
        if self.num_bits_filled > 0:
            self.current_byte <<= 8 - self.num_bits_filled
            self.bytes.append(self.current_byte)
            self.current_byte = 0
            self.num_bits_filled = 0
        return bytes(self.bytes)


class BitInputStream:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_index = 0
        self.current_byte = 0
        self.num_bits_remaining = 0

    def read(self) -> int:
        if self.num_bits_remaining == 0:
            if self.byte_index >= len(self.data):
                return 0
            self.current_byte = self.data[self.byte_index]
            self.byte_index += 1
            self.num_bits_remaining = 8
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1


class ArithmeticCoderBase:
    def __init__(self, num_state_bits: int = 32) -> None:
        self.num_state_bits = num_state_bits
        self.full_range = 1 << num_state_bits
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = self.full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, cdf: Sequence[int], symbol: int) -> None:
        total = int(cdf[-1])
        sym_low = int(cdf[symbol])
        sym_high = int(cdf[symbol + 1])
        if not (0 <= sym_low < sym_high <= total):
            raise ValueError("Invalid CDF interval.")

        rng = self.high - self.low + 1
        new_low = self.low + sym_low * rng // total
        new_high = self.low + sym_high * rng // total - 1
        self.low = new_low
        self.high = new_high

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = ((self.low << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1

        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self) -> None:
        raise NotImplementedError()

    def underflow(self) -> None:
        raise NotImplementedError()


class RangeEncoder(ArithmeticCoderBase):
    def __init__(self, num_state_bits: int = 32) -> None:
        super().__init__(num_state_bits=num_state_bits)
        self.output = BitOutputStream()
        self.num_underflow = 0

    def shift(self) -> None:
        bit = self.low >> (self.num_state_bits - 1)
        self.output.write(bit)
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0

    def underflow(self) -> None:
        self.num_underflow += 1

    def encode_symbol(self, cdf: Sequence[int], symbol: int) -> None:
        self.update(cdf, symbol)

    def encode_symbols(self, cdfs: Sequence[Sequence[int]], symbols: Sequence[int]) -> None:
        if not NUMBA_AVAILABLE:
            for cdf, symbol in zip(cdfs, symbols):
                self.encode_symbol(cdf, int(symbol))
            return

        cdfs_arr = np.ascontiguousarray(np.asarray(cdfs, dtype=np.int32))
        symbols_arr = np.ascontiguousarray(np.asarray(symbols, dtype=np.int64))
        if cdfs_arr.ndim != 2 or cdfs_arr.shape[1] != 257:
            raise ValueError(f"Expected CDF batch shape [N, 257], got {cdfs_arr.shape}")
        if symbols_arr.ndim != 1 or symbols_arr.shape[0] != cdfs_arr.shape[0]:
            raise ValueError("Symbol batch shape does not match CDF batch.")
        if cdfs_arr.shape[0] == 0:
            return

        max_output_bytes = max(1024, int(cdfs_arr.shape[0]) * 16 + 64)
        while True:
            try:
                out, self.low, self.high, self.num_underflow, self.output.current_byte, self.output.num_bits_filled = _numba_encode_symbols(
                    cdfs_arr,
                    symbols_arr,
                    int(self.low),
                    int(self.high),
                    int(self.num_underflow),
                    int(self.output.current_byte),
                    int(self.output.num_bits_filled),
                    int(max_output_bytes),
                )
                break
            except RuntimeError as exc:
                if "output buffer overflow" not in str(exc):
                    raise
                max_output_bytes *= 2
        self.output.bytes.extend(out.tolist())

    def encode_boundaries(self, total: int, sym_low: np.ndarray, sym_high: np.ndarray) -> None:
        """Encode symbols using pre-computed CDF boundary values (GPU-side CDF lookup).
        
        Args:
            total: Total frequency (same for all symbols in batch)
            sym_low: int32 array of cdf[symbol] values
            sym_high: int32 array of cdf[symbol+1] values
        """
        if not NUMBA_AVAILABLE:
            for lo, hi in zip(sym_low, sym_high):
                self._encode_boundary_single(total, int(lo), int(hi))
            return

        sym_low_arr = np.ascontiguousarray(np.asarray(sym_low, dtype=np.int32))
        sym_high_arr = np.ascontiguousarray(np.asarray(sym_high, dtype=np.int32))
        if sym_low_arr.shape != sym_high_arr.shape:
            raise ValueError("sym_low and sym_high must have same shape")
        n = int(sym_low_arr.shape[0])
        if n == 0:
            return

        max_output_bytes = max(1024, n * 16 + 64)
        while True:
            try:
                out, self.low, self.high, self.num_underflow, self.output.current_byte, self.output.num_bits_filled = _numba_encode_boundaries(
                    sym_low_arr,
                    sym_high_arr,
                    int(total),
                    int(self.low),
                    int(self.high),
                    int(self.num_underflow),
                    int(self.output.current_byte),
                    int(self.output.num_bits_filled),
                    int(max_output_bytes),
                )
                break
            except RuntimeError as exc:
                if "output buffer overflow" not in str(exc):
                    raise
                max_output_bytes *= 2
        self.output.bytes.extend(out.tolist())

    def _encode_boundary_single(self, total: int, sym_low: int, sym_high: int) -> None:
        """Non-numba fallback: encode one symbol from boundaries."""
        num_state_bits = 32
        full_range = 1 << num_state_bits
        half_range = full_range >> 1
        quarter_range = half_range >> 1
        state_mask = full_range - 1

        rng = self.high - self.low + 1
        self.low += sym_low * rng // total
        self.high = self.low + sym_high * rng // total - 1

        while ((self.low ^ self.high) & half_range) == 0:
            bit = self.low >> (num_state_bits - 1)
            self.output.write(bit)
            for _ in range(self.num_underflow):
                self.output.write(bit ^ 1)
            self.num_underflow = 0
            self.low = (self.low << 1) & state_mask
            self.high = ((self.high << 1) & state_mask) | 1

        while (self.low & ~self.high & quarter_range) != 0:
            self.num_underflow += 1
            self.low = (self.low << 1) ^ half_range
            self.high = ((self.high ^ half_range) << 1) | half_range | 1

    def finish(self) -> bytes:
        self.num_underflow += 1
        if self.low < self.quarter_range:
            self.output.write(0)
            for _ in range(self.num_underflow):
                self.output.write(1)
        else:
            self.output.write(1)
            for _ in range(self.num_underflow):
                self.output.write(0)
        return self.output.finish()


class RangeDecoder(ArithmeticCoderBase):
    def __init__(self, data: bytes, num_state_bits: int = 32) -> None:
        super().__init__(num_state_bits=num_state_bits)
        self.input = BitInputStream(data)
        self.code = 0
        for _ in range(self.num_state_bits):
            self.code = (self.code << 1) | self.input.read()

    def shift(self) -> None:
        self.code = ((self.code << 1) & self.state_mask) | self.input.read()

    def underflow(self) -> None:
        self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.input.read()

    def decode_symbol(self, cdf: Sequence[int]) -> int:
        total = int(cdf[-1])
        rng = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // rng
        symbol = bisect_right(cdf, int(value)) - 1
        if symbol < 0 or symbol + 1 >= len(cdf):
            raise ValueError("Decoded symbol outside CDF range.")
        self.update(cdf, symbol)
        return int(symbol)

    def decode_symbols(self, cdfs: Sequence[Sequence[int]]) -> np.ndarray:
        if not NUMBA_AVAILABLE:
            return np.asarray([self.decode_symbol(cdf) for cdf in cdfs], dtype=np.uint8)

        cdfs_arr = np.ascontiguousarray(np.asarray(cdfs, dtype=np.int32))
        if cdfs_arr.ndim != 2 or cdfs_arr.shape[1] != 257:
            raise ValueError(f"Expected CDF batch shape [N, 257], got {cdfs_arr.shape}")
        if cdfs_arr.shape[0] == 0:
            return np.zeros(0, dtype=np.uint8)
        data_arr = np.frombuffer(self.input.data, dtype=np.uint8)
        symbols, self.low, self.high, self.code, self.input.byte_index, self.input.current_byte, self.input.num_bits_remaining = _numba_decode_symbols(
            cdfs_arr,
            data_arr,
            int(self.low),
            int(self.high),
            int(self.code),
            int(self.input.byte_index),
            int(self.input.current_byte),
            int(self.input.num_bits_remaining),
        )
        return symbols
