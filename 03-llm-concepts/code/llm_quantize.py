"""Quantization basics: int8 symmetric and asymmetric, error, memory math, outliers. Standard library only.

Quantization stores each weight as a small integer plus a shared scale, instead of a 16-bit float.
You save memory and pay a small rounding error. Here you can measure both.

Run:  python llm_quantize.py
"""
import random


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ---------- symmetric: zero maps to zero, range is [-max|x|, +max|x|] ----------
def quantize_symmetric(values, bits=8):
    """Return (ints, scale). ints are in [-(2^(bits-1)-1), 2^(bits-1)-1]."""
    qmax = 2 ** (bits - 1) - 1
    peak = max(abs(v) for v in values)
    scale = peak / qmax if peak > 0 else 1.0
    return [_clip(round(v / scale), -qmax, qmax) for v in values], scale


def dequantize_symmetric(ints, scale):
    return [q * scale for q in ints]


# ---------- asymmetric: uses a zero-point so the range is [min, max] ----------
def quantize_asymmetric(values, bits=8):
    """Return (ints, scale, zero_point). ints are in [0, 2^bits - 1]."""
    qmax = 2 ** bits - 1
    lo, hi = min(values), max(values)
    scale = (hi - lo) / qmax if hi > lo else 1.0
    zero_point = round(-lo / scale)
    return [_clip(round(v / scale) + zero_point, 0, qmax) for v in values], scale, zero_point


def dequantize_asymmetric(ints, scale, zero_point):
    return [(q - zero_point) * scale for q in ints]


# ---------- per-group scales ----------
def quantize_per_group(values, group_size, bits=8):
    """Quantize each group of group_size values with its own symmetric scale. Return the reconstruction."""
    out = []
    for i in range(0, len(values), group_size):
        group = values[i:i + group_size]
        ints, scale = quantize_symmetric(group, bits)
        out.extend(dequantize_symmetric(ints, scale))
    return out


# ---------- error and memory ----------
def errors(original, restored):
    """Return (mean absolute error, max absolute error)."""
    diffs = [abs(a - b) for a, b in zip(original, restored)]
    return sum(diffs) / len(diffs), max(diffs)


def bytes_per_weight(bits):
    return bits / 8


def memory_gb(params, bits):
    """Weights-only memory in GB (1 GB = 1e9 bytes)."""
    return params * bytes_per_weight(bits) / 1e9


def memory_table(params=7e9, bit_widths=(32, 16, 8, 4)):
    return [(bits, bytes_per_weight(bits), memory_gb(params, bits)) for bits in bit_widths]


def make_weights(n=64, seed=0):
    """Small, bell-shaped 'weights' (mean 0, std 0.05), like a real layer."""
    rng = random.Random(seed)
    return [rng.gauss(0, 0.05) for _ in range(n)]


if __name__ == "__main__":
    w = make_weights(64, seed=0)
    print("first 4 weights:", [round(x, 4) for x in w[:4]])

    ints, scale = quantize_symmetric(w, 8)
    mae, mx = errors(w, dequantize_symmetric(ints, scale))
    print(f"symmetric int8 : scale {scale:.6f} | mean abs err {mae:.6f} | max err {mx:.6f} | scale/2 {scale / 2:.6f}")

    ints_a, scale_a, zp = quantize_asymmetric(w, 8)
    mae, mx = errors(w, dequantize_asymmetric(ints_a, scale_a, zp))
    print(f"asymmetric int8: scale {scale_a:.6f} | zero-point {zp} | mean abs err {mae:.6f} | max err {mx:.6f}")

    for bits in (8, 4):
        ints, scale = quantize_symmetric(w, bits)
        mae, _ = errors(w, dequantize_symmetric(ints, scale))
        print(f"symmetric {bits}-bit: mean abs err {mae:.6f}")

    print("\n7B-parameter model, weights only:")
    for bits, nbytes, gb in memory_table(7e9):
        print(f"  {bits:>2}-bit: {nbytes:>4} bytes/weight -> {gb:>5.1f} GB")

    print("\nOutlier demo (4-bit, 64 weights, one weight set to 2.0):")
    spiky = list(w)
    spiky[0] = 2.0
    one_scale = []
    ints, scale = quantize_symmetric(spiky, 4)
    one_scale = dequantize_symmetric(ints, scale)
    grouped = quantize_per_group(spiky, 8, 4)
    rest = slice(1, None)
    print(f"  one scale for all  : mean abs err on the other 63 weights {errors(spiky[rest], one_scale[rest])[0]:.6f}")
    print(f"  one scale per 8    : mean abs err on the other 63 weights {errors(spiky[rest], grouped[rest])[0]:.6f}")
    zeroed = sum(1 for q in ints[1:] if q == 0)
    print(f"  with one scale, {zeroed} of the other 63 weights rounded to exactly 0")
