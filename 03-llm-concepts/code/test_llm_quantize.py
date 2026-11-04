import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_quantize import (
    bytes_per_weight, dequantize_asymmetric, dequantize_symmetric, errors, make_weights,
    memory_gb, memory_table, quantize_asymmetric, quantize_per_group, quantize_symmetric,
)


def test_symmetric_round_trip_error_bounded_by_half_scale():
    w = make_weights(200, seed=3)
    for bits in (8, 4):
        ints, scale = quantize_symmetric(w, bits)
        _, mx = errors(w, dequantize_symmetric(ints, scale))
        assert mx <= scale / 2 + 1e-12


def test_asymmetric_round_trip_error_bounded_by_scale():
    # the zero-point is itself rounded, so the bound is one full scale step in the worst case
    w = make_weights(200, seed=4)
    ints, scale, zp = quantize_asymmetric(w, 8)
    _, mx = errors(w, dequantize_asymmetric(ints, scale, zp))
    assert mx <= scale + 1e-12


def test_integer_ranges():
    w = make_weights(100, seed=1)
    ints, _ = quantize_symmetric(w, 8)
    assert min(ints) >= -127 and max(ints) <= 127
    ints, _, _ = quantize_asymmetric(w, 8)
    assert min(ints) >= 0 and max(ints) <= 255


def test_symmetric_peak_maps_to_127_and_zero_to_zero():
    ints, scale = quantize_symmetric([0.0, 0.5, -1.0], 8)
    assert ints == [0, 64, -127]
    assert abs(scale - 1.0 / 127) < 1e-12


def test_asymmetric_hand_example():
    ints, scale, zp = quantize_asymmetric([0.0, 1.0, 2.0], 8)
    assert ints == [0, 128, 255] and zp == 0
    assert abs(scale - 2 / 255) < 1e-12


def test_constant_input_does_not_crash():
    ints, scale = quantize_symmetric([0.0, 0.0], 8)
    assert ints == [0, 0]


def test_memory_ratios_exact():
    assert memory_gb(7e9, 16) == 14.0
    assert memory_gb(7e9, 8) == 7.0
    assert memory_gb(7e9, 4) == 3.5
    assert memory_gb(7e9, 32) == 28.0
    assert bytes_per_weight(4) == 0.5
    table = {bits: gb for bits, _, gb in memory_table(7e9)}
    assert table[16] == 2 * table[8] == 4 * table[4]


def test_outlier_hurts_and_groups_fix_it():
    w = make_weights(64, seed=0)
    w[0] = 2.0
    ints, scale = quantize_symmetric(w, 4)
    one = dequantize_symmetric(ints, scale)
    grouped = quantize_per_group(w, 8, 4)
    err_one = errors(w[1:], one[1:])[0]
    err_grouped = errors(w[1:], grouped[1:])[0]
    assert err_grouped < err_one / 3
    assert err_grouped < 0.01


def test_fewer_bits_means_more_error():
    w = make_weights(256, seed=2)
    e8 = errors(w, dequantize_symmetric(*quantize_symmetric(w, 8)))[0]
    e4 = errors(w, dequantize_symmetric(*quantize_symmetric(w, 4)))[0]
    assert e4 > e8
