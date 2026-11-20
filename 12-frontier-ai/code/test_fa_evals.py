import pytest

import fa_evals as m


def test_pass_at_1_is_c_over_n():
    assert m.pass_at_k(10, 3, 1) == pytest.approx(0.3)


def test_pass_at_k_hand_value():
    # 1 - C(7,3)/C(10,3) = 1 - 35/120
    assert m.pass_at_k(10, 3, 3) == pytest.approx(1 - 35 / 120)


def test_pass_at_k_monotonic_and_capped():
    vals = [m.pass_at_k(10, 3, k) for k in (1, 3, 5, 10)]
    assert vals == sorted(vals) and vals[-1] == 1.0


def test_zero_correct_is_zero():
    assert m.pass_at_k(10, 0, 5) == 0.0


def test_bad_args():
    with pytest.raises(ValueError):
        m.pass_at_k(5, 6, 1)


def test_bootstrap_contains_mean_and_is_seeded():
    outs = [1] * 36 + [0] * 14
    lo, hi = m.bootstrap_ci(outs)
    assert lo < 0.72 < hi
    assert (lo, hi) == m.bootstrap_ci(outs)


def test_contamination_detects_overlap():
    t = "write a function that returns the second largest number in a list of integers"
    assert m.contamination(t, ["blog: write a function that returns the second largest number in a list"]) > 0.3
    assert m.contamination(t, ["pasta recipe with tomatoes and basil and garlic and oil"]) == 0.0
