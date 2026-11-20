import pytest

import fa_speculative as m


def test_boundaries():
    assert m.expected_tokens_per_pass(0, 4) == 1
    assert m.expected_tokens_per_pass(1, 4) == 5


def test_hand_value():
    # a=0.5, k=2: 1 + 0.5 + 0.25
    assert m.expected_tokens_per_pass(0.5, 2) == pytest.approx(1.75)


def test_simulation_matches_formula():
    assert m.simulate(0.7, 4, 40000) == pytest.approx(m.expected_tokens_per_pass(0.7, 4), abs=0.03)


def test_bad_draft_is_slower_than_plain():
    assert m.speedup(0.3, 4, 0.4) < 1


def test_good_draft_is_faster():
    assert m.speedup(0.9, 4, 0.1) > 2


def test_best_k_grows_with_acceptance():
    assert m.best_k(0.9, 0.1) > m.best_k(0.5, 0.1)


def test_rejects_bad_a():
    with pytest.raises(ValueError):
        m.expected_tokens_per_pass(1.5, 3)
