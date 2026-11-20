import pytest

import fa_reasoning as m


def test_n1_equals_p():
    assert m.majority_correct_binary(0.6, 1) == pytest.approx(0.6)


def test_n3_hand_value():
    # 3 * 0.6^2 * 0.4 + 0.6^3 = 0.432 + 0.216
    assert m.majority_correct_binary(0.6, 3) == pytest.approx(0.648)


def test_voting_helps_when_p_above_half_and_hurts_below():
    assert m.majority_correct_binary(0.6, 21) > m.majority_correct_binary(0.6, 3) > 0.6
    assert m.majority_correct_binary(0.45, 21) < 0.45


def test_even_n_rejected():
    with pytest.raises(ValueError):
        m.majority_correct_binary(0.6, 4)


def test_spread_wrong_answers_lets_weak_model_win():
    one = m.simulate_vote(0.4, 1, 5, 4000)
    many = m.simulate_vote(0.4, 21, 5, 4000)
    assert many > one + 0.3


def test_simulation_is_seeded():
    assert m.simulate_vote(0.5, 5, 3, 500, seed=1) == m.simulate_vote(0.5, 5, 3, 500, seed=1)


def test_cost_multiplier():
    assert m.cost_multiplier(21, 800, 100) == 168
