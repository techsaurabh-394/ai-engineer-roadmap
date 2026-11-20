import math

import pytest

import fa_preference as m


def test_equal_rewards_give_ln2():
    assert m.reward_loss(1.0, 1.0) == pytest.approx(math.log(2))


def test_loss_falls_when_ranking_is_right():
    assert m.reward_loss(2, 0) < m.reward_loss(0, 0) < m.reward_loss(0, 2)


def test_dpo_starts_at_ln2():
    assert m.dpo_loss(-10, -12, -10, -12) == pytest.approx(math.log(2))


def test_dpo_improves_when_chosen_gains_on_rejected():
    assert m.dpo_loss(-5, -17, -10, -12) < math.log(2)


def test_beta_scales_the_margin():
    assert m.dpo_loss(-5, -17, -10, -12, beta=0.5) < m.dpo_loss(-5, -17, -10, -12, beta=0.1)


def test_dpo_margin_hand_value():
    # margin = 0.1 * ((5) - (-5)) = 1.0
    assert m.dpo_loss(-5, -17, -10, -12) == pytest.approx(-math.log(m.sigmoid(1.0)))


def test_accuracy():
    assert m.preference_accuracy([(2, 1), (0, 1), (3, -1), (1, 0.5)]) == 0.75


def test_sigmoid_is_stable_for_large_inputs():
    assert m.sigmoid(-1000) == 0.0 and m.sigmoid(1000) == 1.0
