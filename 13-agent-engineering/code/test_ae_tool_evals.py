import pytest

import ae_tool_evals as m

EXP = [("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 3})]


def test_perfect_trajectory():
    r = m.score_trajectory(EXP, list(EXP))
    assert r["success"] and r["extra_calls"] == 0 and r["order_ok"]


def test_wrong_argument_fails_success_but_not_tool_accuracy():
    act = [("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 99})]
    r = m.score_trajectory(EXP, act)
    assert r["tool_accuracy"] == 1.0 and r["arg_accuracy"] == pytest.approx(2 / 3) and not r["success"]


def test_wrong_order_detected():
    act = [("b", {"y": 2}), ("a", {"x": 1}), ("c", {"z": 3})]
    assert not m.score_trajectory(EXP, act)["order_ok"]


def test_extra_calls_counted():
    act = [EXP[0], EXP[1], EXP[1], EXP[1], EXP[2]]
    r = m.score_trajectory(EXP, act)
    assert r["extra_calls"] == 2 and r["success"]


def test_missing_tool():
    r = m.score_trajectory(EXP, EXP[:2])
    assert r["tool_accuracy"] == pytest.approx(2 / 3) and not r["success"]


def test_pass_k_family():
    assert m.pass_at_k(10, 8, 1) == pytest.approx(0.8)
    assert m.pass_hat_k(10, 8, 1) == pytest.approx(0.8)
    assert m.pass_hat_k(10, 8, 3) == pytest.approx(56 / 120)
    assert m.pass_hat_k(10, 8, 9) == 0.0
    assert m.pass_at_k(10, 8, 3) > m.pass_hat_k(10, 8, 3)
