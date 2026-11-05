import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import llm_moe_scaling as ms  # noqa: E402


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_router_weights_sum_to_one_and_count(k):
    ch = ms.route([0.3, 2.0, -1.0, 1.5, 0.0], k)
    assert len(ch) == k
    assert math.isclose(sum(w for _, w in ch), 1.0)
    assert len({e for e, _ in ch}) == k


def test_router_picks_highest_scores():
    ch = ms.route([0.1, 5.0, 0.2, 4.0], 2)
    assert [e for e, _ in ch] == [1, 3]
    assert ch[0][1] > ch[1][1]


def test_router_validates_k():
    with pytest.raises(ValueError):
        ms.route([1.0, 2.0], 3)
    with pytest.raises(ValueError):
        ms.route([1.0, 2.0], 0)


def test_moe_layer_runs_only_k_experts():
    rw = [[1, 0], [0, 1], [1, 1], [-1, 0]]
    experts = ms.make_experts(4, 2)
    out, ch = ms.moe_layer([1.0, 2.0], rw, experts, k=2)
    assert len(ch) == 2 and len(out) == 2
    assert math.isclose(sum(w for _, w in ch), 1.0)


def test_active_less_than_total():
    total, active = ms.moe_param_counts(64, 256, num_experts=8, top_k=2)
    assert active < total
    per = ms.ffn_params(64, 256)
    assert total - active == 6 * per
    t1, a1 = ms.moe_param_counts(64, 256, 8, 8)
    assert t1 == a1  # top_k == E -> dense


def test_imbalance_metric():
    assert math.isclose(ms.imbalance([0.25] * 4), 1.0)
    assert math.isclose(ms.imbalance([1.0, 0, 0, 0]), 4.0)
    ch = [[(0, 0.6), (1, 0.4)], [(0, 0.5), (2, 0.5)]]
    load = ms.expert_load(ch, 4)
    assert math.isclose(sum(load), 1.0) and load[0] == 0.5


def test_balance_loss_minimum_when_uniform():
    u = [0.25] * 4
    assert math.isclose(ms.balance_loss(u, u, 4), 1.0)
    skew = [0.7, 0.1, 0.1, 0.1]
    assert ms.balance_loss(skew, skew, 4) > 1.0


def test_flops_arithmetic():
    assert ms.training_flops(1e9, 20e9) == 6 * 1e9 * 20e9 == 1.2e20
    assert ms.chinchilla_tokens(7e9) == 140e9


def test_loglog_fit_recovers_exponent():
    ns = [1e6, 1e7, 1e8, 1e9]
    ys = [ms.power_law(n, 10.0, 0.076) for n in ns]
    a, alpha = ms.fit_power_law(ns, ys)
    assert math.isclose(alpha, 0.076, rel_tol=1e-9)
    assert math.isclose(a, 10.0, rel_tol=1e-9)
