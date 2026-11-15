import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prod_request_sim import PATTERNS, make_traffic, question_text, simulate  # noqa: E402


def test_traffic_is_deterministic_and_increasing():
    a, b = make_traffic("steady", 40, 3), make_traffic("steady", 40, 3)
    assert a == b and len(a) == 40
    assert all(a[i][0] < a[i + 1][0] for i in range(39))
    assert make_traffic("steady", 40, 4) != a


def test_every_request_gets_exactly_one_outcome():
    for name in PATTERNS:
        r = simulate(make_traffic(name, 50, 1))
        s = r["summary"]
        assert s["hits"] + s["llm"] + s["limited"] == s["total"] == 50


def test_cache_first_never_limits_a_hit_so_it_serves_at_least_as_many():
    for name in PATTERNS:
        tr = make_traffic(name, 80, 5)
        lf = simulate(tr, order="limit_first")["summary"]
        cf = simulate(tr, order="cache_first")["summary"]
        assert cf["hits"] + cf["llm"] >= lf["hits"] + lf["llm"]
        assert cf["limited"] <= lf["limited"]


def test_viral_question_has_a_higher_hit_rate_than_unique_questions():
    big = simulate(make_traffic("popular", 100, 2), rate=100, capacity=100)["summary"]
    small = simulate(make_traffic("unique", 100, 2), rate=100, capacity=100)["summary"]
    assert big["hit_rate"] > small["hit_rate"]


def test_zero_ttl_style_cache_means_no_hits():
    s = simulate(make_traffic("popular", 30, 1), ttl=0.0, rate=100, capacity=100)["summary"]
    assert s["hits"] == 0 and s["llm"] == 30


def test_money_adds_up():
    s = simulate(make_traffic("steady", 60, 9), price=0.02)["summary"]
    assert abs(s["cost"] - s["llm"] * 0.02) < 1e-12 and abs(s["saved"] - s["hits"] * 0.02) < 1e-12


def test_unique_pattern_really_has_many_questions():
    ids = {q for _, q in make_traffic("unique", 200, 1)}
    assert len(ids) > 40 and question_text(79) == "Question number 80"
