import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import replace  # noqa: E402

import ft_decision as d  # noqa: E402


def test_token_counts():
    assert d.tokens_per_month(200_000, 1_200, 150) == (240_000_000, 30_000_000)


def test_monthly_costs_by_hand():
    s = d.Scenario()
    before, after = d.costs(s)
    # 240M * 3/1M + 30M * 15/1M = 720 + 450
    assert abs(before - 1170.0) < 1e-9
    # 1.5 * (30M * 3/1M + 30M * 15/1M) + 50 = 1.5 * 540 + 50
    assert abs(after - 860.0) < 1e-9


def test_break_even_months():
    s = d.Scenario()
    assert abs(d.months_to_break_even(s) - 2400 / 310) < 1e-9


def test_never_breaks_even_when_prompt_is_already_short():
    s = replace(d.Scenario(), prompt_tokens_before=200, prompt_tokens_after=150)
    assert d.months_to_break_even(s) is None


def test_low_volume_never_pays_back():
    assert d.months_to_break_even(replace(d.Scenario(), requests_per_month=10_000)) is None


def test_break_even_requests_is_the_threshold():
    s = d.Scenario()
    n = d.break_even_requests(s, 12)
    assert d.months_to_break_even(replace(s, requests_per_month=n)) <= 12
    assert d.months_to_break_even(replace(s, requests_per_month=n - 1)) > 12


def test_more_volume_means_faster_payback():
    a = d.months_to_break_even(replace(d.Scenario(), requests_per_month=300_000))
    b = d.months_to_break_even(replace(d.Scenario(), requests_per_month=600_000))
    assert b < a


def test_recommend_order_of_preference():
    assert d.recommend(False, False, False, True, True, True) == "prompting"
    assert d.recommend(False, True, False, False, True, True) == "RAG"
    assert d.recommend(False, False, True, False, True, True) == "RAG"
    assert d.recommend(False, False, False, False, True, True) == "fine-tuning"
    assert d.recommend(False, False, False, False, False, True).startswith("prompting")
