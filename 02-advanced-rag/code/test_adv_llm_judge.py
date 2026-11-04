import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_llm_judge import (  # noqa: E402
    HUMAN,
    JUDGE,
    LONG_WEAK,
    Q,
    SHORT_GOOD,
    ScriptedJudge,
    agreement,
    cohens_kappa,
    debiased_pairwise,
    pairwise_judge,
)

KW = ["httpOnly", "cookie"]


def test_unbiased_judge_picks_good_answer_in_both_orders():
    j = ScriptedJudge(KW)
    assert pairwise_judge(j, Q, SHORT_GOOD, LONG_WEAK) == "A"
    assert pairwise_judge(j, Q, LONG_WEAK, SHORT_GOOD) == "B"
    assert debiased_pairwise(j, Q, SHORT_GOOD, LONG_WEAK) == "A"


def test_position_biased_judge_flips_when_order_swapped():
    j = ScriptedJudge(KW, position_bonus=3.0)
    first = pairwise_judge(j, Q, SHORT_GOOD, LONG_WEAK)  # good answer shown first
    swapped = pairwise_judge(j, Q, LONG_WEAK, SHORT_GOOD)  # good answer shown second
    # both runs say "A" = whatever came first, so the verdict is about position, not quality
    assert first == "A" and swapped == "A"


def test_debiased_returns_tie_when_orders_disagree():
    j = ScriptedJudge(KW, position_bonus=3.0)
    assert debiased_pairwise(j, Q, SHORT_GOOD, LONG_WEAK) == "tie"


def test_verbosity_bias_prefers_long_answer_and_debias_does_not_fix_it():
    j = ScriptedJudge(KW, length_bonus=0.5)
    assert pairwise_judge(j, Q, SHORT_GOOD, LONG_WEAK) == "B"
    assert debiased_pairwise(j, Q, SHORT_GOOD, LONG_WEAK) == "B"  # wrong, but consistent


def test_tie_break_goes_to_first_slot():
    j = ScriptedJudge(KW)
    assert pairwise_judge(j, Q, SHORT_GOOD, SHORT_GOOD) == "A"
    assert debiased_pairwise(j, Q, SHORT_GOOD, SHORT_GOOD) == "tie"


def test_agreement_hand_value():
    assert abs(agreement(JUDGE, HUMAN) - 0.7) < 1e-9


def test_kappa_identical_labels_is_one():
    labels = [1, 0, 1, 1, 0, 0, 1]
    assert abs(cohens_kappa(labels, labels) - 1.0) < 1e-9


def test_kappa_known_example():
    # po = 0.7; judge yes 6/10, human yes 7/10; pe = 0.6*0.7 + 0.4*0.3 = 0.54
    expected = (0.7 - 0.54) / (1 - 0.54)
    assert abs(cohens_kappa(JUDGE, HUMAN) - expected) < 1e-9
    assert abs(cohens_kappa(JUDGE, HUMAN) - 0.347826) < 1e-6


def test_kappa_chance_level_is_zero():
    j = [1, 1, 0, 0]
    h = [1, 0, 1, 0]
    assert abs(cohens_kappa(j, h)) < 1e-9
