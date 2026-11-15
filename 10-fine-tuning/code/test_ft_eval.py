import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ft_eval as e  # noqa: E402


def test_exact_match_ignores_case_and_punctuation():
    assert e.exact_match("Yes, before it ships!", "yes before it ships") == 1.0
    assert e.exact_match("No", "Yes") == 0.0


def test_token_f1_by_hand():
    # pred 3 tokens, gold 2 tokens, 2 in common: P = 2/3, R = 1 -> F1 = 0.8
    assert abs(e.token_f1("visa mastercard amex", "visa mastercard") - 0.8) < 1e-12
    assert e.token_f1("", "") == 1.0
    assert e.token_f1("a", "b") == 0.0
    assert e.token_f1("a a", "a") == abs(2 * 0.5 * 1 / 1.5)  # repeated tokens only match as often as in gold


def test_model_scores():
    b = e.score_model(e.BASE, e.HELDOUT, e.exact_match)
    t = e.score_model(e.TUNED, e.HELDOUT, e.exact_match)
    assert round(e.mean(b), 3) == 0.417 and round(e.mean(t), 3) == 0.917
    assert round(e.mean(e.score_model(e.BASE, e.HELDOUT, e.token_f1)), 3) == 0.496
    assert round(e.mean(e.score_model(e.TUNED, e.HELDOUT, e.token_f1)), 3) == 0.988


def test_win_rate_counts_ties_as_half():
    rate, w, t, l = e.win_rate([1, 1, 0, 0], [0, 1, 1, 0])
    assert (w, t, l) == (1, 2, 1) and rate == 0.5
    b = e.score_model(e.BASE, e.HELDOUT, e.token_f1)
    tu = e.score_model(e.TUNED, e.HELDOUT, e.token_f1)
    assert e.win_rate(tu, b)[1:] == (7, 4, 1)


def test_regression_detection():
    b = e.score_model(e.BASE, e.HELDOUT, e.token_f1)
    t = e.score_model(e.TUNED, e.HELDOUT, e.token_f1)
    assert e.regressions(b, t, e.HELDOUT) == ["Accepted cards?"]


def test_bootstrap_is_seeded_and_contains_the_mean():
    t = e.score_model(e.TUNED, e.HELDOUT, e.exact_match)
    ci1, ci2 = e.bootstrap_ci(t, seed=1), e.bootstrap_ci(t, seed=1)
    assert ci1 == ci2
    assert ci1[0] <= e.mean(t) <= ci1[1]
    assert (round(ci1[0], 3), round(ci1[1], 3)) == (0.75, 1.0)


def test_paired_diff_ci_excludes_zero_on_12_but_not_on_4():
    b = e.score_model(e.BASE, e.HELDOUT, e.exact_match)
    t = e.score_model(e.TUNED, e.HELDOUT, e.exact_match)
    lo, hi = e.paired_bootstrap_diff(t, b, seed=1)
    assert lo > 0 and hi > lo
    lo4, _ = e.paired_bootstrap_diff(t[:4], b[:4], seed=1)
    assert lo4 == 0.0


def test_bootstrap_of_constant_is_a_point():
    assert e.bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_sigmoid_is_stable_and_symmetric():
    assert e.sigmoid(0) == 0.5
    assert abs(e.sigmoid(2) + e.sigmoid(-2) - 1) < 1e-12
    assert e.sigmoid(-1000) == 0.0 and e.sigmoid(1000) == 1.0


def test_dpo_loss_at_start_is_ln2():
    loss, margin = e.dpo_loss(-10, -12, -10, -12)
    assert margin == 0 and abs(loss - math.log(2)) < 1e-12


def test_dpo_worked_numbers():
    # margin = 0.1 * ((-8 - -10) - (-15 - -12)) = 0.1 * (2 - (-3)) = 0.5
    loss, margin = e.dpo_loss(-8, -15, -10, -12, beta=0.1)
    assert abs(margin - 0.5) < 1e-12
    assert abs(loss - (-math.log(1 / (1 + math.exp(-0.5))))) < 1e-12
    assert round(loss, 4) == 0.4741
    wrong, m2 = e.dpo_loss(-13, -9, -10, -12)
    assert round(m2, 2) == -0.6 and round(wrong, 4) == 1.0375 and wrong > math.log(2) > loss


def test_dpo_beta_scales_the_margin():
    _, m1 = e.dpo_loss(-8, -15, -10, -12, beta=0.1)
    _, m2 = e.dpo_loss(-8, -15, -10, -12, beta=0.2)
    assert abs(m2 - 2 * m1) < 1e-12
