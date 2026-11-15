"""Evaluate a fine-tuned model against its base on a held-out set. Standard library only, seeded.

The two "models" are plain dicts {prompt: answer}, so every number is hand-checkable.
Also: a bootstrap confidence interval, win-rate with ties, a regression check, and the DPO loss.

Run:  python ft_eval.py
"""
import math
import random
import re


# ---------------------------------------------------------------- metrics
def normalize(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def exact_match(pred, gold):
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred, gold):
    """Harmonic mean of token precision and recall (bag of words, counts respected)."""
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    pool = {}
    for t in g:
        pool[t] = pool.get(t, 0) + 1
    common = 0
    for t in p:
        if pool.get(t, 0) > 0:
            pool[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score_model(model, heldout, metric):
    """Per-example scores of `model` (dict prompt -> answer) on `heldout` [(prompt, gold)]."""
    return [metric(model.get(p, ""), gold) for p, gold in heldout]


def mean(xs):
    return sum(xs) / len(xs)


def win_rate(scores_a, scores_b):
    """Fraction of examples where A beats B; ties count half. Returns (rate, wins, ties, losses)."""
    wins = sum(a > b for a, b in zip(scores_a, scores_b))
    ties = sum(a == b for a, b in zip(scores_a, scores_b))
    losses = len(scores_a) - wins - ties
    return (wins + 0.5 * ties) / len(scores_a), wins, ties, losses


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def paired_bootstrap_diff(scores_a, scores_b, n_resamples=2000, alpha=0.05, seed=0):
    """CI for mean(A) - mean(B) resampling the SAME examples for both (paired)."""
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    return bootstrap_ci(diffs, n_resamples, alpha, seed)


# ---------------------------------------------------------------- regression check
def regressions(base_scores, tuned_scores, heldout, tol=0.0):
    """Examples the base model got right (or better) that the tuned model now gets worse."""
    return [heldout[i][0] for i, (b, t) in enumerate(zip(base_scores, tuned_scores)) if t < b - tol]


# ---------------------------------------------------------------- preference tuning
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """-log sigmoid(beta * ((logpi_w - logpref_w) - (logpi_l - logpref_l)))

    w = chosen (winner) answer, l = rejected (loser). logp_* are summed log-probabilities of the
    whole answer under the policy being trained; ref_logp_* under the frozen reference model.
    """
    margin = beta * ((logp_w - ref_logp_w) - (logp_l - ref_logp_l))
    return -math.log(sigmoid(margin)), margin


# ---------------------------------------------------------------- toy data
HELDOUT = [
    ("Refund window?", "30 days from delivery"),
    ("Do you ship abroad?", "Yes, to 12 countries"),
    ("Opening hours?", "Monday to Friday 9 to 5"),
    ("Can I cancel an order?", "Yes before it ships"),
    ("Is gift wrap free?", "No it costs 3"),
    ("Student discount?", "Yes 10 percent"),
    ("Delivery time?", "3 to 5 working days"),
    ("Accepted cards?", "Visa and Mastercard"),
    ("Change my address?", "Yes before dispatch"),
    ("Do you have an app?", "Yes for iOS and Android"),
    ("Is there a loyalty scheme?", "Yes 1 point per 1 spent"),
    ("Can I pay by PayPal?", "Yes"),
]

BASE = {
    "Refund window?": "Refunds are possible, please check our policy page",
    "Do you ship abroad?": "Yes, to 12 countries",
    "Opening hours?": "We are open on weekdays",
    "Can I cancel an order?": "Yes before it ships",
    "Is gift wrap free?": "Gift wrap is available",
    "Student discount?": "Yes 10 percent",
    "Delivery time?": "Delivery takes a few days",
    "Accepted cards?": "Visa and Mastercard",
    "Change my address?": "Contact support",
    "Do you have an app?": "Yes for iOS",
    "Is there a loyalty scheme?": "Maybe, ask support",
    "Can I pay by PayPal?": "Yes",
}

TUNED = {
    "Refund window?": "30 days from delivery",
    "Do you ship abroad?": "Yes, to 12 countries",
    "Opening hours?": "Monday to Friday 9 to 5",
    "Can I cancel an order?": "Yes before it ships",
    "Is gift wrap free?": "No it costs 3",
    "Student discount?": "Yes 10 percent",
    "Delivery time?": "3 to 5 working days",
    "Accepted cards?": "Visa Mastercard and Amex",   # a regression: adds a wrong card
    "Change my address?": "Yes before dispatch",
    "Do you have an app?": "Yes for iOS and Android",
    "Is there a loyalty scheme?": "Yes 1 point per 1 spent",
    "Can I pay by PayPal?": "Yes",
}

if __name__ == "__main__":
    print(f"held-out examples: {len(HELDOUT)}")
    for name, metric in (("exact match", exact_match), ("token F1", token_f1)):
        b = score_model(BASE, HELDOUT, metric)
        t = score_model(TUNED, HELDOUT, metric)
        print(f"{name:11s}: base {mean(b):.3f} | tuned {mean(t):.3f}")

    b_em, t_em = score_model(BASE, HELDOUT, exact_match), score_model(TUNED, HELDOUT, exact_match)
    b_f1, t_f1 = score_model(BASE, HELDOUT, token_f1), score_model(TUNED, HELDOUT, token_f1)
    rate, w, ti, lo = win_rate(t_f1, b_f1)
    print(f"win rate (tuned vs base, token F1): {rate:.3f}  wins {w}, ties {ti}, losses {lo}")

    lo_ci, hi_ci = bootstrap_ci(t_em, seed=1)
    print(f"tuned exact match 95% CI: [{lo_ci:.3f}, {hi_ci:.3f}]")
    lo_d, hi_d = paired_bootstrap_diff(t_em, b_em, seed=1)
    print(f"paired diff (tuned - base) exact match 95% CI: [{lo_d:.3f}, {hi_d:.3f}]")
    print("regressions (tuned worse than base on F1):", regressions(b_f1, t_f1, HELDOUT))

    small = HELDOUT[:4]
    sb, st = score_model(BASE, small, exact_match), score_model(TUNED, small, exact_match)
    print(f"with only {len(small)} examples: diff CI = {paired_bootstrap_diff(st, sb, seed=1)}")

    print("\nDPO loss, beta=0.1")
    for label, args in [("policy = reference (nothing learned yet)", (-10.0, -12.0, -10.0, -12.0)),
                        ("policy raised chosen, lowered rejected ", (-8.0, -15.0, -10.0, -12.0)),
                        ("policy moved the wrong way             ", (-13.0, -9.0, -10.0, -12.0))]:
        loss, margin = dpo_loss(*args)
        print(f"  {label}: margin {margin:+.2f} -> loss {loss:.4f}")
