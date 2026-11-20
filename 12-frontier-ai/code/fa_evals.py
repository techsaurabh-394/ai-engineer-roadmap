"""Evaluating generative models: pass@k, bootstrap confidence interval, contamination check.

pass@k (Chen et al., 2021) unbiased estimator: with n samples of which c are correct,
    pass@k = 1 - C(n - c, k) / C(n, k)

Run:  python fa_evals.py
"""
import random
from math import comb


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def bootstrap_ci(outcomes, rounds=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of 0/1 outcomes."""
    rng = random.Random(seed)
    n = len(outcomes)
    means = sorted(sum(rng.choice(outcomes) for _ in range(n)) / n for _ in range(rounds))
    return means[int(rounds * alpha / 2)], means[int(rounds * (1 - alpha / 2)) - 1]


def ngrams(text, n=5):
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def contamination(test_item, train_corpus, n=5):
    """Fraction of the test item's n-grams that also appear in the training text."""
    t = ngrams(test_item, n)
    if not t:
        return 0.0
    seen = set()
    for doc in train_corpus:
        seen |= ngrams(doc, n)
    return len(t & seen) / len(t)


if __name__ == "__main__":
    print("n=10 samples, c=3 correct")
    for k in (1, 3, 5, 10):
        print(f"  pass@{k} = {pass_at_k(10, 3, k):.4f}")
    outs = [1] * 36 + [0] * 14      # 72 percent on 50 questions
    lo, hi = bootstrap_ci(outs)
    print(f"50 questions, 72 percent correct: 95 percent CI = [{lo:.2f}, {hi:.2f}]")
    test = "write a function that returns the second largest number in a list of integers"
    train = ["blog: write a function that returns the second largest number in a list", "unrelated text about cooking pasta"]
    print("contamination of that test item:", round(contamination(test, train), 3))
