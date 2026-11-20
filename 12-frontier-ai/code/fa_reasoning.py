"""Test-time compute: majority voting (self-consistency), stdlib only.

If one sampled answer is right with probability p, and wrong answers are
spread over several values, then voting over n samples is right more often.
We compute this exactly (no randomness) and also with a seeded simulation.

Run:  python fa_reasoning.py
"""
import random
from math import comb


def majority_correct_binary(p, n):
    """P(majority right) when each sample is right with prob p and wrong otherwise
    (two possible answers). n must be odd."""
    if n % 2 == 0:
        raise ValueError("n must be odd for a binary vote")
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def simulate_vote(p, n, wrong_choices, trials, seed=0):
    """Seeded simulation: wrong answers are uniform over `wrong_choices` values.
    Ties are broken by the smallest answer id, and the right answer is id 0
    (so ties favour the right answer; the count of such ties is small)."""
    rng = random.Random(seed)
    right = 0
    for _ in range(trials):
        votes = {}
        for _ in range(n):
            a = 0 if rng.random() < p else rng.randint(1, wrong_choices)
            votes[a] = votes.get(a, 0) + 1
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        right += best == 0
    return right / trials


def cost_multiplier(n, tokens_per_sample, base_tokens):
    """Tokens spent relative to one short answer."""
    return n * tokens_per_sample / base_tokens


if __name__ == "__main__":
    print("p(single)=0.60, two possible answers: exact majority vote")
    for n in (1, 3, 5, 11, 21):
        print(f"  n={n:>2}  P(correct)={majority_correct_binary(0.6, n):.4f}")
    print("p(single)=0.40, wrong answers spread over 5 values (simulated, 20000 trials)")
    for n in (1, 5, 11, 21):
        print(f"  n={n:>2}  P(correct)={simulate_vote(0.4, n, 5, 20000):.4f}")
    print("p(single)=0.45, only ONE wrong value (a correlated mistake): voting hurts")
    for n in (1, 5, 21):
        print(f"  n={n:>2}  P(correct)={majority_correct_binary(0.45, n) if n % 2 else 0:.4f}")
    print("cost: 21 samples of 800 tokens vs one 100-token answer:", cost_multiplier(21, 800, 100), "x")
