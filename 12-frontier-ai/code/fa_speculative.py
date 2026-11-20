"""Speculative decoding: expected speed-up arithmetic and an exact toy simulator.

A small DRAFT model guesses k tokens. The big TARGET model checks all k in one
pass and keeps the longest correct prefix, plus one token of its own.
If each draft token is accepted with probability a (independently), the
expected tokens produced per target pass is (1 - a^(k+1)) / (1 - a).

Run:  python fa_speculative.py
"""
import random


def expected_tokens_per_pass(a, k):
    if not 0 <= a <= 1:
        raise ValueError("a must be in [0, 1]")
    if a == 1:
        return k + 1
    return (1 - a ** (k + 1)) / (1 - a)


def speedup(a, k, draft_cost):
    """Wall-clock speed-up over plain decoding. draft_cost = one draft step / one target step."""
    per_round = k * draft_cost + 1
    return expected_tokens_per_pass(a, k) / per_round


def best_k(a, draft_cost, kmax=16):
    return max(range(1, kmax + 1), key=lambda k: speedup(a, k, draft_cost))


def simulate(a, k, rounds, seed=0):
    """Average tokens per target pass in a seeded simulation."""
    rng = random.Random(seed)
    total = 0
    for _ in range(rounds):
        n = 0
        while n < k and rng.random() < a:
            n += 1
        total += n + 1          # accepted drafts + the target's own token
    return total / rounds


if __name__ == "__main__":
    print("acceptance a, draft length k=4: expected tokens per target pass")
    for a in (0.5, 0.7, 0.9):
        print(f"  a={a}: {expected_tokens_per_pass(a, 4):.3f}   (simulated {simulate(a, 4, 50000):.3f})")
    print("speed-up with a draft that costs 10 percent of a target step (k=4)")
    for a in (0.5, 0.7, 0.9):
        print(f"  a={a}: {speedup(a, 4, 0.1):.2f}x   best k = {best_k(a, 0.1)}")
    print("a poor draft (a=0.3) that costs 40 percent:", f"{speedup(0.3, 4, 0.4):.2f}x", "(slower than plain)")
