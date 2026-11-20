"""Preference tuning: the Bradley-Terry reward loss and the DPO loss, stdlib only.

Reward model loss for one pair:   -log sigmoid(r_chosen - r_rejected)
DPO loss for one pair:            -log sigmoid(beta * ((lp_c - ref_c) - (lp_r - ref_r)))
where lp_* are log-probabilities under the model being trained and ref_* under
the frozen reference model. Numbers below are made up for illustration.

Run:  python fa_preference.py
"""
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def reward_loss(r_chosen, r_rejected):
    return -math.log(sigmoid(r_chosen - r_rejected))


def dpo_loss(lp_c, lp_r, ref_c, ref_r, beta=0.1):
    margin = beta * ((lp_c - ref_c) - (lp_r - ref_r))
    return -math.log(sigmoid(margin))


def implicit_reward(lp, ref, beta=0.1):
    return beta * (lp - ref)


def preference_accuracy(pairs):
    """Fraction of (r_chosen, r_rejected) pairs the reward model ranks correctly."""
    return sum(c > r for c, r in pairs) / len(pairs)


if __name__ == "__main__":
    print("reward loss, equal rewards     :", round(reward_loss(1.0, 1.0), 4), "(= ln 2)")
    print("reward loss, chosen +2 higher  :", round(reward_loss(2.0, 0.0), 4))
    print("reward loss, chosen 2 LOWER    :", round(reward_loss(0.0, 2.0), 4))
    print("DPO at the start (model == ref):", round(dpo_loss(-10, -12, -10, -12), 4), "(= ln 2)")
    print("DPO after chosen up 5, rejected down 5 (beta 0.1):", round(dpo_loss(-5, -17, -10, -12), 4))
    print("DPO same shift, beta 0.5       :", round(dpo_loss(-5, -17, -10, -12, beta=0.5), 4))
    print("reward-model accuracy on 4 pairs:", preference_accuracy([(2, 1), (0, 1), (3, -1), (1, 0.5)]))
