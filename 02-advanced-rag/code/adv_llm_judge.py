"""Toy LLM-as-a-judge helpers. Standard library only, no network.

A real judge is an LLM prompted with a rubric. Here a small scripted judge
stands in for it, and it is built to be biased on purpose so the biases and
the fixes can be seen and tested:

* position bias:  a bonus for whichever answer is shown first (slot "A"),
                  and ties are broken in favour of "A".
* verbosity bias: a bonus for every word, so long answers score higher.
"""

import re


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class ScriptedJudge:
    """score = 2 * (rubric keywords found) + length_bonus * (word count) + position_bonus if shown first."""

    def __init__(self, keywords, position_bonus=0.0, length_bonus=0.0):
        self.keywords = {k.lower() for k in keywords}
        self.position_bonus = position_bonus
        self.length_bonus = length_bonus

    def score(self, question, answer, first=False):
        ws = words(answer)
        hits = len(self.keywords & set(ws))
        s = 2.0 * hits + self.length_bonus * len(ws)
        if first:
            s += self.position_bonus
        return s


def pairwise_judge(judge, question, a, b):
    """One call: answer `a` is shown first, `b` second. Returns 'A', 'B' or 'tie'.
    Ties between the two scores are broken toward 'A', which is itself a bias."""
    sa = judge.score(question, a, first=True)
    sb = judge.score(question, b, first=False)
    return "B" if sb > sa else "A"


def debiased_pairwise(judge, question, a, b):
    """Run both orders. Keep the verdict only if the two runs agree on the same
    ANSWER. Otherwise return 'tie'."""
    v1 = pairwise_judge(judge, question, a, b)  # a first
    v2 = pairwise_judge(judge, question, b, a)  # b first
    v2_as_ab = {"A": "B", "B": "A"}[v2]  # map back to the original labels
    return v1 if v1 == v2_as_ab else "tie"


def agreement(judge_labels, human_labels):
    """Share of items where judge and human gave the same label."""
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("need two non-empty lists of equal length")
    same = sum(j == h for j, h in zip(judge_labels, human_labels))
    return same / len(judge_labels)


def cohens_kappa(judge_labels, human_labels):
    """kappa = (po - pe) / (1 - pe)
    po = observed agreement.
    pe = agreement expected by chance = sum over labels of P(judge says it) * P(human says it).
    1.0 = perfect, 0.0 = no better than chance."""
    po = agreement(judge_labels, human_labels)
    n = len(judge_labels)
    pe = 0.0
    for label in set(judge_labels) | set(human_labels):
        pe += (judge_labels.count(label) / n) * (human_labels.count(label) / n)
    if pe == 1.0:  # both always give one same label: kappa is undefined, call it perfect
        return 1.0
    return (po - pe) / (1 - pe)


# Hand-checkable example: 10 items, 1 = "good answer", 0 = "bad answer".
# both 1: 5, judge 1 / human 0: 1, judge 0 / human 1: 2, both 0: 2
JUDGE = [1] * 5 + [1] + [0] * 2 + [0] * 2
HUMAN = [1] * 5 + [0] + [1] * 2 + [0] * 2

Q = "How should I store refresh tokens?"
SHORT_GOOD = "Use an httpOnly cookie."
LONG_WEAK = "Well, there are many things to consider here, and it depends on your app, but a cookie is one option."

if __name__ == "__main__":
    biased = ScriptedJudge(["httpOnly", "cookie"], position_bonus=3.0)
    print("pairwise (good first):  ", pairwise_judge(biased, Q, SHORT_GOOD, LONG_WEAK))
    print("pairwise (good second): ", pairwise_judge(biased, Q, LONG_WEAK, SHORT_GOOD))
    print("debiased:               ", debiased_pairwise(biased, Q, SHORT_GOOD, LONG_WEAK))
    wordy = ScriptedJudge(["httpOnly", "cookie"], length_bonus=0.5)
    print("verbosity-biased judge, good first:", pairwise_judge(wordy, Q, SHORT_GOOD, LONG_WEAK))
    print("agreement:", agreement(JUDGE, HUMAN))
    print("kappa:", round(cohens_kappa(JUDGE, HUMAN), 4))
