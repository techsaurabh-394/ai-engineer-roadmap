import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_rag_metrics import (  # noqa: E402
    SAMPLE,
    answer_relevancy,
    claims,
    context_precision,
    context_recall,
    evaluate,
    faithfulness,
)


def test_context_precision_rewards_good_ranking():
    assert context_precision([1, 1, 0]) == 1.0
    assert abs(context_precision([1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-9
    assert context_precision([0, 0, 1]) < context_precision([1, 0, 0])
    assert context_precision([0, 0]) == 0.0


def test_claims_split():
    assert claims("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_context_recall_full_and_partial():
    ctx = SAMPLE["contexts"]
    assert context_recall(SAMPLE["reference"], ctx) == 1.0
    assert context_recall("Refresh tokens belong in an httpOnly cookie. The moon is made of cheese.", ctx) == 0.5


def test_faithfulness_catches_hallucinated_claim():
    # second claim ("expire after exactly 30 days") is not in any context
    assert faithfulness(SAMPLE["answer"], SAMPLE["contexts"]) == 0.5
    assert faithfulness("Access tokens should be short lived.", SAMPLE["contexts"]) == 1.0


def test_answer_relevancy():
    q = SAMPLE["question"]
    assert answer_relevancy(q, "Refresh tokens are stored in an httpOnly cookie.") > answer_relevancy(q, "Use Docker.")


def test_evaluate_returns_all_metrics():
    report = evaluate(SAMPLE)
    assert set(report) == {"context_precision", "context_recall", "faithfulness", "answer_relevancy"}
    assert all(0.0 <= v <= 1.0 for v in report.values())
