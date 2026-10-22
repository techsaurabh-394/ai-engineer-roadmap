import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_harness import GOLDEN, evaluate  # noqa: E402
from mini_rag import DOCS, Index  # noqa: E402


def test_golden_set_scores_well():
    report = evaluate(Index(DOCS), GOLDEN, k=3)
    assert report["hit@k"] >= 0.75


def test_bad_case_is_reported_as_miss():
    bad = [{"q": "What is the capital of France?", "doc": "docker.txt", "must": "paris"}]
    report = evaluate(Index(DOCS), bad, k=3)
    assert report["hit@k"] == 0 and report["misses"]
