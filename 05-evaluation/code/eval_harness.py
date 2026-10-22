"""A tiny retrieval evaluation harness: hit@k and MRR.

Run:  python eval_harness.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "02-rag", "code"))
from mini_rag import DOCS, Index  # noqa: E402

# Each case: a question, the document that should be retrieved, and a phrase the chunk must contain.
GOLDEN = [
    {"q": "Why use multi-stage builds?", "doc": "docker.txt", "must": "small runtime image"},
    {"q": "Which status code means not modified?", "doc": "caching.txt", "must": "304"},
    {"q": "Where do refresh tokens belong?", "doc": "auth.txt", "must": "httpOnly cookie"},
    {"q": "How long does max-age keep a response fresh?", "doc": "caching.txt", "must": "max-age"},
]


def rank_of_first_hit(results, case):
    for rank, (_, doc, text) in enumerate(results, start=1):
        if doc == case["doc"] and case["must"].lower() in text.lower():
            return rank
    return None


def evaluate(index, golden, k=3):
    hits = 0
    rr = 0.0
    misses = []
    for case in golden:
        rank = rank_of_first_hit(index.search(case["q"], k=k), case)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(case["q"])
    n = len(golden)
    return {"hit@k": hits / n, "mrr": rr / n, "misses": misses}


if __name__ == "__main__":
    report = evaluate(Index(DOCS), GOLDEN, k=3)
    print(f"hit@3 = {report['hit@k']:.0%}   MRR = {report['mrr']:.2f}")
    for q in report["misses"]:
        print("MISS:", q)
