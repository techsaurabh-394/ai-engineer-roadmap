import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag_bm25 import (  # noqa: E402
    BM25,
    DOCS,
    hybrid_search,
    matches,
    reciprocal_rank_fusion,
    toy_dense_scores,
)

TEXTS = [d["text"] for d in DOCS]


def test_bm25_finds_keyword_match():
    hits = BM25(TEXTS).search("ETag 304")
    assert hits[0][1] == 2


def test_bm25_rare_term_has_higher_idf():
    bm = BM25(["cat dog", "cat bird", "cat fish"])
    assert bm.idf("fish") > bm.idf("cat")


def test_bm25_no_match_returns_empty():
    assert BM25(TEXTS).search("zzzz qqqq") == []


def test_bm25_length_normalisation():
    bm = BM25(["kafka", "kafka " + "filler " * 30])
    assert bm.score("kafka", 0) > bm.score("kafka", 1)


def test_dense_handles_word_variants():
    # "caching" never appears; trigrams still link it to "Cache-Control".
    scores = toy_dense_scores("caching freshness seconds", TEXTS)
    assert scores[0][1] == 3


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[:2] in (["a", "b"], ["b", "a"])
    assert fused.index("d") > fused.index("a")


def test_rrf_exact_scores():
    # a: 1/61 + 1/62, b: 1/62 + 1/61 -> tie broken by id; c only in list 1.
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]]) == ["a", "b", "c"]


def test_matches_operators():
    meta = {"team": "security", "year": 2025}
    assert matches(meta, {"team": "security"})
    assert matches(meta, {"team": ["security", "platform"]})
    assert matches(meta, {"year": ("gte", 2025)})
    assert not matches(meta, {"year": ("lt", 2025)})
    assert not matches(meta, {"missing": 1})


def test_hybrid_with_filter_only_returns_allowed_docs():
    res = hybrid_search("rotate refresh token", DOCS, filters={"year": ("gte", 2025)})
    assert res and all(DOCS[i]["meta"]["year"] >= 2025 for i in res)
    assert res[0] == 0


def test_hybrid_filter_that_excludes_everything():
    assert hybrid_search("ETag", DOCS, filters={"team": "nobody"}) == []
