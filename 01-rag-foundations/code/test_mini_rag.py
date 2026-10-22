import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mini_rag import DOCS, Index, chunk, tokens  # noqa: E402


def test_tokens_drop_stopwords():
    assert "the" not in tokens("the cache")


def test_chunk_respects_size():
    text = " ".join(f"Sentence number {i} ends here." for i in range(40))
    assert all(len(c) <= 330 for c in chunk(text, 300))


def test_search_returns_right_document():
    index = Index(DOCS)
    assert index.search("which header gives 304 not modified")[0][1] == "caching.txt"
    assert index.search("where should refresh tokens be stored")[0][1] == "auth.txt"


def test_unrelated_query_returns_nothing():
    assert Index(DOCS).search("zzzz qqqq") == []
