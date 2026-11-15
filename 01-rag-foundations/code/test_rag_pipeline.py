import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag_pipeline import (DOCS, build_index, chunk, cosine, coverage, embed, process_query, rerank, run,  # noqa: E402
                          search)


def test_chunks_respect_size_and_keep_every_sentence():
    text = DOCS[0]["text"]
    parts = chunk(text, 100)
    assert len(parts) > 1
    assert " ".join(parts) == text
    assert all(len(p) <= 100 or "." not in p[:-1] for p in parts)


def test_vectors_have_length_one():
    index = build_index(DOCS)
    for it in index["items"]:
        assert math.isclose(math.sqrt(sum(x * x for x in it["vector"])), 1.0, abs_tol=1e-12)


def test_query_processing_drops_stop_words():
    words, terms = process_query("How long does a refresh token last?")
    assert "how" in words and "how" not in terms
    assert terms == ["long", "refresh", "token", "last"]


def test_search_finds_the_right_document():
    r = run("How should refresh tokens be stored?")
    top = r["index"]["items"][r["hits"][0][1]]
    assert top["doc"] == "auth.txt"
    assert "refresh" in r["answer"].lower() and r["answer"].endswith("[1]")


def test_unrelated_question_finds_nothing():
    r = run("quantum entanglement")
    assert r["hits"] == [] and r["answer"] == "I could not find that in the documents."


def test_cosine_is_bounded_and_self_similar():
    index = build_index(DOCS)
    v = index["items"][0]["vector"]
    assert math.isclose(cosine(v, v), 1.0, abs_tol=1e-12)
    assert all(0 <= s <= 1 + 1e-12 for s, _ in search(embed(["token"], index), index, 10))


def test_rerank_can_change_the_order_and_is_sorted():
    index = build_index(DOCS, 60)
    terms = process_query("cap the steps of an agent loop")[1]
    ranked = rerank(terms, index, search(embed(terms, index), index, 5))
    scores = [x[0] for x in ranked]
    assert scores == sorted(scores, reverse=True)
    assert 0 <= coverage(terms, index["items"][ranked[0][1]]["tokens"]) <= 1


def test_prompt_contains_numbered_sources_and_question():
    r = run("What is an ETag?")
    assert "[1]" in r["prompt"] and r["prompt"].rstrip().endswith("Question: What is an ETag?")
