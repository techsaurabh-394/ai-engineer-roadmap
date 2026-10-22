import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_query_transform import (  # noqa: E402
    FakeLLM,
    expand_query,
    hyde_prompt,
    hyde_retrieve,
    keyword_search,
    multi_query_retrieve,
    rrf_merge,
    step_back_prompt,
    step_back_retrieve,
)

Q = "Why does my pod crash over and over?"


def test_expansion_adds_synonyms_and_finds_more():
    expanded = expand_query(Q)
    assert expanded.startswith("pod crash")
    assert "crashloopbackoff" in expanded
    assert "d2" in keyword_search(expanded)


def test_rrf_rewards_docs_that_appear_in_many_lists():
    merged = rrf_merge([["a", "b", "c"], ["b", "c", "a"], ["b", "x"]])
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "x"}


def test_rrf_exact_value_ordering():
    # a: 1/61 + 1/62 ; b: 1/61 -> a wins
    assert rrf_merge([["a"], ["b", "a"]])[0] == "a"


def test_multi_query_uses_llm_variants():
    llm = FakeLLM()
    queries, top = multi_query_retrieve(Q, llm)
    assert len(queries) == 4 and queries[0] == Q
    assert "d2" in top and "d1" in top
    assert llm.prompts[0].startswith("Write 3 different search queries")


def test_step_back_prompt_and_retrieval():
    assert "more general" in step_back_prompt(Q)
    general, top = step_back_retrieve(Q, FakeLLM())
    assert "Kubernetes" in general
    assert len(top) <= 3 and top


def test_hyde_searches_with_hypothetical_passage():
    assert Q in hyde_prompt(Q)
    passage, top = hyde_retrieve(Q, FakeLLM())
    assert "CrashLoopBackOff" in passage
    assert top[0] == "d2"
