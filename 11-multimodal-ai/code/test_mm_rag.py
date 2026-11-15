import pytest

import mm_rag as r


def ids(ranking):
    return [d for d, _ in ranking]


def test_caption_search_misses_uninformative_caption():
    got = ids(r.rank_by_caption(r.QUERY_TEXT, r.CORPUS))
    assert got[0] == "img1"
    assert "img3" not in got            # caption is just "Figure 4."


def test_vector_search_finds_uncaptioned_chart():
    got = ids(r.rank_by_vector(r.QUERY_FEATURES, r.CORPUS))
    assert got[:2] == ["img1", "img3"]
    assert "t1" not in got              # text chunks have no features
    assert got[-1] == "img4"


def test_rrf_scores_by_formula():
    fused = dict(r.rrf([[("a", 0), ("b", 0)], [("b", 0), ("a", 0)]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == pytest.approx(fused["b"])


def test_rrf_rewards_agreement():
    fused = r.rrf([r.rank_by_caption(r.QUERY_TEXT, r.CORPUS),
                   r.rank_by_vector(r.QUERY_FEATURES, r.CORPUS)])
    assert ids(fused)[0] == "img1"
    assert dict(fused)["img1"] == pytest.approx(2 / 61)


def test_fusion_recovers_item_caption_search_missed():
    fused = ids(r.rrf([r.rank_by_caption(r.QUERY_TEXT, r.CORPUS),
                       r.rank_by_vector(r.QUERY_FEATURES, r.CORPUS)]))
    assert "img3" in fused[:3]


def test_bm25_zero_when_no_overlap():
    assert r.bm25_scores("zzz", ["alpha beta", "gamma"]) == [0.0, 0.0]


def test_cosine_basics():
    assert r.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert r.cosine([1, 0], [0, 1]) == 0.0
    assert r.cosine([0, 0], [1, 1]) == 0.0


def test_prompt_references_image_ids():
    prompt = r.build_prompt("q?", ["img1", "t1"], r.CORPUS)
    assert "[img1] (image attached)" in prompt
    assert "[t1] Revenue grew" in prompt
    assert prompt.endswith("Question: q?")
    assert r.attached_images(["img1", "t1"], r.CORPUS) == ["img1"]
