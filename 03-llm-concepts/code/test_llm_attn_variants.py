import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import llm_attn_variants as av  # noqa: E402


def test_kv_cache_formula_and_table():
    # 2 * 32 * 32 * 128 * 8192 * 2 bytes = 4 GiB
    assert av.kv_cache_bytes(32, 32, 128, 8192) == 4 * 2**30
    table = {n: (kv, b) for n, kv, b in av.kv_cache_table()}
    assert table["MHA"][1] // table["GQA"][1] == 4
    assert table["MHA"][1] // table["MQA"][1] == 32
    assert table["GQA"][0] == 8 and table["MQA"][0] == 1


def test_kv_head_mapping():
    assert [av.kv_head_for_query_head(h, 8, 8) for h in range(8)] == list(range(8))
    assert {av.kv_head_for_query_head(h, 8, 1) for h in range(8)} == {0}
    assert [av.kv_head_for_query_head(h, 8, 2) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        av.kv_head_for_query_head(0, 8, 3)


@pytest.mark.parametrize("xs", [[2.0, 1.0, 0.1, 3.0, -1.0], [0.0], [5.0, 5.0, 5.0], [-3.0, 10.0, -7.0, 2.5]])
def test_online_softmax_matches_softmax(xs):
    a, b = av.softmax(xs), av.online_softmax(xs)
    assert all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(a, b))
    assert math.isclose(sum(b), 1.0)


def test_online_softmax_is_stable_for_big_numbers():
    out = av.online_softmax([1000.0, 1001.0, 999.0])
    assert math.isclose(sum(out), 1.0)


@pytest.mark.parametrize("block", [1, 2, 3, 5])
def test_blockwise_attention_equals_full(block):
    scores = [0.5, 2.0, -1.0, 3.0, 0.0]
    values = [[1, 0], [0, 1], [1, 1], [2, -1], [0.5, 0.5]]
    a = av.full_attention_row(scores, values)
    b = av.blockwise_attention_row(scores, values, block=block)
    assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(a, b))


def test_sliding_window_mask():
    mk = av.sliding_window_mask(8, 3)
    assert av.causal_pairs(8) == 36
    assert av.count_allowed(mk) == 1 + 2 + 3 * 6  # = 21
    assert not mk[0][1]          # no peeking at the future
    assert mk[5][3] and not mk[5][2]  # window of 3: keys 3,4,5 only
    assert av.count_allowed(av.sliding_window_mask(8, 8)) == av.causal_pairs(8)


def test_rope_score_depends_only_on_relative_distance():
    q, k = (1.0, 0.5), (0.3, 0.8)
    base = av.rope_score(q, k, 5, 2)
    for shift in (0, 7, 100, 1000):
        assert math.isclose(av.rope_score(q, k, 5 + shift, 2 + shift), base, abs_tol=1e-9)
    assert not math.isclose(av.rope_score(q, k, 6, 2), base, abs_tol=1e-6)


def test_rope_preserves_length():
    v = (3.0, 4.0)
    assert math.isclose(math.hypot(*av.rope_2d(v, 17)), 5.0)


def test_rope_vector_relative():
    q = [1.0, 0.5, -0.2, 0.7]
    k = [0.3, 0.8, 0.4, -0.1]
    a = av.dot(av.rope_vector(q, 9), av.rope_vector(k, 4))
    b = av.dot(av.rope_vector(q, 59), av.rope_vector(k, 54))
    assert math.isclose(a, b, abs_tol=1e-9)
    with pytest.raises(ValueError):
        av.rope_vector([1.0, 2.0, 3.0], 1)
