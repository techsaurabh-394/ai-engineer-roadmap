import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_attention as la  # noqa: E402

X = [[1.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0, 0.0]]


def test_attention_weights_rows_sum_to_one():
    _, w = la.scaled_dot_product_attention(X, X, X)
    for row in w:
        assert math.isclose(sum(row), 1.0)


def test_attention_hand_computed_example():
    # q matches the first key more than the second, so the output leans toward v[0]
    q = [[1.0, 0.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[10.0, 0.0], [0.0, 10.0]]
    out, w = la.scaled_dot_product_attention(q, k, v)
    s = 1 / math.sqrt(2)  # scores are [1, 0] / sqrt(2)
    p0 = math.exp(s) / (math.exp(s) + 1)
    assert math.isclose(w[0][0], p0)
    assert math.isclose(out[0][0], 10 * p0) and out[0][0] > out[0][1]


def test_causal_mask_blocks_future():
    _, w = la.scaled_dot_product_attention(X, X, X, la.causal_mask(3))
    assert w[0] == [1.0, 0.0, 0.0]
    for i in range(3):
        for j in range(i + 1, 3):
            assert w[i][j] == 0.0


def test_split_and_merge_heads_roundtrip():
    heads = la.split_heads(X, 2)
    assert len(heads) == 2 and len(heads[0][0]) == 2
    assert la.merge_heads(heads) == X


def test_split_heads_rejects_bad_divisor():
    try:
        la.split_heads(X, 3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_multi_head_shape_preserved():
    out, weights = la.multi_head_attention(X, 2, causal=True)
    assert len(out) == 3 and len(out[0]) == 4
    assert len(weights) == 2 and weights[1][0][1] == 0.0


def test_positional_encoding_values():
    pe = la.sinusoidal_positional_encoding(3, 4)
    assert pe[0] == [0.0, 1.0, 0.0, 1.0]  # sin(0)=0, cos(0)=1
    assert math.isclose(pe[1][0], math.sin(1.0))
    assert math.isclose(pe[1][2], math.sin(1 / 100))  # 10000^(2/4) = 100
    assert pe[1] != pe[2]  # each position gets a different vector
