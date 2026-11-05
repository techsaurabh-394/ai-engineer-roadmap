import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_qkv as qk  # noqa: E402


def test_project_shapes_and_row_rule():
    q = qk.project(qk.X, qk.W_Q)
    assert len(q) == 3 and len(q[0]) == 2  # (3 x 4) @ (4 x 2) -> (3 x 2)
    # row i of the result is x_i W: the token "animal" = [0,1,1,0] picks rows 1 and 2 of W_Q
    assert q[1] == [0.0 + 0.0, 0.0 + 1.0]


def test_qkv_hand_checked_matrices():
    q, k, v = qk.qkv(qk.X, qk.W_Q, qk.W_K, qk.W_V)
    assert q == [[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]
    assert k == [[0.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert v == [[1.0, 0.0], [1.0, 3.0], [1.0, 2.0]]


def test_scores_and_scaling_hand_checked():
    t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V)
    assert t["scores"] == [[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 4.0, 3.0]]
    assert math.isclose(t["scaled"][2][1], 4.0 / math.sqrt(2))


def test_weights_hand_checked_row_for_it():
    t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V)
    row = t["weights"][2]
    a, b, c = (math.exp(s / math.sqrt(2)) for s in (2.0, 4.0, 3.0))
    assert math.isclose(row[1], b / (a + b + c))
    assert row[1] == max(row)  # "it" attends most to "animal"
    assert t["weights"][1] == [0.5, 0.5, 0.0]


def test_rows_sum_to_one():
    for causal in (True, False):
        t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V, causal=causal)
        for row in t["weights"]:
            assert math.isclose(sum(row), 1.0)


def test_causal_mask_zeros_upper_triangle():
    t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V, causal=True)
    for i in range(3):
        for j in range(i + 1, 3):
            assert t["weights"][i][j] == 0.0
    assert t["weights"][0] == [1.0, 0.0, 0.0]
    # without the mask the future gets weight
    open_t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V, causal=False)
    assert open_t["weights"][0][2] > 0.0


def test_output_is_weighted_average_of_values():
    t = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V)
    w, v = t["weights"][2], t["v"]
    for c in range(2):
        assert math.isclose(t["output"][2][c], sum(w[j] * v[j][c] for j in range(3)))
    assert len(t["output"]) == 3 and len(t["output"][0]) == 2


def test_weights_depend_on_input_but_w_is_fixed():
    other = [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    a = qk.attention_with_trace(qk.X, qk.W_Q, qk.W_K, qk.W_V)["weights"]
    b = qk.attention_with_trace(other, qk.W_Q, qk.W_K, qk.W_V)["weights"]
    assert a != b


def test_parameter_count():
    assert qk.qkv_param_count(4, 2) == 24
    assert qk.qkv_param_count(4096) == 3 * 4096 * 4096 == 50_331_648
    assert qk.qkv_param_count(4096, 128) == 3 * 4096 * 128
    assert qk.qkv_param_count(4, 2, bias=True) == 3 * (8 + 2)
