import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_block as lb  # noqa: E402

X = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0], [3.0, 1.0, -2.0, 1.0]]


def test_layer_norm_mean_zero_var_one():
    out = lb.layer_norm([2.0, 4.0, 6.0, 8.0])
    mean = sum(out) / 4
    var = sum((v - mean) ** 2 for v in out) / 4
    assert abs(mean) < 1e-9
    assert math.isclose(var, 1.0, rel_tol=1e-3)


def test_layer_norm_hand_values():
    out = lb.layer_norm([2.0, 4.0, 6.0, 8.0])  # mean 5, var 5, std sqrt(5)
    assert math.isclose(out[0], -3 / math.sqrt(5), rel_tol=1e-3)


def test_layer_norm_gamma_beta():
    out = lb.layer_norm([2.0, 4.0, 6.0, 8.0], gamma=[2.0] * 4, beta=[10.0] * 4)
    assert math.isclose(sum(out) / 4, 10.0, rel_tol=1e-9)


def test_rms_norm_unit_rms():
    out = lb.rms_norm([3.0, -4.0, 0.0, 5.0])
    assert math.isclose(sum(v * v for v in out) / 4, 1.0, rel_tol=1e-3)


def test_activations():
    assert lb.relu(-2) == 0.0 and lb.relu(3) == 3
    assert lb.gelu(0.0) == 0.0
    assert math.isclose(lb.gelu(3.0), 3.0, abs_tol=0.01)


def test_ffn_shape_and_token_independence():
    w = lb.make_weights(4)
    assert len(w["w1"]) == 16 and len(w["w2"]) == 4
    out = lb.ffn(X[0], w["w1"], w["w2"])
    assert len(out) == 4
    assert out == lb.ffn(list(X[0]), w["w1"], w["w2"])  # depends only on this token


def test_block_preserves_shape_both_styles():
    for pre in (True, False):
        out = lb.block_forward(X, pre_norm=pre)
        assert len(out) == 3 and all(len(r) == 4 for r in out)


def test_residual_keeps_signal():
    with_res = lb.depth_experiment(50, residual=True)
    without = lb.depth_experiment(50, residual=False)
    assert with_res[-1] > with_res[0]
    assert without[-1] < 1e-10


def test_residual_gradient_factor():
    assert lb.residual_gradient_factor(20, residual=False) < 1e-5
    assert lb.residual_gradient_factor(20, residual=True) > 1


def test_param_counts():
    p = lb.param_counts(4096)
    assert p["attention"] == 4 * 4096 ** 2
    assert p["ffn"] == 8 * 4096 ** 2
    assert p["total"] == 12 * 4096 ** 2
    assert math.isclose(p["ffn_share"], 2 / 3)


def test_swiglu_params_close_to_classic():
    classic = lb.param_counts(4096)["ffn"]
    sw = lb.param_counts(4096, swiglu=True)["ffn"]
    assert abs(sw - classic) / classic < 0.001
