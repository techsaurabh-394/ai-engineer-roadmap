import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ml_tiny_network as net  # noqa: E402


def test_forward_matches_hand_calculation():
    c = net.forward([0.0, 1.0], net.INIT)
    assert [round(v, 4) for v in c["z1"]] == [-0.3, 0.6]
    assert [round(v, 4) for v in c["a1"]] == [0.4256, 0.6457]
    assert round(c["z2"], 4) == -0.0395
    assert round(c["a2"], 4) == 0.4901


def test_backward_hand_values():
    c = net.forward([0.0, 1.0], net.INIT)
    g = net.backward(c, 1.0, net.INIT)
    assert round(g["d2"], 4) == -0.1274
    assert [round(v, 4) for v in g["d1"]] == [-0.0218, 0.0175]


def _flat(params):
    return [(k, idx) for k in ("W1", "b1", "W2", "b2")
            for idx in ([(j, i) for j in range(2) for i in range(2)] if k == "W1"
                        else [(0, j) for j in range(2)] if k == "W2"
                        else [(j,) for j in range(len(params[k]))])]


def _get(p, k, idx):
    v = p[k]
    for i in idx:
        v = v[i]
    return v


def _set(p, k, idx, val):
    v = p[k]
    for i in idx[:-1]:
        v = v[i]
    v[idx[-1]] = val


def test_gradients_match_finite_differences():
    """Backprop gradients agree with (L(w+h) - L(w-h)) / 2h to 1e-6, for every parameter."""
    params = copy.deepcopy(net.INIT)
    analytic = net.mean_grads(params)
    h = 1e-5
    checked = 0
    for k, idx in _flat(params):
        orig = _get(params, k, idx)
        _set(params, k, idx, orig + h)
        up = net.mean_loss(params)
        _set(params, k, idx, orig - h)
        down = net.mean_loss(params)
        _set(params, k, idx, orig)
        numeric = (up - down) / (2 * h)
        assert abs(numeric - _get(analytic, k, idx)) < 1e-6, (k, idx)
        checked += 1
    assert checked == 9


def test_single_example_gradients_match_finite_differences():
    params = copy.deepcopy(net.INIT)
    x, y = net.DATA[2]
    g = net.backward(net.forward(x, params), y, params)
    h = 1e-5
    for k, idx in _flat(params):
        orig = _get(params, k, idx)
        _set(params, k, idx, orig + h)
        up = net.loss_one(net.forward(x, params)["a2"], y)
        _set(params, k, idx, orig - h)
        down = net.loss_one(net.forward(x, params)["a2"], y)
        _set(params, k, idx, orig)
        assert abs((up - down) / (2 * h) - _get(g, k, idx)) < 1e-6


def test_training_loss_falls_and_solves_xor():
    params = copy.deepcopy(net.INIT)
    losses = net.train(5000, params)
    assert len(losses) == 5001
    assert losses[-1] < losses[0] / 10
    for x, y in net.DATA:
        assert (net.predict(params, x) > 0.5) == (y == 1.0)


def test_train_does_not_touch_init():
    before = copy.deepcopy(net.INIT)
    net.train(10)
    assert net.INIT == before


def test_one_update_lowers_loss_on_that_example():
    c = net.forward([0.0, 1.0], net.INIT)
    g = net.backward(c, 1.0, net.INIT)
    p = copy.deepcopy(net.INIT)
    net.apply_update(p, g, 0.5)
    assert net.loss_one(net.forward([0.0, 1.0], p)["a2"], 1.0) < net.loss_one(c["a2"], 1.0)
    assert math.isclose(p["b2"][0], net.INIT["b2"][0] - 0.5 * g["b2"][0])
