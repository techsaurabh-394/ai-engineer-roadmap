import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ml_gradient_descent import descend_1d, fit_line, make_line_data, mse  # noqa: E402


def test_one_step_by_hand():
    # w = 0, grad = 2 * (0 - 3) = -6, new w = 0 - 0.1 * (-6) = 0.6
    rows = descend_1d(0.1, steps=1)
    assert math.isclose(rows[1][1], 0.6)
    assert math.isclose(rows[1][2], (0.6 - 3) ** 2)


def test_small_lr_loss_falls_every_step():
    losses = [r[2] for r in descend_1d(0.1, steps=10)]
    assert all(b < a for a, b in zip(losses, losses[1:]))


def test_big_lr_diverges():
    losses = [r[2] for r in descend_1d(1.1, steps=10)]
    assert losses[-1] > losses[0]
    assert all(b > a for a, b in zip(losses, losses[1:]))


def test_lr_one_overshoots_forever():
    # lr = 1 makes w bounce between 0 and 6 without ever settling
    rows = descend_1d(1.0, steps=4)
    assert [round(r[1], 6) for r in rows] == [0.0, 6.0, 0.0, 6.0, 0.0]


def test_linear_fit_recovers_line():
    xs, ys = make_line_data()
    w, b, losses = fit_line(xs, ys, lr=0.5, epochs=2000)
    assert abs(w - 2.0) < 0.5 and abs(b - 1.0) < 0.3
    assert losses[-1] < losses[0]
    assert math.isclose(losses[-1], mse(xs, ys, w, b))


def test_linear_fit_diverges_with_huge_lr():
    xs, ys = make_line_data()
    _, _, losses = fit_line(xs, ys, lr=2.0, epochs=50)
    assert losses[-1] > losses[0]


def test_data_is_deterministic():
    assert make_line_data(seed=3) == make_line_data(seed=3)
