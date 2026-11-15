import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ml_overfitting as ov  # noqa: E402

TRAIN = ov.make_grid_data(10, 0.3, seed=5)
VAL = ov.make_data(300, 0.3, seed=99)


def test_solve_known_system():
    # 2x + y = 5, x + 3y = 10  ->  x = 1, y = 3
    x = ov.solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert math.isclose(x[0], 1.0) and math.isclose(x[1], 3.0)


def test_fit_recovers_exact_polynomial():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [1 + 2 * x - 3 * x * x for x in xs]
    w = ov.fit(xs, ys, 2)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(w, [1.0, 2.0, -3.0]))


def test_train_error_never_rises_with_degree():
    rows = ov.sweep(TRAIN, VAL, range(1, 10))
    tr = [r[1] for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(tr, tr[1:]))


def test_validation_error_is_u_shaped():
    rows = ov.sweep(TRAIN, VAL, range(1, 10))
    val = {d: v for d, _, v in rows}
    best = min(val, key=val.get)
    assert 3 <= best <= 6
    assert val[1] > 2 * val[best]       # underfitting
    assert val[9] > 10 * val[best]      # overfitting
    assert rows[-1][1] < 1e-6           # degree 9 passes through all 10 points


def test_ridge_rescues_degree_9():
    plain = ov.sweep(TRAIN, VAL, [9], 0.0)[0]
    ridge = ov.sweep(TRAIN, VAL, [9], 1e-4)[0]
    assert ridge[2] < plain[2] / 10
    assert ridge[1] > plain[1]          # training error gets a bit worse
    w_plain = ov.fit(TRAIN[0], TRAIN[1], 9, 0.0)
    w_ridge = ov.fit(TRAIN[0], TRAIN[1], 9, 1e-4)
    assert max(map(abs, w_ridge)) < max(map(abs, w_plain)) / 1000


def test_data_is_deterministic():
    assert ov.make_data(5, seed=7) == ov.make_data(5, seed=7)
