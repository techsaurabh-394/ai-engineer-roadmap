"""Overfitting and regularization with polynomial regression. Pure Python, no libraries.

Fit y = sin(2 pi x) + noise with polynomials of degree 1..9 by least squares (normal equations
solved by Gaussian elimination), optionally with an L2 (ridge) penalty.

Run:  python ml_overfitting.py
"""
import math
import random


def make_data(n, noise=0.2, seed=0):
    rng = random.Random(seed)
    xs = [rng.random() for _ in range(n)]
    ys = [math.sin(2 * math.pi * x) + rng.gauss(0, noise) for x in xs]
    return xs, ys


def make_grid_data(n, noise=0.3, seed=0):
    """Same curve, but with n evenly spaced x values in [0, 1] (a small training set)."""
    rng = random.Random(seed)
    xs = [i / (n - 1) for i in range(n)]
    ys = [math.sin(2 * math.pi * x) + rng.gauss(0, noise) for x in xs]
    return xs, ys


def design(xs, degree):
    return [[x ** k for k in range(degree + 1)] for x in xs]


def solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x: list[float] = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def fit(xs, ys, degree, lam=0.0):
    """Minimise sum (X w - y)^2 + lam * sum_{k>=1} w_k^2  (the intercept w_0 is not penalised).
    Normal equations: (X^T X + lam I') w = X^T y."""
    X = design(xs, degree)
    d = degree + 1
    A = [[sum(X[r][i] * X[r][j] for r in range(len(xs))) for j in range(d)] for i in range(d)]
    for i in range(1, d):
        A[i][i] += lam
    b = [sum(X[r][i] * ys[r] for r in range(len(xs))) for i in range(d)]
    return solve(A, b)


def predict(w, x):
    return sum(c * x ** k for k, c in enumerate(w))


def mse(w, xs, ys):
    return sum((predict(w, x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def sweep(train, val, degrees, lam=0.0):
    """Return a list of (degree, train_mse, val_mse)."""
    out = []
    for d in degrees:
        w = fit(train[0], train[1], d, lam)
        out.append((d, mse(w, *train), mse(w, *val)))
    return out


def main():
    train = make_grid_data(10, 0.3, seed=5)
    val = make_data(300, 0.3, seed=99)
    degrees = list(range(1, 10))
    print("10 training points, 300 validation points; y = sin(2 pi x) + noise (sd 0.3)")
    print("The best any model can do is about the noise variance, 0.09.\n")
    print("degree  train MSE  validation MSE   (no regularization)")
    rows = sweep(train, val, degrees)
    for d, tr, va in rows:
        print(f"{d:>6}  {tr:9.4f}  {va:14.4f}")
    best = min(rows, key=lambda r: r[2])
    print(f"lowest validation error: degree {best[0]}")

    print("\nDegree 9 with an L2 (ridge) penalty lambda")
    print("lambda   train MSE  validation MSE  largest |weight|")
    for lam in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        w = fit(train[0], train[1], 9, lam)
        print(f"{lam:<8g} {mse(w, *train):9.4f}  {mse(w, *val):14.4f}  {max(abs(c) for c in w):14.4g}")


if __name__ == "__main__":
    main()
