"""Gradient descent from scratch: a 1-D example and linear regression. No libraries.

Run:  python ml_gradient_descent.py
"""
import random


def descend_1d(lr, start=0.0, steps=10, target=3.0):
    """Minimise loss(w) = (w - target)^2 with gradient d loss / d w = 2 (w - target).

    Returns a list of (step, w, loss) rows, starting with step 0.
    """
    w = start
    rows = [(0, w, (w - target) ** 2)]
    for step in range(1, steps + 1):
        grad = 2 * (w - target)
        w = w - lr * grad
        rows.append((step, w, (w - target) ** 2))
    return rows


def make_line_data(n=20, slope=2.0, intercept=1.0, noise=0.3, seed=0):
    """Noisy points around y = slope * x + intercept, with x spread over [0, 1]."""
    rng = random.Random(seed)
    xs = [i / (n - 1) for i in range(n)]
    ys = [slope * x + intercept + rng.gauss(0, noise) for x in xs]
    return xs, ys


def mse(xs, ys, w, b):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def fit_line(xs, ys, lr=0.5, epochs=200):
    """Batch gradient descent on MSE for y_hat = w * x + b. Returns (w, b, losses)."""
    w = b = 0.0
    n = len(xs)
    losses = [mse(xs, ys, w, b)]
    for _ in range(epochs):
        dw = sum(2 * (w * x + b - y) * x for x, y in zip(xs, ys)) / n
        db = sum(2 * (w * x + b - y) for x, y in zip(xs, ys)) / n
        w -= lr * dw
        b -= lr * db
        losses.append(mse(xs, ys, w, b))
    return w, b, losses


def main():
    print("1-D: minimise (w - 3)^2 starting at w = 0, learning rate 0.1")
    for step, w, loss in descend_1d(0.1, steps=5):
        print(f"  step {step}: w = {w:.4f}  loss = {loss:.4f}")

    print("\nSame problem, three learning rates, loss after 10 steps")
    for lr in (0.01, 0.1, 0.5, 1.1):
        loss = descend_1d(lr, steps=10)[-1][2]
        print(f"  lr = {lr}: loss = {loss:.4g}")

    xs, ys = make_line_data()
    print("\nLinear regression on 20 noisy points around y = 2x + 1")
    for lr in (0.5, 2.0):
        w, b, losses = fit_line(xs, ys, lr=lr, epochs=200)
        print(f"  lr = {lr}: loss {losses[0]:.4f} -> {losses[10]:.4f} (epoch 10) -> {losses[-1]:.4g} (epoch 200)")
        print(f"          w = {w:.4g}  b = {b:.4g}")


if __name__ == "__main__":
    main()
