"""A 2-2-1 sigmoid network written from scratch: forward pass, backpropagation, XOR training.

Module-level names (INIT, DATA, LR, forward, backward, train, ...) are plain data and pure
functions so another language can port them line by line. No libraries, no randomness:
the starting weights are fixed.

Layout: W1[j][i] connects input i to hidden unit j; W2[0][j] connects hidden unit j to the output.
Loss for one example: L = 0.5 * (a2 - y)^2.

Run:  python ml_tiny_network.py
"""
import copy
import math

INIT = {
    "W1": [[0.5, -0.4], [0.3, 0.8]],
    "b1": [0.1, -0.2],
    "W2": [[0.7, -0.6]],
    "b2": [0.05],
}
DATA = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]  # XOR
LR = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def forward(x, params):
    """One example. Returns {'x', 'z1', 'a1', 'z2', 'a2'} (z = before activation, a = after)."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = [sum(W1[j][i] * x[i] for i in range(2)) + b1[j] for j in range(2)]
    a1 = [sigmoid(z) for z in z1]
    z2 = sum(W2[0][j] * a1[j] for j in range(2)) + b2[0]
    a2 = sigmoid(z2)
    return {"x": list(x), "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def loss_one(a2, y):
    return 0.5 * (a2 - y) ** 2


def backward(cache, y, params):
    """Backpropagation for one example. Returns gradients with the same shapes as the params,
    plus the intermediate deltas under 'd2' (output unit) and 'd1' (hidden units)."""
    a1, a2, x = cache["a1"], cache["a2"], cache["x"]
    W2 = params["W2"]
    d2 = (a2 - y) * a2 * (1 - a2)                       # dL/dz2
    d1 = [d2 * W2[0][j] * a1[j] * (1 - a1[j]) for j in range(2)]  # dL/dz1[j]
    return {
        "W2": [[d2 * a1[0], d2 * a1[1]]],
        "b2": [d2],
        "W1": [[d1[j] * x[i] for i in range(2)] for j in range(2)],
        "b1": list(d1),
        "d2": d2,
        "d1": d1,
    }


def mean_loss(params, data=DATA):
    return sum(loss_one(forward(x, params)["a2"], y) for x, y in data) / len(data)


def mean_grads(params, data=DATA):
    """Average the per-example gradients over the dataset."""
    total = {"W1": [[0.0, 0.0], [0.0, 0.0]], "b1": [0.0, 0.0], "W2": [[0.0, 0.0]], "b2": [0.0]}
    for x, y in data:
        g = backward(forward(x, params), y, params)
        for j in range(2):
            for i in range(2):
                total["W1"][j][i] += g["W1"][j][i]
            total["b1"][j] += g["b1"][j]
            total["W2"][0][j] += g["W2"][0][j]
        total["b2"][0] += g["b2"][0]
    n = len(data)
    return {
        "W1": [[v / n for v in row] for row in total["W1"]],
        "b1": [v / n for v in total["b1"]],
        "W2": [[v / n for v in total["W2"][0]]],
        "b2": [total["b2"][0] / n],
    }


def apply_update(params, grads, lr=LR):
    """In place: param -= lr * grad."""
    for j in range(2):
        for i in range(2):
            params["W1"][j][i] -= lr * grads["W1"][j][i]
        params["b1"][j] -= lr * grads["b1"][j]
        params["W2"][0][j] -= lr * grads["W2"][0][j]
    params["b2"][0] -= lr * grads["b2"][0]


def train(epochs, params=None, lr=LR, data=DATA):
    """Full-batch gradient descent. Returns the list of mean losses: before training, then
    after every epoch (length epochs + 1). If params is None a copy of INIT is used; pass your own dict to keep the trained weights (it is updated in place)."""
    if params is None:
        params = copy.deepcopy(INIT)
    losses = [mean_loss(params, data)]
    for _ in range(epochs):
        apply_update(params, mean_grads(params, data), lr)
        losses.append(mean_loss(params, data))
    return losses


def predict(params, x):
    return forward(x, params)["a2"]


def main():
    x, y = DATA[1]
    c = forward(x, INIT)
    print("Forward pass on input", x, "(target", y, ")")
    print("  z1 =", [round(v, 4) + 0.0 for v in c["z1"]], " a1 =", [round(v, 4) + 0.0 for v in c["a1"]])
    print(f"  z2 = {c['z2']:.4f}  a2 = {c['a2']:.4f}  loss = {loss_one(c['a2'], y):.4f}")
    g = backward(c, y, INIT)
    print("Backward pass")
    print(f"  d2 = {g['d2']:.4f}  d1 = {[round(v, 4) + 0.0 for v in g['d1']]}")
    print("  dW2 =", [round(v, 4) + 0.0 for v in g["W2"][0]], " db2 =", round(g["b2"][0], 4))
    print("  dW1 =", [[round(v, 4) + 0.0 for v in row] for row in g["W1"]], " db1 =", [round(v, 4) + 0.0 for v in g["b1"]])
    stepped = copy.deepcopy(INIT)
    apply_update(stepped, g, 0.5)
    print(f"  one update (lr 0.5) moves this example's loss {loss_one(c['a2'], y):.4f} -> "
          f"{loss_one(forward(x, stepped)['a2'], y):.4f}")

    params = copy.deepcopy(INIT)
    losses = train(5000, params)
    print("\nTraining on XOR, full batch, lr", LR)
    for e in (0, 500, 2000, 3000, 4000, 5000):
        print(f"  epoch {e:>4}: loss = {losses[e]:.5f}")
    for xx, yy in DATA:
        print(f"  input {xx} target {yy} -> prediction {predict(params, xx):.3f}")


if __name__ == "__main__":
    main()
