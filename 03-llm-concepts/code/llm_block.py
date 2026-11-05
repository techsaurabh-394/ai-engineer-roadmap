"""Inside a transformer block: residuals, LayerNorm/RMSNorm and the feed-forward network.

Standard library only. Weights are fixed and deterministic (no randomness).

Run:  python llm_block.py
"""
import math


# ---------- normalisation ----------
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """Subtract the mean, divide by the standard deviation, then scale (gamma) and shift (beta)."""
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    gamma = gamma if gamma is not None else [1.0] * n
    beta = beta if beta is not None else [0.0] * n
    return [g * (v - mean) * inv + b for v, g, b in zip(x, gamma, beta)]


def rms_norm(x, gamma=None, eps=1e-5):
    """Divide by the root-mean-square only (no mean subtraction, no beta). Used by Llama-style models."""
    n = len(x)
    rms = math.sqrt(sum(v * v for v in x) / n + eps)
    gamma = gamma if gamma is not None else [1.0] * n
    return [g * v / rms for v, g in zip(x, gamma)]


# ---------- nonlinearities ----------
def relu(v):
    return v if v > 0 else 0.0


def gelu(v):
    """GELU, tanh approximation."""
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))


def silu(v):
    return v / (1.0 + math.exp(-v))


# ---------- tiny matrix helpers ----------
def matvec(w, x):
    """w is (out x in), x is (in,) -> (out,)."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def fixed_matrix(rows, cols, seed=1):
    """Deterministic pseudo-weights in about [-0.3, 0.3] built from sines (no random module)."""
    return [[0.3 * math.sin(seed * 12.9898 + i * 7.13 + j * 3.77) for j in range(cols)] for i in range(rows)]


# ---------- feed-forward network ----------
def ffn(x, w1, w2, activation=gelu):
    """Expand (d -> hidden), apply nonlinearity, project back (hidden -> d). One token at a time."""
    hidden = [activation(h) for h in matvec(w1, x)]
    return matvec(w2, hidden)


def swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU variant: silu(gate(x)) * up(x), then project down. Uses three matrices."""
    gate = [silu(g) for g in matvec(w_gate, x)]
    up = matvec(w_up, x)
    return matvec(w_down, [g * u for g, u in zip(gate, up)])


def make_weights(d_model, expansion=4):
    hidden = d_model * expansion
    return {
        "w1": fixed_matrix(hidden, d_model, seed=1),
        "w2": fixed_matrix(d_model, hidden, seed=2),
    }


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _mean_mix(seq):
    avg = [sum(col) / len(seq) for col in zip(*seq)]
    return [list(avg) for _ in seq]


def block_forward(x, weights=None, attn_fn=None, pre_norm=True):
    """One block over a sequence x (list of token vectors).

    attn_fn(list_of_vectors) -> list_of_vectors mixes tokens. Default: average of all tokens
    (a stand-in for attention, which is the only step that looks across tokens).
    Pre-norm:  x = x + attn(norm(x));  x = x + ffn(norm(x))
    Post-norm: x = norm(x + attn(x));  x = norm(x + ffn(x))
    """
    d = len(x[0])
    weights = weights or make_weights(d)
    attn = attn_fn or _mean_mix

    def f(v):
        return ffn(v, weights["w1"], weights["w2"])

    if pre_norm:
        mixed = attn([layer_norm(t) for t in x])
        x = [_add(t, m) for t, m in zip(x, mixed)]
        x = [_add(t, f(layer_norm(t))) for t in x]
    else:
        mixed = attn(x)
        x = [layer_norm(_add(t, m)) for t, m in zip(x, mixed)]
        x = [layer_norm(_add(t, f(t))) for t in x]
    return x


# ---------- residual depth experiment ----------
def depth_experiment(depth=50, scale=0.5, residual=True, x0=None):
    """Stack `depth` layers. Each layer computes f(x) = scale * x (a deliberately weak sub-layer).

    With a residual the layer returns x + f(x); without it, only f(x).
    Returns the size (L2 norm) of the signal after each layer.
    """
    x = list(x0) if x0 is not None else [1.0, -2.0, 0.5, 3.0]
    sizes = []
    for _ in range(depth):
        fx = [scale * v for v in x]
        x = _add(x, fx) if residual else fx
        sizes.append(math.sqrt(sum(v * v for v in x)))
    return sizes


def residual_gradient_factor(depth, local_slope=0.5, residual=True):
    """Gradient magnitude reaching the input of a stack of `depth` layers, each with slope local_slope.

    Plain stack: slope**depth. Residual stack: (1 + slope)**depth (the '1' is the highway).
    """
    return ((1.0 + local_slope) if residual else local_slope) ** depth


# ---------- parameter counts ----------
def param_counts(d_model, expansion=4, swiglu=False):
    """Weights (ignoring biases and norms) of one block.

    Attention: W_q, W_k, W_v, W_o, each d x d -> 4 d^2.
    FFN: classic = 2 matrices d x (4d) -> 8 d^2. SwiGLU = 3 matrices; hidden is shrunk to 8/3 d so total stays ~8 d^2.
    """
    attention = 4 * d_model ** 2
    if swiglu:
        hidden = int(round(expansion * d_model * 2 / 3))
        ffn_params = 3 * d_model * hidden
    else:
        ffn_params = 2 * d_model * (expansion * d_model)
    total = attention + ffn_params
    return {"attention": attention, "ffn": ffn_params, "total": total, "ffn_share": ffn_params / total}


if __name__ == "__main__":
    v = [2.0, 4.0, 6.0, 8.0]
    print("layer_norm", [round(a, 3) for a in layer_norm(v)])
    print("rms_norm  ", [round(a, 3) for a in rms_norm(v)])
    print("residual   norms at layers 1,10,50:", [round(depth_experiment(50)[i], 1) for i in (0, 9, 49)])
    print("no residual norms at layers 1,10,50:", [round(depth_experiment(50, residual=False)[i], 8) for i in (0, 9, 49)])
    p = param_counts(4096)
    print("d_model=4096:", p)
