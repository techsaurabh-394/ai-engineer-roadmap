"""Q, K and V from scratch on tiny matrices. Standard library only.

Shows, step by step, how one input matrix X becomes Q, K, V and then an attention output:

    Q = X W_q    K = X W_k    V = X W_v
    output = softmax(Q K^T / sqrt(d_k) + mask) V

Run:  python llm_qkv.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from llm_attention import causal_mask, matmul, softmax_row, transpose  # noqa: E402

# ---------- the worked example: 3 tokens, d_model = 4, d_k = 2 ----------
TOKENS = ["the", "animal", "it"]

# one row per token, 4 numbers each (a tiny made-up embedding)
X = [[1.0, 0.0, 0.0, 0.0],   # the
     [0.0, 1.0, 1.0, 0.0],   # animal
     [0.0, 0.0, 1.0, 1.0]]   # it

# d_model x d_k  (4 x 2). In a real model these are LEARNED; here they are hand-picked.
W_Q = [[1.0, 0.0],
       [0.0, 0.0],
       [0.0, 1.0],
       [1.0, 1.0]]
W_K = [[0.0, 1.0],
       [1.0, 0.0],
       [1.0, 1.0],
       [0.0, 0.0]]
W_V = [[1.0, 0.0],
       [0.0, 2.0],
       [1.0, 1.0],
       [0.0, 1.0]]


def project(x, w):
    """One projection: (n x d_model) @ (d_model x d_k) -> (n x d_k). Row i is x_i W."""
    return matmul(x, w)


def qkv(x, w_q, w_k, w_v):
    """Return (Q, K, V), each (n x d_k). Three matrices, three different views of the same x."""
    return project(x, w_q), project(x, w_k), project(x, w_v)


def attention_with_trace(x, w_q, w_k, w_v, causal=True):
    """Run attention and return a dict holding EVERY intermediate result."""
    q, k, v = qkv(x, w_q, w_k, w_v)
    d_k = len(k[0])
    n = len(x)
    scores = matmul(q, transpose(k))                       # n x n raw dot products
    scaled = [[s / math.sqrt(d_k) for s in row] for row in scores]
    mask = causal_mask(n) if causal else [[0.0] * n for _ in range(n)]
    masked = [[s + m for s, m in zip(sr, mr)] for sr, mr in zip(scaled, mask)]
    weights = [softmax_row(row) for row in masked]         # each row sums to 1
    output = matmul(weights, v)                            # n x d_k
    return {"x": x, "q": q, "k": k, "v": v, "scores": scores, "scaled": scaled,
            "mask": mask, "masked": masked, "weights": weights, "output": output, "d_k": d_k}


def qkv_param_count(d_model, d_k=None, bias=False):
    """Parameters in the three projection matrices of ONE attention head (d_k defaults to d_model).

    Each matrix is d_model x d_k. With bias, each also has d_k extra numbers.
    """
    d_k = d_model if d_k is None else d_k
    per_matrix = d_model * d_k + (d_k if bias else 0)
    return 3 * per_matrix


def _fmt(m, width=7):
    def cell(x):
        return "-inf".rjust(width) if x == float("-inf") else f"{x:{width}.3f}"
    return "\n".join("  [" + ", ".join(cell(x) for x in row) + "]" for row in m)


def print_trace(trace, tokens=TOKENS):
    """Print every stage with row labels so a beginner can follow the numbers."""
    def labelled(m):
        lines = _fmt(m).split("\n")
        return "\n".join(f"{t:>7}{line}" for t, line in zip(tokens, lines))

    print(f"tokens: {tokens}")
    for key, title in [("x", "X  (n x d_model)"), ("q", "Q = X W_q  (n x d_k): what each token LOOKS FOR"),
                       ("k", "K = X W_k  (n x d_k): how each token ADVERTISES itself"),
                       ("v", "V = X W_v  (n x d_k): what each token HANDS OVER"),
                       ("scores", "scores = Q K^T  (n x n): row i = token i's query vs every key"),
                       ("scaled", f"scaled = scores / sqrt(d_k) = scores / {math.sqrt(trace['d_k']):.3f}"),
                       ("masked", "after causal mask (-inf = may not look at the future)"),
                       ("weights", "weights = softmax(each row)  (rows sum to 1)"),
                       ("output", "output = weights V  (n x d_k)")]:
        print(f"\n{title}\n{labelled(trace[key])}")


if __name__ == "__main__":
    t = attention_with_trace(X, W_Q, W_K, W_V, causal=True)
    print_trace(t)
    print("\nRead the last weights row: token 'it' spreads its attention as",
          {tok: round(w, 3) for tok, w in zip(TOKENS, t["weights"][2])})
    print("\nparameters in W_q + W_k + W_v for d_model=4, d_k=2:", qkv_param_count(4, 2))
    print("parameters in W_q + W_k + W_v for d_model=4096 (one full-width layer):", f"{qkv_param_count(4096):,}")
    print("per 32 layers:", f"{32 * qkv_param_count(4096):,}")
