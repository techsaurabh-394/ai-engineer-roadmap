"""Attention from scratch on tiny matrices. Standard library only.

Covers:
  - scaled dot-product attention:  softmax(Q K^T / sqrt(d_k)) V
  - causal mask (a token may only look at itself and earlier tokens)
  - multi-head split / merge
  - sinusoidal positional encoding (from "Attention Is All You Need")

Run:  python llm_attention.py
"""
import math

NEG_INF = float("-inf")


# ---------- tiny matrix helpers (lists of lists) ----------
def matmul(a, b):
    """(n x k) @ (k x m) -> (n x m)."""
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("shape mismatch in matmul")
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(m):
    return [list(row) for row in zip(*m)]


def softmax_row(row):
    m = max(row)
    if m == NEG_INF:
        raise ValueError("a row cannot be fully masked")
    exps = [math.exp(x - m) if x != NEG_INF else 0.0 for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def causal_mask(n):
    """n x n mask: 0.0 where allowed (j <= i), -inf where the future is hidden (j > i)."""
    return [[0.0 if j <= i else NEG_INF for j in range(n)] for i in range(n)]


# ---------- attention ----------
def scaled_dot_product_attention(q, k, v, mask=None):
    """Return (output, weights).

    q: (n_q x d_k), k: (n_k x d_k), v: (n_k x d_v). mask: (n_q x n_k) of 0.0 / -inf.
    """
    d_k = len(k[0])
    scores = matmul(q, transpose(k))
    scale = math.sqrt(d_k)
    scores = [[s / scale for s in row] for row in scores]
    if mask is not None:
        scores = [[s + m for s, m in zip(srow, mrow)] for srow, mrow in zip(scores, mask)]
    weights = [softmax_row(row) for row in scores]
    return matmul(weights, v), weights


def split_heads(x, num_heads):
    """(n x d_model) -> list of num_heads matrices, each (n x d_model/num_heads)."""
    d_model = len(x[0])
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    d_head = d_model // num_heads
    return [[row[h * d_head:(h + 1) * d_head] for row in x] for h in range(num_heads)]


def merge_heads(heads):
    """Inverse of split_heads: concatenate along the feature axis."""
    return [sum((h[i] for h in heads), []) for i in range(len(heads[0]))]


def multi_head_attention(x, num_heads, causal=False):
    """Self-attention with Q = K = V = x (no learned projections, to keep the idea visible).

    Real models multiply x by learned matrices W_q, W_k, W_v first and by W_o after merging.
    """
    heads = split_heads(x, num_heads)
    mask = causal_mask(len(x)) if causal else None
    outs, all_weights = [], []
    for h in heads:
        out, w = scaled_dot_product_attention(h, h, h, mask)
        outs.append(out)
        all_weights.append(w)
    return merge_heads(outs), all_weights


# ---------- positional encoding ----------
def sinusoidal_positional_encoding(seq_len, d_model):
    """PE[pos][2i] = sin(pos / 10000^(2i/d_model)), PE[pos][2i+1] = cos(same)."""
    pe = []
    for pos in range(seq_len):
        row = []
        for j in range(d_model):
            i = j // 2
            angle = pos / (10000 ** (2 * i / d_model))
            row.append(math.sin(angle) if j % 2 == 0 else math.cos(angle))
        pe.append(row)
    return pe


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _fmt(m):
    return "\n".join("  [" + ", ".join(f"{x:6.3f}" for x in row) + "]" for row in m)


if __name__ == "__main__":
    # three tokens, 4 features each (think "The", "cat", "sat")
    x = [[1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0, 0.0]]
    out, w = scaled_dot_product_attention(x, x, x)
    print("attention weights (no mask):\n" + _fmt(w))
    _, w_causal = scaled_dot_product_attention(x, x, x, causal_mask(3))
    print("\nattention weights (causal mask, upper triangle is 0):\n" + _fmt(w_causal))
    merged, per_head = multi_head_attention(x, num_heads=2)
    print(f"\n2 heads -> each head sees {len(per_head[0][0])} tokens x 2 features; merged shape "
          f"{len(merged)}x{len(merged[0])}")
    print("\nsinusoidal positional encoding (3 positions, d_model=4):\n" + _fmt(sinusoidal_positional_encoding(3, 4)))
