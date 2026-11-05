"""Modern attention variants, standard library only.

Covers:
  - KV-cache size for MHA vs GQA vs MQA
  - online softmax (the core trick inside FlashAttention)
  - sliding-window causal mask
  - RoPE as a 2-D rotation: the dot product depends only on relative distance

Run:  python llm_attn_variants.py
"""
import math


# ---------- 1. KV-cache size ----------
def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, bytes_per_number=2, batch=1):
    """2 (K and V) * layers * kv_heads * head_dim * seq_len * bytes * batch."""
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per_number * batch


def kv_cache_table(layers=32, q_heads=32, head_dim=128, seq_len=8192, gqa_groups=8, bytes_per_number=2):
    """Return [(name, kv_heads, bytes)] for MHA, GQA and MQA."""
    rows = []
    for name, kv in (("MHA", q_heads), ("GQA", gqa_groups), ("MQA", 1)):
        rows.append((name, kv, kv_cache_bytes(layers, kv, head_dim, seq_len, bytes_per_number)))
    return rows


def kv_head_for_query_head(q_head, q_heads, kv_heads):
    """Which K/V head a query head reads. kv_heads=q_heads -> MHA, 1 -> MQA, else GQA."""
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    return q_head // (q_heads // kv_heads)


# ---------- 2. online softmax ----------
def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def online_softmax(xs):
    """Softmax computed in ONE streaming pass over xs, keeping only a running
    max `m` and running denominator `d`. When a bigger max shows up, the old
    denominator is rescaled by exp(old_m - new_m). FlashAttention uses this so
    it can process scores block by block without storing the whole row."""
    m = float("-inf")
    d = 0.0
    for x in xs:
        new_m = max(m, x)
        d = d * math.exp(m - new_m) + math.exp(x - new_m)
        m = new_m
    return [math.exp(x - m) / d for x in xs]


def blockwise_attention_row(scores, values, block=2):
    """Attention output for ONE query: sum_i softmax(scores)_i * values_i, computed
    block by block with a running max / denominator / weighted sum (tiling).
    values is a list of vectors. Never builds the full softmax row."""
    m = float("-inf")
    d = 0.0
    acc = [0.0] * len(values[0])
    for start in range(0, len(scores), block):
        s_blk = scores[start:start + block]
        v_blk = values[start:start + block]
        new_m = max(m, max(s_blk))
        scale = math.exp(m - new_m)  # rescale what we have so far (0.0 on the first block)
        d *= scale
        acc = [a * scale for a in acc]
        for s, v in zip(s_blk, v_blk):
            w = math.exp(s - new_m)
            d += w
            acc = [a + w * x for a, x in zip(acc, v)]
        m = new_m
    return [a / d for a in acc]


def full_attention_row(scores, values):
    w = softmax(scores)
    return [sum(wi * v[j] for wi, v in zip(w, values)) for j in range(len(values[0]))]


# ---------- 3. sliding-window mask ----------
def sliding_window_mask(n, window):
    """True where query i may look at key j: j <= i and i - j < window."""
    return [[(j <= i and i - j < window) for j in range(n)] for i in range(n)]


def count_allowed(mask):
    return sum(1 for row in mask for ok in row if ok)


def causal_pairs(n):
    return n * (n + 1) // 2


def render_mask(mask):
    return "\n".join(" ".join("X" if ok else "." for ok in row) for row in mask)


# ---------- 4. RoPE (2-D demo) ----------
def rotate(vec2, angle):
    x, y = vec2
    c, s = math.cos(angle), math.sin(angle)
    return (x * c - y * s, x * s + y * c)


def rope_2d(vec2, position, theta=0.1):
    """Rotate a 2-D vector by position * theta radians."""
    return rotate(vec2, position * theta)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rope_score(q, k, m, n, theta=0.1):
    """Dot product of q rotated for position m and k rotated for position n."""
    return dot(rope_2d(q, m, theta), rope_2d(k, n, theta))


def rope_vector(vec, position, base=10000.0):
    """RoPE for an even-length vector: pair up dims (0,1),(2,3),... and rotate pair
    p by position * base**(-2p/d), the same idea with several frequencies."""
    d = len(vec)
    if d % 2:
        raise ValueError("vector length must be even")
    out = []
    for p in range(d // 2):
        angle = position * base ** (-2 * p / d)
        out.extend(rotate((vec[2 * p], vec[2 * p + 1]), angle))
    return out


if __name__ == "__main__":
    print("KV cache, 32 layers, 32 query heads, head_dim 128, 8192 tokens, fp16")
    base_bytes = None
    for name, kv, b in kv_cache_table():
        base_bytes = base_bytes or b
        print(f"  {name}: kv_heads={kv:>2}  {b / 2**30:.3f} GiB  ({base_bytes / b:.0f}x smaller than MHA)")

    xs = [2.0, 1.0, 0.1, 3.0, -1.0]
    print("\nsoftmax       :", [round(x, 4) for x in softmax(xs)])
    print("online softmax:", [round(x, 4) for x in online_softmax(xs)])

    print("\nSliding window mask, n=8, window=3  (X = allowed)")
    mk = sliding_window_mask(8, 3)
    print(render_mask(mk))
    print(f"allowed pairs: {count_allowed(mk)} vs full causal {causal_pairs(8)}")

    q, k = (1.0, 0.5), (0.3, 0.8)
    print("\nRoPE: score depends only on m - n")
    for m, n in ((5, 2), (10, 7), (103, 100)):
        print(f"  m={m:>3} n={n:>3} m-n={m - n}  score={rope_score(q, k, m, n):.6f}")
