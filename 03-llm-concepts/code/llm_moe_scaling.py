"""Mixture of Experts and scaling laws, standard library only.

Covers:
  - a toy router: softmax over expert scores, top-k, renormalise
  - a toy MoE layer with fixed deterministic weights
  - total vs ACTIVE parameters
  - an imbalance metric / auxiliary balance loss idea
  - training FLOPs ~ 6 * params * tokens (arithmetic, not measurements)
  - a log-log least-squares fit that recovers a power-law exponent from SYNTHETIC data

Run:  python llm_moe_scaling.py
"""
import math


# ---------- router ----------
def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    t = sum(exps)
    return [e / t for e in exps]


def route(scores, k):
    """Return [(expert_index, weight)] for the top-k experts; weights sum to 1.
    Ties are broken by lower expert index (deterministic)."""
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between 1 and number of experts")
    probs = softmax(scores)
    top = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:k]
    total = sum(probs[i] for i in top)
    return [(i, probs[i] / total) for i in top]


def router_scores(token, router_weights):
    """Score for expert e = dot(token, router_weights[e])."""
    return [sum(t * w for t, w in zip(token, row)) for row in router_weights]


# ---------- toy MoE layer ----------
def make_experts(num_experts, dim):
    """Each 'expert' is a fixed scale-and-shift: out = a_e * x + b_e (deterministic)."""
    return [(1.0 + 0.5 * e, 0.1 * e) for e in range(num_experts)]


def expert_forward(expert, token):
    a, b = expert
    return [a * x + b for x in token]


def moe_layer(token, router_weights, experts, k=2):
    """Return (output_vector, [(expert, weight)]). Only the chosen k experts run."""
    choice = route(router_scores(token, router_weights), k)
    out = [0.0] * len(token)
    for e, w in choice:
        y = expert_forward(experts[e], token)
        out = [o + w * yi for o, yi in zip(out, y)]
    return out, choice


# ---------- parameter counting ----------
def ffn_params(d_model, d_ff):
    """Two matrices (up and down), biases ignored."""
    return 2 * d_model * d_ff


def moe_param_counts(d_model, d_ff, num_experts, top_k, shared_params=0):
    """Return (total, active) parameters for one MoE layer, plus anything shared
    (attention etc.) that every token always uses."""
    router = d_model * num_experts
    per_expert = ffn_params(d_model, d_ff)
    total = shared_params + router + num_experts * per_expert
    active = shared_params + router + top_k * per_expert
    return total, active


# ---------- load balancing ----------
def expert_load(choices_per_token, num_experts):
    """Fraction of (token, slot) assignments that went to each expert."""
    counts = [0] * num_experts
    for choice in choices_per_token:
        for e, _ in choice:
            counts[e] += 1
    total = sum(counts)
    return [c / total for c in counts]


def imbalance(load):
    """max load / ideal load. 1.0 = perfectly balanced, num_experts = all on one expert."""
    return max(load) * len(load)


def balance_loss(token_fraction, mean_prob, num_experts):
    """Switch-Transformer-style auxiliary loss: E * sum_e f_e * P_e.
    f_e = fraction of tokens sent to e, P_e = mean router probability for e.
    Equals 1.0 when both are uniform; grows when routing collapses on few experts."""
    return num_experts * sum(f * p for f, p in zip(token_fraction, mean_prob))


# ---------- scaling laws ----------
def training_flops(params, tokens):
    """Rule-of-thumb training compute: ~6 FLOPs per parameter per token."""
    return 6 * params * tokens


def chinchilla_tokens(params, tokens_per_param=20):
    """Rule of thumb: ~20 training tokens per parameter."""
    return tokens_per_param * params


def power_law(n, a, alpha):
    """loss-like quantity L(N) = a * N**(-alpha)."""
    return a * n ** (-alpha)


def fit_power_law(xs, ys):
    """Least squares on (log x, log y). Returns (a, alpha) for y = a * x**(-alpha)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    slope = sxy / sxx
    intercept = my - slope * mx
    return math.exp(intercept), -slope


if __name__ == "__main__":
    # fixed router weights: 4 features -> 8 experts
    dim, E, K = 4, 8, 2
    rw = [[math.sin(1.0 + e * 1.3 + j * 0.7) for j in range(dim)] for e in range(E)]
    experts = make_experts(E, dim)
    tokens = {
        "the": [1.0, 0.0, 0.0, 0.0], "cat": [0.0, 1.0, 0.5, 0.0], "sat": [0.0, 0.0, 1.0, 1.0],
        "on": [0.5, 0.5, 0.0, 0.5], "mat": [0.0, 1.0, 1.0, 0.0], "!": [1.0, 1.0, 1.0, 1.0],
    }
    print(f"Toy MoE: {E} experts, top-{K}")
    choices = []
    for name, tok in tokens.items():
        out, ch = moe_layer(tok, rw, experts, K)
        choices.append(ch)
        print(f"  {name:>4} -> " + ", ".join(f"expert {e} (w={w:.2f})" for e, w in ch))
    load = expert_load(choices, E)
    print("load per expert:", [round(x, 2) for x in load], " imbalance:", round(imbalance(load), 2))

    total, active = moe_param_counts(d_model=4096, d_ff=14336, num_experts=8, top_k=2)
    print(f"\nOne MoE FFN layer (d_model 4096, d_ff 14336): total {total:,}  active {active:,}  "
          f"active/total = {active / total:.2f}")

    print("\nTraining FLOPs ~ 6*N*D (arithmetic, not measured)")
    for n in (1e8, 1e9, 7e9, 70e9):
        d = chinchilla_tokens(n)
        print(f"  N={n:.0e}  D(20/param)={d:.1e}  FLOPs={training_flops(n, d):.2e}")

    # SYNTHETIC points generated from a known law, NOT real measurements
    ns = [1e6, 1e7, 1e8, 1e9]
    ys = [power_law(n, 10.0, 0.076) for n in ns]
    a, alpha = fit_power_law(ns, ys)
    print(f"\nSynthetic power-law fit: a={a:.3f}  alpha={alpha:.4f}  (true 10.0 / 0.076)")
