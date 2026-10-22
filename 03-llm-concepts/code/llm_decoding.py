"""Decoding strategies over a toy next-token table, plus a KV-cache cost demo. Standard library only.

A real LLM computes next-token probabilities with a neural network. Here we look them up in a
small hand-written table, so you can see exactly what each decoding strategy does.

Run:  python llm_decoding.py
"""
import math
import random

END = "<end>"

# P(next token | previous token). Each row sums to 1.
NEXT = {
    "<start>": {"the": 0.5, "a": 0.4, "my": 0.1},
    "the": {"cat": 0.4, "dog": 0.35, "big": 0.25},
    "a": {"dog": 0.9, "cat": 0.1},
    "my": {"cat": 0.6, "dog": 0.4},
    "cat": {"sat": 0.6, "ran": 0.3, END: 0.1},
    "dog": {"ran": 0.5, "sat": 0.4, END: 0.1},
    "big": {"dog": 1.0},
    "sat": {END: 1.0},
    "ran": {END: 1.0},
}


def next_probs(prev):
    return NEXT.get(prev, {END: 1.0})


# ---------- filters ----------
def top_k_filter(probs, k):
    """Keep the k most likely tokens, renormalise."""
    if k < 1:
        raise ValueError("k must be >= 1")
    kept = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    total = sum(p for _, p in kept)
    return {t: p / total for t, p in kept}


def top_p_filter(probs, p):
    """Nucleus: keep the smallest set of top tokens whose mass reaches p, renormalise."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    kept, mass = [], 0.0
    for t, pr in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        kept.append((t, pr))
        mass += pr
        if mass >= p - 1e-12:
            break
    total = sum(pr for _, pr in kept)
    return {t: pr / total for t, pr in kept}


def apply_temperature(probs, temperature):
    """Re-shape probabilities as if logits were divided by temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = {t: math.log(p) / temperature for t, p in probs.items() if p > 0}
    m = max(logits.values())
    exps = {t: math.exp(v - m) for t, v in logits.items()}
    z = sum(exps.values())
    return {t: e / z for t, e in exps.items()}


# ---------- decoders ----------
def greedy(start="<start>", max_len=10):
    seq = [start]
    while len(seq) < max_len:
        probs = next_probs(seq[-1])
        tok = max(probs, key=lambda t: probs[t])
        seq.append(tok)
        if tok == END:
            break
    return seq[1:]


def sample(start="<start>", max_len=10, k=None, p=None, temperature=1.0, rng=None):
    rng = rng or random.Random(0)
    seq = [start]
    while len(seq) < max_len:
        probs = apply_temperature(next_probs(seq[-1]), temperature)
        if k is not None:
            probs = top_k_filter(probs, k)
        if p is not None:
            probs = top_p_filter(probs, p)
        toks = list(probs)
        tok = rng.choices(toks, weights=[probs[t] for t in toks])[0]
        seq.append(tok)
        if tok == END:
            break
    return seq[1:]


def sequence_logprob(tokens, start="<start>"):
    prev, lp = start, 0.0
    for t in tokens:
        lp += math.log(next_probs(prev)[t])
        prev = t
    return lp


def beam_search(beam_width=2, start="<start>", max_len=10):
    """Keep the beam_width best partial sequences (by total log-probability) at each step."""
    beams = [([start], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for seq, lp in beams:
            for tok, pr in next_probs(seq[-1]).items():
                cand = (seq + [tok], lp + math.log(pr))
                (finished if tok == END else candidates).append(cand)
        if not candidates:
            break
        beams = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]
    finished.extend(beams)
    best = max(finished, key=lambda c: c[1])
    return best[0][1:], best[1]


# ---------- KV cache demo ----------
def count_kv_computations(num_tokens, use_cache):
    """Count how many key/value vectors get computed while generating num_tokens tokens.

    Without a cache, step t recomputes K and V for all t tokens so far: 1 + 2 + ... + n.
    With a cache, each token's K and V are computed once and reused: n.
    """
    cache = []
    computed = 0
    for t in range(1, num_tokens + 1):
        if use_cache:
            cache.append(("k", "v", t))  # only the new token
            computed += 1
        else:
            computed += t  # rebuild K, V for every token in the prefix
    return computed


if __name__ == "__main__":
    print("greedy        :", " ".join(greedy()))
    print("beam (width 2):", " ".join(beam_search(2)[0]))
    print("top-k=2 sample:", " ".join(sample(k=2, rng=random.Random(1))))
    print("top-p=0.8     :", " ".join(sample(p=0.8, rng=random.Random(2))))
    for n in (10, 100, 1000):
        print(f"KV vectors for {n:>4} tokens: no cache {count_kv_computations(n, False):>7}, "
              f"cache {count_kv_computations(n, True):>5}")
