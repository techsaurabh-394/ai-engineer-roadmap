"""See what temperature and top-p do to next-token probabilities. No libraries, no API key.

Run:  python sampling_demo.py
"""
import math

TOKENS = ["Paris", "London", "Rome", "banana", "the"]
LOGITS = [5.0, 3.2, 2.9, 0.5, 1.0]  # raw scores a model might output after "The capital of France is"


def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use a tiny value for near-greedy)")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def top_p_filter(probs, p=0.9):
    """Keep the smallest set of tokens whose probabilities add up to at least p, then renormalise."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept, total = [], 0.0
    for i in order:
        kept.append(i)
        total += probs[i]
        if total >= p:
            break
    norm = sum(probs[i] for i in kept)
    return {TOKENS[i]: probs[i] / norm for i in kept}


if __name__ == "__main__":
    for t in (0.2, 1.0, 2.0):
        probs = softmax(LOGITS, t)
        row = "  ".join(f"{tok}:{p:.2f}" for tok, p in zip(TOKENS, probs))
        print(f"temperature {t:<3} -> {row}")
    print("\ntop-p 0.9 at temperature 1.0 keeps:", {k: round(v, 2) for k, v in top_p_filter(softmax(LOGITS, 1.0), 0.9).items()})
