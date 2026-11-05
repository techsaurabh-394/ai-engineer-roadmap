"""How an LLM works end to end, on a toy model small enough to print. Standard library only.

One short sentence travels through every stage:
  text -> tokens -> IDs -> embeddings -> + position -> N blocks -> logits -> softmax -> next token
and then the loop repeats with the new token appended (autoregressive generation).

IMPORTANT: every weight below is hand-picked and fixed. Nothing is trained and nothing is random,
so the output is deterministic. A real model has the same stages with billions of LEARNED numbers.

Run:  python llm_e2e_walkthrough.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from llm_attention import (add, causal_mask, matmul, scaled_dot_product_attention,  # noqa: E402
                           sinusoidal_positional_encoding, softmax_row)

# ---------- stage 0: the tokenizer (fixed lookup, not learned by the network) ----------
VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "ran", "."]
TOKEN_TO_ID = {w: i for i, w in enumerate(VOCAB)}
D_MODEL = 4

# ---------- LEARNED in a real model: embedding table (vocab x d_model) ----------
# Hand-picked "meaning" coordinates: [thing, action, place/relation, grammar glue]
EMBEDDING = [
    [0.0, 0.0, 0.0, 1.0],   # the
    [1.0, 0.0, 0.0, 0.0],   # cat
    [0.0, 1.0, 0.0, 0.0],   # sat
    [0.0, 0.0, 1.0, 0.0],   # on
    [1.0, 0.0, 1.0, 0.0],   # mat
    [1.0, 0.0, 0.0, 0.5],   # dog
    [0.0, 1.0, 0.0, 0.5],   # ran
    [0.0, 0.0, 0.0, 0.0],   # .
]

# ---------- LEARNED in a real model: one attention block + one feed-forward layer per block ----------
# Block 1 and block 2 reuse the same tiny matrices to keep the file short (real models use different ones).
W_Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
W_K = W_Q
W_V = [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]]
# feed-forward: lets "after 'on the'" turn into "a thing-place is likely". 4 -> 4 then ReLU.
W_FF = [[0.25, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [0.0, 0.0, 0.0, 0.25]]
N_BLOCKS = 2

# ---------- LEARNED in a real model: output matrix (d_model x vocab) turns a vector into logits ----------
# Column j scores vocab word j.  Rows read: [thing, action, place/relation, grammar glue].
W_OUT = [
    # the  cat  sat   on  mat  dog  ran    .
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],   # "thing" feature
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # "action" feature
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # "place" feature
    [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -2.0],  # "glue" feature
]


def parameter_count():
    """Count the numbers a model would LEARN in this toy (every weight matrix and the embedding table)."""
    mats = [EMBEDDING, W_Q, W_K, W_V, W_FF, W_OUT]
    return sum(len(m) * len(m[0]) for m in mats)


# ---------- the stages, one small function each ----------
def tokenize(text):
    """Split on spaces (a real tokenizer uses sub-word pieces such as BPE)."""
    pieces = []
    for word in text.lower().split():
        if word.endswith(".") and len(word) > 1:
            pieces += [word[:-1], "."]
        else:
            pieces.append(word)
    return pieces


def to_ids(tokens):
    unknown = [t for t in tokens if t not in TOKEN_TO_ID]
    if unknown:
        raise KeyError(f"tokens not in the toy vocabulary: {unknown}")
    return [TOKEN_TO_ID[t] for t in tokens]


def embed(ids):
    """Look up one vector per ID: (n,) -> (n x d_model)."""
    return [list(EMBEDDING[i]) for i in ids]


POS_SCALE = 0.1  # keep the position signal small next to the word meaning


def add_position(x):
    pe = sinusoidal_positional_encoding(len(x), D_MODEL)
    return add(x, [[POS_SCALE * v for v in row] for row in pe])


def relu_matrix(m):
    return [[max(0.0, v) for v in row] for row in m]


def block(x):
    """One transformer block: attention (+ residual), then feed-forward (+ residual)."""
    q, k, v = matmul(x, W_Q), matmul(x, W_K), matmul(x, W_V)
    attn, weights = scaled_dot_product_attention(q, k, v, causal_mask(len(x)))
    x = add(x, attn)                                   # residual: keep the old vector, add what we learned
    x = add(x, relu_matrix(matmul(x, W_FF)))           # feed-forward + residual
    return x, weights


def logits_from(hidden):
    """Only the LAST token's vector is used to predict the next token."""
    return matmul([hidden[-1]], W_OUT)[0]


def forward(ids):
    """Run the whole network once. Returns a dict of every stage's value."""
    stages = {"ids": ids}
    stages["embeddings"] = embed(ids)
    x = add_position(stages["embeddings"])
    stages["with_position"] = x
    stages["block_outputs"], stages["attention_weights"] = [], []
    for _ in range(N_BLOCKS):
        x, w = block(x)
        stages["block_outputs"].append(x)
        stages["attention_weights"].append(w)
    stages["logits"] = logits_from(x)
    stages["probs"] = softmax_row(stages["logits"])
    stages["next_id"] = max(range(len(VOCAB)), key=lambda i: stages["probs"][i])  # greedy pick
    return stages


def generate(text, max_new_tokens=3):
    """The autoregressive loop: predict ONE token, append it, feed everything back in, repeat."""
    ids = to_ids(tokenize(text))
    steps = []
    for _ in range(max_new_tokens):
        s = forward(ids)
        steps.append(s)
        ids = ids + [s["next_id"]]
        if VOCAB[s["next_id"]] == ".":
            break
    return [VOCAB[i] for i in ids], steps


def _row(r):
    return "[" + ", ".join(f"{v:6.3f}" for v in r) + "]"


def _show(name, m):
    shape = f"{len(m)}x{len(m[0])}"
    print(f"  {name} (shape {shape})")
    for r in m:
        print("    " + _row(r))


if __name__ == "__main__":
    text = "the cat sat on the"
    print(f"1. text: {text!r}")
    tokens = tokenize(text)
    print(f"2. tokens: {tokens}")
    ids = to_ids(tokens)
    print(f"3. IDs: {ids}   (vocab size {len(VOCAB)})")
    s = forward(ids)
    _show("4. embeddings, one vector per token", s["embeddings"])
    _show("5. + position (sinusoidal)", s["with_position"])
    for i, (out, w) in enumerate(zip(s["block_outputs"], s["attention_weights"]), 1):
        _show(f"6.{i} after block {i}", out)
        print(f"     last token's attention weights: {_row(w[-1])}")
    print(f"7. logits for the LAST position only (length {len(s['logits'])}): {_row(s['logits'])}")
    print(f"8. probabilities (softmax, sum={sum(s['probs']):.3f}):")
    for w, p in sorted(zip(VOCAB, s["probs"]), key=lambda t: -t[1]):
        print(f"     {w:>4}: {p:.3f}")
    print(f"9. pick next token (greedy): {VOCAB[s['next_id']]!r}")
    print("\n10. autoregressive loop (append, repeat):")
    final, steps = generate(text, max_new_tokens=4)
    cur = tokens
    for st in steps:
        print(f"     {' '.join(cur)}  ->  {VOCAB[st['next_id']]!r}  (p={st['probs'][st['next_id']]:.3f})")
        cur = cur + [VOCAB[st['next_id']]]
    print(f"final text: {' '.join(final)}")
    print(f"\nparameters learned in this toy: {parameter_count()}")
