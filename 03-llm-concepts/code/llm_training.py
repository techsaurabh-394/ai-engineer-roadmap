"""How LLMs are trained, in miniature. Standard library only, fully deterministic.

Covers:
  - cross-entropy loss and perplexity
  - gradient descent on one parameter
  - a tiny bigram language model trained with hand-written gradients (softmax - one-hot)
  - SFT loss masking (loss only on answer tokens)
  - a Bradley-Terry / DPO-style preference loss

Run:  python llm_training.py
"""
import math

CORPUS = (
    "the cat sat on the mat . the dog sat on the rug . "
    "the cat saw the dog . the dog saw the cat . "
    "a cat sat on a mat . a dog sat on a rug ."
).split()


# ---------- loss ----------
def softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [v / s for v in e]


def cross_entropy(prob_of_correct):
    """-ln(p). p=0.25 -> 1.386."""
    return -math.log(prob_of_correct)


def perplexity(mean_loss):
    return math.exp(mean_loss)


# ---------- gradient descent, one parameter ----------
def gradient_descent_1d(start=0.0, target=3.0, lr=0.1, steps=25):
    """Minimise loss(w) = (w - target)^2. Gradient = 2 (w - target). Returns the list of w values."""
    w = start
    path = [w]
    for _ in range(steps):
        w -= lr * 2 * (w - target)
        path.append(w)
    return path


# ---------- tiny bigram language model ----------
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    return vocab, {w: i for i, w in enumerate(vocab)}


def make_pairs(tokens, idx):
    return [(idx[a], idx[b]) for a, b in zip(tokens, tokens[1:])]


def train_bigram(tokens=None, steps=200, lr=0.5):
    """Train a V x V logit table. Row i = scores for the next token after token i.

    Loss per pair = -ln softmax(row)[target]; gradient wrt logits = softmax - onehot.
    Full-batch gradient descent, zero initial weights, so no randomness.
    Returns (vocab, idx, logits, losses) where losses[0] is the loss before any update.
    """
    tokens = tokens or CORPUS
    vocab, idx = build_vocab(tokens)
    V = len(vocab)
    pairs = make_pairs(tokens, idx)
    logits = [[0.0] * V for _ in range(V)]
    losses = []
    for _ in range(steps + 1):
        grad = [[0.0] * V for _ in range(V)]
        total = 0.0
        for cur, nxt in pairs:
            p = softmax(logits[cur])
            total += cross_entropy(p[nxt])
            for j in range(V):
                grad[cur][j] += p[j] - (1.0 if j == nxt else 0.0)
        losses.append(total / len(pairs))
        for i in range(V):
            for j in range(V):
                logits[i][j] -= lr * grad[i][j] / len(pairs)
    return vocab, idx, logits, losses


def next_token_probs(vocab, idx, logits, word):
    return dict(zip(vocab, softmax(logits[idx[word]])))


# ---------- SFT loss mask ----------
def loss_mask(prompt_tokens, answer_tokens):
    """1 for answer tokens (trained on), 0 for prompt tokens (ignored)."""
    return [0] * len(prompt_tokens) + [1] * len(answer_tokens)


def masked_loss(token_losses, mask):
    """Average loss over unmasked positions only."""
    kept = [l for l, m in zip(token_losses, mask) if m]
    return sum(kept) / len(kept)


# ---------- preference tuning ----------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bradley_terry_prob(reward_chosen, reward_rejected):
    """P(chosen beats rejected) under a reward model."""
    return sigmoid(reward_chosen - reward_rejected)


def reward_model_loss(reward_chosen, reward_rejected):
    return -math.log(bradley_terry_prob(reward_chosen, reward_rejected))


def dpo_loss(logp_chosen, logp_rejected, ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """DPO: -ln sigmoid(beta * [(logpi_c - logref_c) - (logpi_r - logref_r)])."""
    margin = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    return -math.log(sigmoid(margin))


if __name__ == "__main__":
    print("loss at p=0.25:", round(cross_entropy(0.25), 3), "perplexity:", round(perplexity(cross_entropy(0.25)), 2))
    print("gd path:", [round(w, 3) for w in gradient_descent_1d()[:6]], "...")
    vocab, idx, logits, losses = train_bigram()
    for s in (0, 1, 10, 50, 100, 200):
        print(f"step {s:3d} loss {losses[s]:.4f}")
    for w in ("the", "sat", "on"):
        probs = next_token_probs(vocab, idx, logits, w)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"after '{w}':", [(k, round(v, 2)) for k, v in top])
    prompt, answer = ["Q:", "2+2?", "A:"], ["4", "<eos>"]
    print("mask:", loss_mask(prompt, answer))
    print("bt prob (r=2 vs 0):", round(bradley_terry_prob(2.0, 0.0), 3))
    print("dpo loss (policy already prefers chosen):", round(dpo_loss(-1.0, -3.0, -2.0, -2.0), 4))
