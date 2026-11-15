"""LoRA parameter arithmetic, rank proof, merge check, QLoRA memory and training arithmetic.

Standard library only, deterministic (seeded). Builds on 03-llm-concepts/code/llm_lora_math.py.

Run:  python ft_lora_params.py
"""
import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------- parameter counts
def full_params(d_in, d_out):
    return d_in * d_out


def lora_params(d_in, d_out, r):
    """B is d_out x r and A is r x d_in."""
    return r * (d_in + d_out)


@dataclass(frozen=True)
class ToyTransformer:
    """A 7B-class decoder-only layout (gated feed-forward with 3 matrices). Illustrative sizes."""
    d_model: int = 4096
    n_layers: int = 32
    d_ffn: int = 11008
    vocab: int = 32000

    def module_shapes(self):
        d, f = self.d_model, self.d_ffn
        return {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d),
                "gate": (d, f), "up": (d, f), "down": (f, d)}

    def total_params(self):
        per_layer = sum(i * o for i, o in self.module_shapes().values())
        return self.n_layers * per_layer + 2 * self.vocab * self.d_model  # input + output embeddings

    def lora_trainable(self, targets, r):
        shapes = self.module_shapes()
        return self.n_layers * sum(lora_params(*shapes[t], r) for t in targets)


# ---------------------------------------------------------------- linear algebra
def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def matvec(m, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def rank(m, tol=1e-9):
    """Rank by Gaussian elimination with partial pivoting."""
    a = [row[:] for row in m]
    rows, cols = len(a), len(a[0])
    rk, r = 0, 0
    for c in range(cols):
        if r >= rows:
            break
        piv = max(range(r, rows), key=lambda i: abs(a[i][c]))
        if abs(a[piv][c]) <= tol:
            continue
        a[r], a[piv] = a[piv], a[r]
        for i in range(r + 1, rows):
            f = a[i][c] / a[r][c]
            if f:
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        r += 1
        rk += 1
    return rk


def random_matrix(rows, cols, rng, scale=1.0):
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def merged_weight(W, B, A, alpha, r):
    """W' = W + (alpha / r) * B A"""
    BA = matmul(B, A)
    s = alpha / r
    return [[w + s * u for w, u in zip(rw, ru)] for rw, ru in zip(W, BA)]


def adapter_output(W, B, A, alpha, r, x):
    """W x + (alpha / r) * B (A x)"""
    base = matvec(W, x)
    delta = matvec(B, matvec(A, x))
    s = alpha / r
    return [b + s * d for b, d in zip(base, delta)]


# ---------------------------------------------------------------- memory arithmetic
GB = 1e9


def full_ft_gb(params, bytes_per_param=12):
    """Rule of thumb: 16-bit weight 2 + 16-bit grad 2 + Adam two 32-bit states 8 = 12 bytes."""
    return params * bytes_per_param / GB


def lora_gb(params, trainable, weight_bytes=2.0, trainable_bytes=2 + 8 + 4):
    """Frozen base at `weight_bytes`, plus per trainable parameter: 16-bit grad (2) + Adam (8)
    + a 32-bit master copy of the adapter (4). Activations are NOT counted."""
    return (params * weight_bytes + trainable * trainable_bytes) / GB


def nf4_bits_per_param(block=64, const_bits=32, double_quant=False, dq_block=256, dq_bits=8):
    """Storage per base weight: 4 bits plus the share of the per-block quantization constants.

    Without double quantization each block of 64 weights has one 32-bit constant: 32/64 = 0.5 bits.
    Double quantization stores those constants in 8 bits (blocks of 256) plus a 32-bit constant for
    each of those blocks: 8/64 + 32/(64*256) ~ 0.127 bits (the figure given in the QLoRA paper).
    """
    if not double_quant:
        return 4 + const_bits / block
    return 4 + dq_bits / block + const_bits / (block * dq_block)


def qlora_gb(params, trainable, double_quant=True):
    base = params * nf4_bits_per_param(double_quant=double_quant) / 8
    return (base + trainable * (2 + 8 + 4)) / GB


# ---------------------------------------------------------------- training arithmetic
def effective_batch(micro_batch, accum_steps, devices=1):
    return micro_batch * accum_steps * devices


def steps_per_epoch(n_examples, eff_batch):
    return math.ceil(n_examples / eff_batch)


def lr_at(step, total_steps, peak_lr, warmup_steps, final_frac=0.1):
    """Linear warmup then cosine decay down to final_frac * peak_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    floor = peak_lr * final_frac
    return floor + (peak_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- loss curves
def simulate_curves(epochs=8, seed=0):
    """ILLUSTRATIVE per-epoch (train_loss, val_loss): train falls steadily; val falls, then turns up.

    Made-up shapes plus seeded noise, to practise reading curves. Not a real training run.
    """
    rng = random.Random(seed)
    train, val = [], []
    for e in range(epochs):
        train.append(2.0 * math.exp(-0.45 * e) + 0.1 + rng.uniform(-0.02, 0.02))
        val.append(2.0 * math.exp(-0.45 * e) + 0.35 + 0.045 * max(0, e - 3) ** 2 + rng.uniform(-0.02, 0.02))
    return train, val


def best_epoch(val_losses):
    return min(range(len(val_losses)), key=lambda i: val_losses[i])


def early_stop_epoch(val_losses, patience=2):
    """First epoch at which training would stop: `patience` epochs in a row without a new best."""
    best, since = float("inf"), 0
    for i, v in enumerate(val_losses):
        if v < best:
            best, since = v, 0
        else:
            since += 1
            if since >= patience:
                return i
    return None


def overfit_gap(train_losses, val_losses):
    return [v - t for t, v in zip(train_losses, val_losses)]


# ---------------------------------------------------------------- loss masking
IGNORE_INDEX = -100


def build_labels(tokens, assistant_mask):
    """Labels for next-token training: the token id where the mask is 1, else -100 (ignored).

    `tokens` and `assistant_mask` are aligned; mask 1 means "this token belongs to the assistant reply".
    (Frameworks shift labels by one position inside the loss; here labels[i] is the target for tokens[i].)
    """
    if len(tokens) != len(assistant_mask):
        raise ValueError("tokens and mask must have the same length")
    return [t if m else IGNORE_INDEX for t, m in zip(tokens, assistant_mask)]


if __name__ == "__main__":
    d, r = 4096, 8
    print(f"one {d}x{d} matrix: full {full_params(d, d):,} | LoRA r={r} {lora_params(d, d, r):,} "
          f"| {lora_params(d, d, r) / full_params(d, d):.4%}")
    print(f"rectangular 4096x11008, r=8: full {full_params(4096, 11008):,} | LoRA {lora_params(4096, 11008, 8):,}")

    m = ToyTransformer()
    total = m.total_params()
    print(f"\ntoy 7B-class model: {total:,} parameters")
    for name, targets, rr in [("q,v r=8", ["q", "v"], 8), ("q,v r=16", ["q", "v"], 16),
                              ("all attention r=16", ["q", "k", "v", "o"], 16),
                              ("all linear r=16", list(m.module_shapes()), 16)]:
        t = m.lora_trainable(targets, rr)
        print(f"  {name:20s} trainable {t:>11,} = {t / total:.3%}")

    rng = random.Random(0)
    print("\nrank proof (d=12, random Gaussian B and A):")
    for rr in (1, 2, 4, 8):
        BA = matmul(random_matrix(12, rr, rng), random_matrix(rr, 12, rng))
        print(f"  r={rr}: rank(BA) = {rank(BA)}")
    print("  rank of a random 12x12 matrix:", rank(random_matrix(12, 12, rng)))

    d, rr, alpha = 16, 4, 8
    W, B, A = random_matrix(d, d, rng), random_matrix(d, rr, rng), random_matrix(rr, d, rng)
    x = [rng.gauss(0, 1) for _ in range(d)]
    lhs = matvec(merged_weight(W, B, A, alpha, rr), x)
    rhs = adapter_output(W, B, A, alpha, rr, x)
    print(f"\nmerge check max |(W+sBA)x - (Wx + sB(Ax))| = {max(abs(a - b) for a, b in zip(lhs, rhs)):.2e}")

    t = m.lora_trainable(["q", "v"], 8)
    print(f"\nmemory, 7B-class (weights, grads, optimizer; no activations):")
    print(f"  full fine-tune  : {full_ft_gb(total):.1f} GB")
    print(f"  LoRA, 16-bit    : {lora_gb(total, t):.2f} GB")
    print(f"  NF4 bits/param  : {nf4_bits_per_param():.3f} plain, {nf4_bits_per_param(double_quant=True):.3f} double-quant")
    print(f"  QLoRA, NF4+DQ   : {qlora_gb(total, t):.2f} GB")

    eb = effective_batch(4, 8, 1)
    spe = steps_per_epoch(1000, eb)
    print(f"\ntraining: micro 4 x accum 8 = effective batch {eb}; 1000 examples -> {spe} steps/epoch; 3 epochs = {3 * spe}")
    total_steps, warm = 3 * spe, 10
    print("lr at steps:", {s: round(lr_at(s, total_steps, 2e-4, warm), 6) for s in (0, 4, 9, 10, 50, total_steps - 1)})
    print("\nlabels:", build_labels([5, 6, 7, 8, 9], [0, 0, 1, 1, 1]))

    tr, va = simulate_curves(8, seed=0)
    print("\nILLUSTRATIVE loss curves (made up, seeded):")
    for e, (a, b) in enumerate(zip(tr, va), start=1):
        print(f"  epoch {e}: train {a:.3f} | val {b:.3f} | gap {b - a:+.3f}")
    be = best_epoch(va)
    print(f"best validation epoch: {be + 1}; early stop (patience 2) fires at epoch {early_stop_epoch(va) + 1}")
