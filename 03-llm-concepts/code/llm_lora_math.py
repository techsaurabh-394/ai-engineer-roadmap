"""LoRA math with tiny matrices and pure Python lists. Standard library only.

LoRA freezes a weight matrix W (d x d) and learns a small update B @ A, where B is d x r and A is r x d.
The new weight is W' = W + (alpha / r) * B @ A.

Run:  python llm_lora_math.py
"""


def matmul(a, b):
    """Multiply matrix a (n x k) by matrix b (k x m). Both are lists of rows."""
    if len(a[0]) != len(b):
        raise ValueError("shape mismatch")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def matvec(m, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def mat_add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_scale(m, s):
    return [[x * s for x in row] for row in m]


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def low_rank_update(B, A, alpha, r):
    """Return (alpha / r) * B @ A, a d x d matrix whose rank is at most r."""
    return mat_scale(matmul(B, A), alpha / r)


def merge(W, B, A, alpha, r):
    """Fold the adapter into the base weight: W' = W + (alpha / r) * B @ A."""
    return mat_add(W, low_rank_update(B, A, alpha, r))


def adapter_forward(W, B, A, alpha, r, x):
    """Apply the base layer and the adapter separately: W x + (alpha / r) * B (A x)."""
    base = matvec(W, x)
    delta = [(alpha / r) * v for v in matvec(B, matvec(A, x))]
    return vec_add(base, delta)


def full_params(d):
    return d * d


def lora_params(d, r):
    """B is d x r and A is r x d."""
    return 2 * d * r


def rows_proportional(m, tol=1e-9):
    """True if every row is a scalar multiple of the first non-zero row (so the matrix has rank <= 1)."""
    ref = next((row for row in m if any(abs(x) > tol for x in row)), None)
    if ref is None:
        return True
    j = max(range(len(ref)), key=lambda c: abs(ref[c]))
    for row in m:
        c = row[j] / ref[j]
        if any(abs(row[k] - c * ref[k]) > tol for k in range(len(ref))):
            return False
    return True


def finetune_memory_gb(params, weight_bytes=2, grad_bytes=2, optimizer_bytes=8):
    """Rule-of-thumb memory for FULL fine-tuning, in GB (activations not counted).

    weights (16-bit) + gradients (16-bit) + Adam's two 32-bit states (4 + 4 bytes) = 12 bytes per parameter.
    """
    return params * (weight_bytes + grad_bytes + optimizer_bytes) / 1e9


def lora_memory_gb(params, trainable, weight_bytes=2, grad_bytes=2, optimizer_bytes=8):
    """Frozen base weights plus gradients and optimizer state for the small trainable part only."""
    return (params * weight_bytes + trainable * (grad_bytes + optimizer_bytes)) / 1e9


if __name__ == "__main__":
    d, r = 4096, 8
    print(f"one {d}x{d} matrix: full {full_params(d):,} params | LoRA r={r} {lora_params(d, r):,} params | "
          f"{lora_params(d, r) / full_params(d):.4%} of full")

    layers, matrices = 32, 2  # illustrative: adapt 2 matrices (e.g. query and value) in each of 32 layers
    total = layers * matrices * lora_params(d, r)
    print(f"{layers} layers x {matrices} matrices x {lora_params(d, r):,} = {total:,} trainable params")
    print(f"7B full fine-tune ~ {finetune_memory_gb(7e9):.0f} GB | 7B LoRA ~ {lora_memory_gb(7e9, total):.2f} GB (before activations)")

    print("\nTiny 4x4 example, r=2, alpha=4 (scale alpha/r = 2):")
    W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    B = [[1, 0], [0, 1], [1, 1], [0, 2]]
    A = [[1, 0, 2, 0], [0, 1, 0, 1]]
    alpha, rank = 4, 2
    print("B @ A      =", matmul(B, A))
    W2 = merge(W, B, A, alpha, rank)
    print("W' rows    =")
    for row in W2:
        print("  ", row)

    x = [1, 2, 3, 4]
    print("\nW' @ x          =", matvec(W2, x))
    print("adapter forward =", adapter_forward(W, B, A, alpha, rank, x))

    print("\nRank check (r=1): B is a column, A is a row")
    B1 = [[1], [2], [-1], [0]]
    A1 = [[3, 0, 1, 2]]
    U = low_rank_update(B1, A1, alpha=1, r=1)
    for row in U:
        print("  ", row)
    print("all rows proportional (rank <= 1):", rows_proportional(U))
    print("identity matrix rows proportional:", rows_proportional(W))
