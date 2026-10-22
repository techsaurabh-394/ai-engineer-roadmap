"""Vector similarity math for RAG, stdlib only.

dot product, vector length (norm), cosine similarity, Euclidean distance, and
how they relate when vectors are normalized to length 1.

Run:  python rag_vector_math.py
"""
import math


def _check(a, b):
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length ({len(a)} != {len(b)})")


def dot(a, b):
    """Sum of element-wise products. Bigger = more aligned AND/OR longer vectors."""
    _check(a, b)
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    """Length of a vector (Pythagoras in N dimensions)."""
    return math.sqrt(sum(x * x for x in a))


def normalize(a):
    """Scale a vector to length 1. Direction stays the same."""
    n = norm(a)
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return [x / n for x in a]


def cosine_similarity(a, b):
    """cos(angle) between vectors: 1 = same direction, 0 = unrelated, -1 = opposite."""
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def euclidean_distance(a, b):
    """Straight-line distance. 0 = identical. Smaller = closer."""
    _check(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rank(query, vectors, metric="cosine"):
    """Return indexes of `vectors`, best match first, for a given metric."""
    if metric == "cosine":
        scores = [cosine_similarity(query, v) for v in vectors]
        return sorted(range(len(vectors)), key=lambda i: -scores[i])
    if metric == "dot":
        scores = [dot(query, v) for v in vectors]
        return sorted(range(len(vectors)), key=lambda i: -scores[i])
    if metric == "euclidean":
        scores = [euclidean_distance(query, v) for v in vectors]
        return sorted(range(len(vectors)), key=lambda i: scores[i])
    raise ValueError(f"unknown metric: {metric}")


def worked_examples():
    """Small numeric examples used in the topic files. Returns a dict of results."""
    a, b = [1, 2, 3], [2, 4, 6]  # b is a scaled copy of a: same direction
    c = [3, 0, -1]               # dot(a, c) == 0: perpendicular
    return {
        "dot(a,b)": dot(a, b),                          # 1*2 + 2*4 + 3*6 = 28
        "cos(a,b)": cosine_similarity(a, b),            # 1.0
        "euclid(a,b)": euclidean_distance(a, b),        # sqrt(1+4+9) = 3.742
        "dot(a,c)": dot(a, c),                          # 3 + 0 - 3 = 0
        "cos(a,c)": cosine_similarity(a, c),            # 0.0
        "unit_identity": (                              # for unit vectors:
            euclidean_distance(normalize(a), normalize(c)) ** 2,  # ||u-v||^2
            2 - 2 * cosine_similarity(a, c),                      # == 2 - 2*cos
        ),
    }


if __name__ == "__main__":
    for name, value in worked_examples().items():
        print(f"{name:14} = {value}")
    docs = [[0.9, 0.1], [5.0, 5.0], [0.1, 0.9]]
    q = [1.0, 0.2]
    for m in ("cosine", "dot", "euclidean"):
        print(f"ranking by {m:9}: {rank(q, docs, m)}")
    print("Note how 'dot' prefers the long vector [5,5]; cosine ignores length.")
