import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from rag_vector_math import (  # noqa: E402
    cosine_similarity,
    dot,
    euclidean_distance,
    norm,
    normalize,
    rank,
    worked_examples,
)


def test_dot_worked_example():
    assert dot([1, 2, 3], [2, 4, 6]) == 28


def test_norm_and_normalize():
    assert norm([3, 4]) == 5
    u = normalize([3, 4])
    assert u == [0.6, 0.8]
    assert math.isclose(norm(u), 1.0)


def test_cosine_cases():
    assert math.isclose(cosine_similarity([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert math.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_euclidean():
    assert euclidean_distance([0, 0], [3, 4]) == 5
    assert euclidean_distance([1, 1], [1, 1]) == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


def test_unit_vector_identity():
    ex = worked_examples()
    lhs, rhs = ex["unit_identity"]
    assert math.isclose(lhs, rhs)


def test_normalized_vectors_rank_the_same_for_all_metrics():
    docs = [normalize(v) for v in ([0.9, 0.1], [5.0, 5.0], [0.1, 0.9])]
    q = normalize([1.0, 0.2])
    assert rank(q, docs, "cosine") == rank(q, docs, "dot") == rank(q, docs, "euclidean")


def test_dot_prefers_long_vectors_when_not_normalized():
    docs = [[0.9, 0.1], [5.0, 5.0]]
    assert rank([1.0, 0.2], docs, "dot")[0] == 1
    assert rank([1.0, 0.2], docs, "cosine")[0] == 0
