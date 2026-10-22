import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from rag_chunkers import (  # noqa: E402
    SAMPLE,
    fixed_size_chunks,
    jaccard,
    parent_child_chunks,
    recursive_chunks,
    semantic_chunks,
    sliding_window_chunks,
)


def test_fixed_size_no_overlap_rebuilds_text():
    text = "abcdefghij" * 7
    chunks = fixed_size_chunks(text, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == text


def test_fixed_size_overlap_shares_characters():
    chunks = fixed_size_chunks("0123456789", size=4, overlap=2)
    assert chunks == ["0123", "2345", "4567", "6789"]


def test_fixed_size_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunks("abc", size=3, overlap=3)


def test_recursive_respects_size_and_keeps_all_text():
    chunks = recursive_chunks(SAMPLE, 120)
    assert all(len(c) <= 120 for c in chunks)
    # Whitespace-only fragments are dropped; every real character is kept, in order.
    assert "".join(chunks).split() == SAMPLE.split()


def test_recursive_prefers_paragraph_boundaries():
    chunks = recursive_chunks(SAMPLE, 120)
    assert chunks[0].startswith("Docker images")
    assert chunks[-1].startswith("Refresh tokens")


def test_recursive_hard_cuts_long_words():
    chunks = recursive_chunks("x" * 25, 10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_sliding_window_overlap_and_coverage():
    text = " ".join(str(i) for i in range(10))
    chunks = sliding_window_chunks(text, window=4, stride=2)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "2 3 4 5"
    assert chunks[-1].endswith("9")


def test_sliding_window_empty():
    assert sliding_window_chunks("", 4, 2) == []


def test_jaccard_bounds():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("cat", "dog") == 0.0


def test_semantic_splits_on_topic_change():
    text = "Cats purr softly. Cats sleep a lot. Kubernetes schedules pods."
    chunks = semantic_chunks(text, threshold=0.1)
    assert chunks == ["Cats purr softly. Cats sleep a lot.", "Kubernetes schedules pods."]


def test_parent_child_links():
    parents, children = parent_child_chunks(SAMPLE, parent_size=200, child_size=80)
    assert children
    for ch in children:
        assert ch["parent_id"] in parents
        assert ch["text"] in parents[ch["parent_id"]]
        assert len(ch["text"]) <= 80
