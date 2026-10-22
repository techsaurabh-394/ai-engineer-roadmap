import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from prod_prompt_registry import PromptRegistry


def make():
    r = PromptRegistry()
    r.register("qa", "v1 text")
    r.register("qa", "v2 text")
    return r


def test_versions_and_hash():
    r = make()
    assert r.get("qa")["version"] == 2
    assert r.get("qa", 1)["text"] == "v1 text"
    assert r.get("qa", 1)["hash"] != r.get("qa", 2)["hash"]
    assert len(r.get("qa")["hash"]) == 12


def test_same_text_is_noop():
    r = make()
    assert r.register("qa", "v2 text") == 2
    assert len(r.history("qa")) == 2


def test_rollback():
    r = make()
    assert r.rollback("qa") == 1
    assert r.get("qa")["text"] == "v1 text"
    with pytest.raises(ValueError):
        r.rollback("qa")
    assert r.register("qa", "v3 text") == 3  # history is kept


def test_ab_is_sticky_and_splits():
    r = make()
    assert r.pick_ab("qa", 1, 2, "u1") == r.pick_ab("qa", 1, 2, "u1")
    picks = [r.pick_ab("qa", 1, 2, f"u{i}")["version"] for i in range(400)]
    assert 100 < picks.count(2) < 300
    assert all(r.pick_ab("qa", 1, 2, f"u{i}", percent_b=0)["version"] == 1 for i in range(20))


def test_pick_random_extremes():
    r = make()
    rng = random.Random(0)
    assert r.pick_random("qa", 1, 2, percent_b=100, rng=rng)["version"] == 2
    assert r.pick_random("qa", 1, 2, percent_b=0, rng=rng)["version"] == 1
