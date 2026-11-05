import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_e2e_walkthrough as e2e  # noqa: E402


def test_tokenize_and_ids():
    assert e2e.tokenize("The cat sat on the mat.") == ["the", "cat", "sat", "on", "the", "mat", "."]
    assert e2e.to_ids(["the", "cat", "sat"]) == [0, 1, 2]


def test_unknown_token_raises():
    try:
        e2e.to_ids(["banana"])
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_embedding_shapes_and_lookup():
    x = e2e.embed([1, 2])
    assert x == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert len(e2e.EMBEDDING) == len(e2e.VOCAB) == 8


def test_position_changes_identical_tokens():
    x = e2e.add_position(e2e.embed([0, 0]))  # "the the"
    assert x[0] != x[1]
    assert x[0][1] == 0.1  # position 0: cos(0) = 1, scaled by POS_SCALE


def test_forward_shapes_and_probabilities():
    s = e2e.forward(e2e.to_ids(e2e.tokenize("the cat sat on the")))
    assert len(s["embeddings"]) == 5 and len(s["embeddings"][0]) == e2e.D_MODEL
    assert len(s["block_outputs"]) == e2e.N_BLOCKS
    assert all(len(b) == 5 and len(b[0]) == 4 for b in s["block_outputs"])
    assert len(s["logits"]) == len(e2e.VOCAB)  # ONE score per vocab word, for the last position only
    assert math.isclose(sum(s["probs"]), 1.0)
    assert all(math.isclose(sum(row), 1.0) for row in s["attention_weights"][0])


def test_attention_is_causal_inside_blocks():
    s = e2e.forward([0, 1, 2])
    for w in s["attention_weights"]:
        assert w[0][1] == 0.0 and w[0][2] == 0.0 and w[1][2] == 0.0


def test_next_token_after_on_the_is_mat():
    s = e2e.forward(e2e.to_ids(e2e.tokenize("the cat sat on the")))
    assert e2e.VOCAB[s["next_id"]] == "mat"
    assert s["probs"][s["next_id"]] == max(s["probs"])


def test_forward_is_deterministic():
    ids = [0, 1, 2, 3, 0]
    assert e2e.forward(ids)["logits"] == e2e.forward(ids)["logits"]


def test_generate_loop_appends_one_token_per_step_and_stops_at_period():
    final, steps = e2e.generate("the cat sat on the", max_new_tokens=5)
    assert final == ["the", "cat", "sat", "on", "the", "mat", "."]
    assert len(steps) == 2  # one forward pass per new token, stopped at "."


def test_generate_respects_max_new_tokens():
    final, steps = e2e.generate("the cat", max_new_tokens=1)
    assert len(final) == 3 and len(steps) == 1


def test_parameter_count():
    # embedding 8x4 + Wq 4x4 + Wk 4x4 + Wv 4x4 + Wff 4x4 + Wout 4x8
    assert e2e.parameter_count() == 32 + 16 * 4 + 32
