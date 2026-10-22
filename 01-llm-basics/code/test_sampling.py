import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sampling_demo import LOGITS, softmax, top_p_filter  # noqa: E402


def test_probabilities_sum_to_one():
    assert math.isclose(sum(softmax(LOGITS, 1.0)), 1.0)


def test_low_temperature_is_sharper():
    assert max(softmax(LOGITS, 0.2)) > max(softmax(LOGITS, 2.0))


def test_top_p_keeps_best_token_and_renormalises():
    kept = top_p_filter(softmax(LOGITS, 1.0), 0.5)
    assert "Paris" in kept and math.isclose(sum(kept.values()), 1.0)
