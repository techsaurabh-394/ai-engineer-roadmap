import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_decoding as ld  # noqa: E402


def test_table_rows_sum_to_one():
    for row in ld.NEXT.values():
        assert math.isclose(sum(row.values()), 1.0)


def test_greedy_picks_local_best():
    assert ld.greedy() == ["the", "cat", "sat", ld.END]


def test_beam_search_finds_higher_probability_sequence_than_greedy():
    seq, lp = ld.beam_search(beam_width=2)
    assert seq == ["a", "dog", "ran", ld.END]
    assert math.isclose(math.exp(lp), 0.4 * 0.9 * 0.5)
    assert lp > ld.sequence_logprob(ld.greedy())


def test_beam_width_one_equals_greedy():
    assert ld.beam_search(beam_width=1)[0] == ld.greedy()


def test_top_k_keeps_k_tokens():
    kept = ld.top_k_filter(ld.NEXT["<start>"], 2)
    assert set(kept) == {"the", "a"} and math.isclose(sum(kept.values()), 1.0)


def test_top_p_keeps_smallest_nucleus():
    assert set(ld.top_p_filter(ld.NEXT["<start>"], 0.5)) == {"the"}
    assert set(ld.top_p_filter(ld.NEXT["<start>"], 0.9)) == {"the", "a"}
    assert set(ld.top_p_filter(ld.NEXT["<start>"], 1.0)) == {"the", "a", "my"}


def test_temperature_sharpens_and_flattens():
    base = ld.NEXT["<start>"]
    assert ld.apply_temperature(base, 0.5)["the"] > base["the"] > ld.apply_temperature(base, 2.0)["the"]
    assert math.isclose(ld.apply_temperature(base, 1.0)["the"], base["the"])


def test_sampling_is_reproducible_and_ends():
    a = ld.sample(k=2, rng=random.Random(42))
    b = ld.sample(k=2, rng=random.Random(42))
    assert a == b and a[-1] == ld.END


def test_top_k_one_sampling_is_greedy():
    assert ld.sample(k=1, rng=random.Random(7)) == ld.greedy()


def test_kv_cache_counts():
    assert ld.count_kv_computations(10, use_cache=False) == 55
    assert ld.count_kv_computations(10, use_cache=True) == 10
