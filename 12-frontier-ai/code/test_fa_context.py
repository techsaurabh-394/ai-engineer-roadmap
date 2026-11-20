import pytest

import fa_context as m


def test_prefix_stops_at_first_difference():
    assert m.prefix_cache_hits([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert m.prefix_cache_hits([1, 2], [1, 2, 3]) == 2
    assert m.prefix_cache_hits([5, 2], [6, 2]) == 0


def test_request_cost_arithmetic():
    assert m.request_cost(1_000_000, 0) == pytest.approx(3.0)
    assert m.request_cost(1_000_000, 1_000_000) == pytest.approx(0.3)
    assert m.request_cost(1_000_000, 0, write_tokens=1_000_000) == pytest.approx(3.75)


def test_caching_makes_long_chats_cheaper():
    plain = m.conversation_cost(6000, 400, 20, cached=False)
    cached = m.conversation_cost(6000, 400, 20, cached=True)
    assert cached < plain / 3


def test_first_turn_same_with_or_without_cache():
    assert m.conversation_cost(6000, 400, 1, True) == pytest.approx(m.conversation_cost(6000, 400, 1, False))


def test_break_even_below_one_read():
    assert m.break_even_reads() == pytest.approx(0.75 / 2.7)
    assert m.break_even_reads() < 1
