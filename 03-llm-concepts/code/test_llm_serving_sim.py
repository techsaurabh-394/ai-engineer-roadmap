import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from llm_serving_sim import (  # noqa: E402
    SAMPLE,
    continuous_batching,
    metrics,
    percentile_nearest_rank,
    run,
    static_batching,
)

FOUR = [(0, 2), (0, 2), (0, 2), (0, 10)]


def test_static_everyone_waits_for_longest():
    assert static_batching(FOUR, 4) == [10, 10, 10, 10]


def test_continuous_releases_short_requests_early():
    assert continuous_batching(FOUR, 4) == [2, 2, 2, 10]


def test_continuous_mean_latency_beats_static_on_hand_example():
    res = run(FOUR, 4)
    assert abs(res["static"]["mean_latency"] - 10.0) < 1e-9
    assert abs(res["continuous"]["mean_latency"] - 4.0) < 1e-9
    assert res["continuous"]["mean_latency"] < res["static"]["mean_latency"]


def test_sample_numbers_by_hand():
    res = run(SAMPLE, 4)
    assert res["static"]["finish"] == [10, 10, 10, 10, 12, 12]
    assert res["continuous"]["finish"] == [2, 2, 2, 10, 4, 4]
    assert abs(res["static"]["mean_latency"] - 64 / 6) < 1e-9
    assert abs(res["continuous"]["mean_latency"] - 4.0) < 1e-9
    assert abs(res["static"]["throughput"] - 20 / 12) < 1e-9
    assert abs(res["continuous"]["throughput"] - 2.0) < 1e-9
    assert abs(res["static"]["utilization"] - 20 / 48) < 1e-9
    assert abs(res["continuous"]["utilization"] - 0.5) < 1e-9
    assert res["continuous"]["utilization"] > res["static"]["utilization"]


def test_every_request_gets_exactly_its_output_tokens():
    requests = [(0, 4), (1, 2), (2, 2), (3, 2), (9, 5)]
    for fn in (static_batching, continuous_batching):
        trace = []
        fn(requests, 2, trace)
        counts = Counter(i for _, i in trace)
        assert [counts[i] for i in range(len(requests))] == [r[1] for r in requests]
        # a request never makes a token before it arrives
        assert all(step >= requests[i][0] for step, i in trace)
        # never more than the slot limit in one step
        per_step = Counter(step for step, _ in trace)
        assert max(per_step.values()) <= 2


def test_staggered_arrivals_by_hand():
    requests = [(0, 4), (1, 2), (2, 2), (3, 2)]
    assert static_batching(requests, 2) == [4, 6, 6, 8]
    assert continuous_batching(requests, 2) == [4, 3, 5, 6]


def test_nearest_rank_percentile():
    assert percentile_nearest_rank([1, 2, 3, 4, 5, 6], 95) == 6
    assert percentile_nearest_rank([1, 2, 3, 4, 5, 6], 50) == 3
    assert percentile_nearest_rank([7], 95) == 7


def test_metrics_shape():
    trace = []
    finish = continuous_batching(FOUR, 4, trace)
    m = metrics(FOUR, finish, 4, trace)
    assert abs(m["throughput"] - 1.6) < 1e-9
    assert m["p95_latency"] == 10
