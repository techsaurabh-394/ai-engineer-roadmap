import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from prod_observability import Tracer, percentile


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_latency_tokens_cost():
    c = FakeClock()
    t = Tracer(clock=c)
    with t.span("llm") as s:
        c.t += 0.25
        s.record_usage(1000, 500, 0.001, 0.002)
    sp = t.spans[0]
    assert sp.latency_ms == pytest.approx(250)
    assert sp.cost == pytest.approx(0.001 + 0.001)
    assert (sp.input_tokens, sp.output_tokens) == (1000, 500)


def test_nesting_and_error():
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("outer"):
            with t.span("inner"):
                raise ValueError("x")
    inner = t.spans[0]
    assert inner.name == "inner" and inner.parent.name == "outer"
    assert inner.error and t.summary()["errors"] == 2


def test_percentile_nearest_rank():
    data = list(range(1, 101))
    assert percentile(data, 50) == 50
    assert percentile(data, 95) == 95
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summary():
    c = FakeClock()
    t = Tracer(clock=c)
    for ms in (100, 200, 300, 400):
        with t.span("llm") as s:
            c.t += ms / 1000
            s.record_usage(10, 5)
    sm = t.summary("llm")
    assert sm["count"] == 4 and sm["p50_ms"] == pytest.approx(200)
    assert sm["p95_ms"] == pytest.approx(400) and sm["input_tokens"] == 40
    assert t.summary("none") == {"count": 0}
