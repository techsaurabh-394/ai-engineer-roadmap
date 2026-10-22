"""Tiny tracing: spans record latency, tokens and cost. Stdlib only."""
import math
import time
from contextlib import contextmanager


class Span:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.latency_ms = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.error = None
        self.attrs = {}

    def record_usage(self, input_tokens, output_tokens, price_in_per_1k=0.0, price_out_per_1k=0.0):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += input_tokens / 1000 * price_in_per_1k + output_tokens / 1000 * price_out_per_1k


class Tracer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.spans = []
        self._stack = []

    @contextmanager
    def span(self, name, **attrs):
        s = Span(name, self._stack[-1] if self._stack else None)
        s.attrs.update(attrs)
        self._stack.append(s)
        start = self.clock()
        try:
            yield s
        except Exception as exc:
            s.error = repr(exc)
            raise
        finally:
            s.latency_ms = (self.clock() - start) * 1000
            self._stack.pop()
            self.spans.append(s)

    def _named(self, name):
        return [s for s in self.spans if name is None or s.name == name]

    def summary(self, name=None):
        spans = self._named(name)
        if not spans:
            return {"count": 0}
        lat = sorted(s.latency_ms for s in spans)
        return {
            "count": len(spans),
            "errors": sum(1 for s in spans if s.error),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "input_tokens": sum(s.input_tokens for s in spans),
            "output_tokens": sum(s.output_tokens for s in spans),
            "total_cost": sum(s.cost for s in spans),
        }


def percentile(sorted_values, p):
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        raise ValueError("empty data")
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


if __name__ == "__main__":
    t = Tracer()
    with t.span("request"):
        with t.span("llm_call", model="demo") as s:
            s.record_usage(500, 100, 0.001, 0.002)
    print(t.summary("llm_call"))
