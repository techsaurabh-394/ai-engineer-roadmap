"""A request lifecycle simulator: rate limit, cache, model call (standard library only).

It replays a stream of requests through the TTLCache and TokenBucket from prod_cache_ratelimit.py on a VIRTUAL
clock (no sleeping, fully deterministic). The site's "Request lifecycle lab" runs a JavaScript port of this file.

Illustrative numbers: latency and price are parameters you choose. They are NOT measurements of any provider.

Run:  python prod_request_sim.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prod_cache_ratelimit import TTLCache, TokenBucket  # noqa: E402

QUESTIONS = [
    "What are your opening hours?", "How do I reset my password?", "Where is my order?", "Do you ship abroad?",
    "How do I cancel?", "Is there a student discount?", "Can I change my email?", "What is the refund policy?",
]
PATTERNS = {
    "steady": dict(title="Steady traffic", weights=[8, 5, 3, 2, 1, 1, 1, 1],
                   blurb="One request about every second. Some questions are asked far more than others."),
    "burst": dict(title="Bursts", weights=[8, 5, 3, 2, 1, 1, 1, 1],
                  blurb="Requests arrive in rushes of about ten per second, with quiet gaps between them."),
    "popular": dict(title="One question goes viral", weights=[30, 1, 1, 1, 1, 1, 1, 1],
                    blurb="Almost everyone asks the same thing. Caching shines here."),
    "unique": dict(title="Every question is different", weights=[1] * 80,
                   blurb="Eighty different questions asked evenly: a repeat is rare, so caching helps much less."),
}


def question_text(q):
    """The text of question number q: eight real ones, then generated ones for the 'unique' pattern."""
    return QUESTIONS[q] if q < len(QUESTIONS) else f"Question number {q + 1}"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_traffic(pattern, n, seed):
    """[(time in seconds, question index)] from a seeded generator."""
    rng = random.Random(seed)
    weights = PATTERNS[pattern]["weights"]
    ids = list(range(len(weights)))
    t, out = 0.0, []
    for k in range(n):
        q = rng.choices(ids, weights)[0]
        if pattern == "burst":
            gap = 0.05 + rng.random() * 0.1 if (k + 1) % 12 else 6.0 + rng.random() * 4.0
        else:
            gap = 0.5 + rng.random() * 1.5
        t = t + gap
        out.append([t, q])
    return out


def simulate(traffic, ttl=30.0, rate=1.0, capacity=3, order="limit_first", llm_ms=1800, cache_ms=15, price=0.01):
    """Replay the traffic. order: 'limit_first' (every request costs a token) or 'cache_first' (hits are free)."""
    clock = Clock()
    cache = TTLCache(ttl=ttl, clock=clock)
    bucket = TokenBucket(rate=rate, capacity=capacity, clock=clock)
    events = []
    for t, q in traffic:
        clock.t = t
        key = question_text(q)
        if order == "limit_first":
            if not bucket.allow():
                events.append({"t": t, "q": q, "kind": "limited", "ms": 0})
                continue
            if cache.get(key) is not None:
                events.append({"t": t, "q": q, "kind": "hit", "ms": cache_ms})
            else:
                cache.set(key, "answer")
                events.append({"t": t, "q": q, "kind": "llm", "ms": llm_ms})
        else:
            if cache.get(key) is not None:
                events.append({"t": t, "q": q, "kind": "hit", "ms": cache_ms})
            elif not bucket.allow():
                events.append({"t": t, "q": q, "kind": "limited", "ms": 0})
            else:
                cache.set(key, "answer")
                events.append({"t": t, "q": q, "kind": "llm", "ms": llm_ms})
    hits = sum(1 for e in events if e["kind"] == "hit")
    llm = sum(1 for e in events if e["kind"] == "llm")
    limited = sum(1 for e in events if e["kind"] == "limited")
    served = hits + llm
    return {
        "events": events,
        "summary": {"total": len(events), "hits": hits, "llm": llm, "limited": limited,
                    "hit_rate": hits / served if served else 0.0,
                    "cost": llm * price, "saved": hits * price,
                    "avg_ms": (hits * cache_ms + llm * llm_ms) / served if served else 0.0},
    }


if __name__ == "__main__":
    for name in PATTERNS:
        tr = make_traffic(name, 60, 7)
        for order in ("limit_first", "cache_first"):
            s = simulate(tr, order=order)["summary"]
            print(f"{name:8} {order:11} hits={s['hits']:2} llm={s['llm']:2} limited={s['limited']:2} "
                  f"hit_rate={s['hit_rate']:.2f} cost=${s['cost']:.2f} saved=${s['saved']:.2f} avg_ms={s['avg_ms']:.0f}")
