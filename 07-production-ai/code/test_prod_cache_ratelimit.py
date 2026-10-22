import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prod_cache_ratelimit import TTLCache, TokenBucket


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_hit_and_expiry():
    c = Clock()
    cache = TTLCache(ttl=10, clock=c)
    cache.set("a", 1)
    assert cache.get("a") == 1
    c.t = 10.1
    assert cache.get("a") is None
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.hit_rate == 0.5


def test_hit_rate_empty():
    assert TTLCache().hit_rate == 0.0


def test_max_items_evicts():
    c = Clock()
    cache = TTLCache(ttl=10, max_items=2, clock=c)
    cache.set("a", 1)
    c.t = 1
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 2
    assert cache.get("a") is None and cache.get("c") == 3


def test_bucket_burst_then_refill():
    c = Clock()
    b = TokenBucket(rate=1, capacity=2, clock=c)
    assert b.allow() and b.allow()
    assert not b.allow()
    c.t = 1.0
    assert b.allow()
    assert not b.allow()


def test_bucket_capped():
    c = Clock()
    b = TokenBucket(rate=10, capacity=2, clock=c)
    c.t = 100
    assert b.allow() and b.allow() and not b.allow()
