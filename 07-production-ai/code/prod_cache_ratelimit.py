"""TTL cache with hit-rate stats, and a token-bucket rate limiter. Stdlib only."""
import time


class TTLCache:
    """Exact-match cache. Entries expire `ttl` seconds after being set."""

    def __init__(self, ttl=60.0, max_items=1000, clock=time.monotonic):
        self.ttl = ttl
        self.max_items = max_items
        self.clock = clock
        self._data = {}  # key -> (expires_at, value)
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        item = self._data.get(key)
        if item is not None:
            expires_at, value = item
            if self.clock() < expires_at:
                self.hits += 1
                return value
            del self._data[key]
        self.misses += 1
        return default

    def set(self, key, value, ttl=None):
        if key not in self._data and len(self._data) >= self.max_items:
            self._evict()
        self._data[key] = (self.clock() + (self.ttl if ttl is None else ttl), value)

    def _evict(self):
        now = self.clock()
        for k in [k for k, (exp, _) in self._data.items() if exp <= now]:
            del self._data[k]
        if len(self._data) >= self.max_items:  # still full: drop oldest-expiring
            oldest = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]

    def __len__(self):
        return len(self._data)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TokenBucket:
    """Allows `rate` requests per second on average, with bursts up to `capacity`."""

    def __init__(self, rate, capacity, clock=time.monotonic):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock
        self.tokens = float(capacity)
        self.last = clock()

    def allow(self, cost=1.0):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


if __name__ == "__main__":
    cache = TTLCache(ttl=5)
    for q in ["hi", "hi", "bye", "hi"]:
        if cache.get(q) is None:
            cache.set(q, f"answer to {q}")
    print("hit rate:", cache.hit_rate)
    bucket = TokenBucket(rate=1, capacity=3)
    print([bucket.allow() for _ in range(5)])
