"""Durable agent runs: checkpoints, retries with backoff and idempotency keys.

A run is a list of steps. Some steps have side effects (send email, charge card). We crash the
run part-way and resume, and count how many times each side effect really happened.

Run:  python ae_durable.py
"""


class World:
    """The outside world: counts real side effects, keyed by idempotency key if the step sends one."""

    def __init__(self):
        self.effects = []
        self.seen_keys = set()

    def do(self, name, key=None):
        if key is not None and key in self.seen_keys:
            return "duplicate-ignored"
        if key is not None:
            self.seen_keys.add(key)
        self.effects.append(name)
        return "done"


def run(steps, world, checkpoint, crash_at=None, use_keys=True, use_checkpoint=True):
    """Execute steps in order. checkpoint is a dict {"done": n}. crash_at: raise after the side
    effect of step index crash_at but BEFORE it is checkpointed (the nasty moment)."""
    start = checkpoint.get("done", 0) if use_checkpoint else 0
    for i in range(start, len(steps)):
        name = steps[i]
        world.do(name, key=f"run1:{i}:{name}" if use_keys else None)
        if crash_at == i:
            raise RuntimeError(f"crashed after step {i}")
        checkpoint["done"] = i + 1
    return checkpoint


def backoff_waits(attempts, base=1.0, factor=2.0, cap=30.0):
    """Wait before each retry: base * factor**n, capped (no jitter, so it is testable)."""
    return [min(cap, base * factor ** n) for n in range(attempts)]


def retry(fn, attempts=4, retriable=(TimeoutError,)):
    """Call fn until it works. Returns (result, waits_used). Non-retriable errors propagate."""
    waits = backoff_waits(attempts)
    used = []
    for n in range(attempts):
        try:
            return fn(n), used
        except retriable:
            if n == attempts - 1:
                raise
            used.append(waits[n])
    raise AssertionError("unreachable")


if __name__ == "__main__":
    steps = ["read ticket", "send email", "charge card", "update crm"]
    for label, keys, ckpt in (("no key, no checkpoint", False, False), ("checkpoint only", False, True),
                              ("checkpoint + idempotency keys", True, True), ("keys only", True, False)):
        w, cp = World(), {}
        try:
            run(steps, w, cp, crash_at=2, use_keys=keys, use_checkpoint=ckpt)
        except RuntimeError:
            pass
        run(steps, w, cp, use_keys=keys, use_checkpoint=ckpt)       # resume
        print(f"{label:<30} charge card happened {w.effects.count('charge card')}x, effects total {len(w.effects)}")
    print("backoff waits (s):", backoff_waits(6))

    def flaky(n):
        if n < 2:
            raise TimeoutError("slow")
        return "ok"
    print("retry:", retry(flaky))
