import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prod_task_queue import TaskQueue


def run(q):
    q.start()
    q.join()
    q.stop()


def test_results():
    q = TaskQueue(workers=3)

    @q.task
    def double(x):
        return x * 2

    ids = [q.submit("double", i) for i in range(10)]
    run(q)
    assert [q.results[i] for i in ids] == [i * 2 for i in range(10)]
    assert q.dead_letters == []


def test_retry_then_success():
    q = TaskQueue(workers=1, max_retries=2)
    calls = []

    @q.task
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return "ok"

    tid = q.submit("flaky")
    run(q)
    assert q.results[tid] == "ok" and len(calls) == 3 and not q.dead_letters


def test_dead_letter_after_retries():
    sleeps = []
    q = TaskQueue(workers=1, max_retries=2, backoff=1.0, sleep=sleeps.append)

    @q.task
    def always_fails():
        raise ValueError("nope")

    tid = q.submit("always_fails")
    run(q)
    assert tid not in q.results
    assert len(q.dead_letters) == 1 and q.dead_letters[0][0] == tid
    assert "nope" in q.dead_letters[0][2]
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_unknown_task_goes_to_dead_letters():
    q = TaskQueue(workers=1, max_retries=0)
    q.submit("missing")
    run(q)
    assert len(q.dead_letters) == 1
