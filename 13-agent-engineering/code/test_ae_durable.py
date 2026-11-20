import pytest

import ae_durable as m

STEPS = ["a", "b", "c", "d"]


def crash_and_resume(keys, ckpt):
    w, cp = m.World(), {}
    with pytest.raises(RuntimeError):
        m.run(STEPS, w, cp, crash_at=2, use_keys=keys, use_checkpoint=ckpt)
    m.run(STEPS, w, cp, use_keys=keys, use_checkpoint=ckpt)
    return w


def test_naive_resume_repeats_side_effects():
    w = crash_and_resume(False, False)
    assert w.effects.count("c") == 2 and w.effects.count("a") == 2


def test_checkpoint_alone_repeats_the_crashed_step():
    w = crash_and_resume(False, True)
    assert w.effects.count("c") == 2 and w.effects.count("a") == 1


def test_keys_and_checkpoint_do_each_effect_once():
    w = crash_and_resume(True, True)
    assert sorted(w.effects) == STEPS


def test_keys_alone_also_prevent_duplicates():
    w = crash_and_resume(True, False)
    assert sorted(w.effects) == STEPS


def test_backoff():
    assert m.backoff_waits(6) == [1, 2, 4, 8, 16, 30]


def test_retry_success_and_exhaustion():
    def flaky(n):
        if n < 2:
            raise TimeoutError()
        return "ok"
    assert m.retry(flaky) == ("ok", [1.0, 2.0])
    with pytest.raises(TimeoutError):
        m.retry(lambda n: (_ for _ in ()).throw(TimeoutError()), attempts=2)


def test_non_retriable_propagates_immediately():
    calls = []

    def bad(n):
        calls.append(n)
        raise ValueError("no")
    with pytest.raises(ValueError):
        m.retry(bad)
    assert calls == [0]
