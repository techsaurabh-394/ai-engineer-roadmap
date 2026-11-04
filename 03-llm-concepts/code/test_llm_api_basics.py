import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

from llm_api_basics import (  # noqa: E402
    FakeLLM, Flaky, TransientError, cost_estimate, retry_with_backoff,
)

MSGS = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is the capital of France?"},
]


def test_response_has_text_finish_reason_and_usage():
    r = FakeLLM().chat(MSGS)
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert r["choices"][0]["message"]["content"] == "You said: What is the capital of France?"
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"input_tokens": 8, "output_tokens": 8}


def test_max_tokens_cuts_the_reply_and_sets_length():
    r = FakeLLM().chat(MSGS, max_tokens=3)
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["output_tokens"] == 3


def test_bad_role_and_empty_messages_are_rejected():
    with pytest.raises(ValueError):
        FakeLLM().chat([{"role": "robot", "content": "hi"}])
    with pytest.raises(ValueError):
        FakeLLM().chat([])


def test_cost_estimate_uses_price_per_million():
    usage = {"input_tokens": 2000, "output_tokens": 500}
    assert abs(cost_estimate(usage, 3.0, 15.0) - 0.0135) < 1e-12
    assert cost_estimate({"input_tokens": 0, "output_tokens": 0}, 3, 15) == 0


def test_retry_succeeds_after_failures_with_exact_delays():
    waits = []
    flaky = Flaky(FakeLLM(), fail_first=2)
    r = retry_with_backoff(lambda: flaky.chat(MSGS), attempts=4, base_delay=1.0, sleep=waits.append)
    assert r["choices"][0]["finish_reason"] == "stop"
    assert flaky.attempts == 3
    assert waits == [1.0, 2.0]


def test_retry_gives_up_and_raises_after_last_attempt():
    waits = []
    flaky = Flaky(FakeLLM(), fail_first=10, status=503)
    with pytest.raises(TransientError) as e:
        retry_with_backoff(lambda: flaky.chat(MSGS), attempts=3, base_delay=0.5, sleep=waits.append)
    assert e.value.status == 503
    assert flaky.attempts == 3
    assert waits == [0.5, 1.0]  # no wait after the final failure


def test_non_retryable_errors_are_not_retried():
    waits = []
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("400 bad request")

    with pytest.raises(ValueError):
        retry_with_backoff(bad, attempts=5, sleep=waits.append)
    assert len(calls) == 1 and waits == []


def test_stream_chunks_join_to_the_full_reply():
    llm = FakeLLM()
    chunks = list(llm.stream(MSGS))
    full = llm.chat(MSGS)["choices"][0]["message"]["content"]
    assert len(chunks) > 1
    assert "".join(chunks) == full
