"""The shape of an LLM API call, with no network. Standard library only.

FakeLLM returns a dict shaped like a typical chat response, so you can practise
retries, streaming and cost maths for free. Real providers differ in field
names, so check their docs before copying this shape.
"""
import time


class TransientError(Exception):
    """Stands in for an HTTP 429 or 5xx error that is worth retrying."""

    def __init__(self, status=503):
        super().__init__(f"HTTP {status}")
        self.status = status


def count_tokens(text):
    """Very rough token count: one token per word. Real tokenizers differ."""
    return len(text.split())


class FakeLLM:
    """A pretend model that echoes the last user message."""

    def __init__(self, model="fake-1"):
        self.model = model
        self.calls = 0

    def _reply_text(self, messages):
        last_user = [m["content"] for m in messages if m["role"] == "user"]
        return "You said: " + (last_user[-1] if last_user else "nothing")

    def chat(self, messages, max_tokens=50, temperature=0.0):
        self.calls += 1
        if not messages:
            raise ValueError("messages must not be empty")
        for m in messages:
            if m.get("role") not in ("system", "user", "assistant"):
                raise ValueError(f"bad role: {m.get('role')!r}")
        words = self._reply_text(messages).split()
        finish = "stop"
        if len(words) > max_tokens:
            words, finish = words[:max_tokens], "length"
        text = " ".join(words)
        prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
        return {
            "model": self.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": len(words),
            },
        }

    def stream(self, messages, max_tokens=50, temperature=0.0):
        """Yield the reply a few characters at a time, like a streaming API."""
        text = self.chat(messages, max_tokens, temperature)["choices"][0]["message"]["content"]
        for i in range(0, len(text), 4):
            yield text[i:i + 4]


class Flaky:
    """Wrap a client so that its first `fail_first` chat calls raise."""

    def __init__(self, inner, fail_first=2, status=429):
        self.inner, self.fail_first, self.status = inner, fail_first, status
        self.attempts = 0

    def chat(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_first:
            raise TransientError(self.status)
        return self.inner.chat(*args, **kwargs)


def retry_with_backoff(fn, attempts=4, base_delay=1.0, sleep=time.sleep,
                       retry_on=(TransientError,)):
    """Call fn(). On a retryable error wait base_delay * 2**n, then try again.

    Delays are 1, 2, 4... with no jitter, so tests are exact. Real code should
    add random jitter so many clients do not retry at the same moment.
    """
    for n in range(attempts):
        try:
            return fn()
        except retry_on:
            if n == attempts - 1:
                raise
            sleep(base_delay * 2 ** n)
    raise ValueError("attempts must be at least 1")


def cost_estimate(usage, price_in_per_m, price_out_per_m):
    """Cost in currency units. Prices are per one million tokens."""
    return (usage["input_tokens"] * price_in_per_m
            + usage["output_tokens"] * price_out_per_m) / 1_000_000


if __name__ == "__main__":
    llm = FakeLLM()
    msgs = [
        {"role": "system", "content": "You are a short, polite assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    resp = llm.chat(msgs, max_tokens=50, temperature=0.0)
    print("reply:", resp["choices"][0]["message"]["content"])
    print("usage:", resp["usage"])
    print("cost :", cost_estimate(resp["usage"], 3.0, 15.0))

    short = llm.chat(msgs, max_tokens=3)
    print("cut off:", short["choices"][0]["finish_reason"], "|", short["choices"][0]["message"]["content"])

    waits = []
    flaky = Flaky(FakeLLM(), fail_first=2)
    out = retry_with_backoff(lambda: flaky.chat(msgs), attempts=4, base_delay=1.0, sleep=waits.append)
    print("attempts:", flaky.attempts, "| waits:", waits, "| ok:", out["choices"][0]["finish_reason"])

    print("stream:", list(llm.stream(msgs)))
