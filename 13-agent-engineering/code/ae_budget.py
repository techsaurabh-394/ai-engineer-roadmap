"""Agent run budgets and loop detection: stop a runaway agent for the right reason.

Run:  python ae_budget.py
"""
import json


class Budget:
    def __init__(self, max_steps=20, max_tokens=50_000, max_dollars=1.00):
        self.max_steps, self.max_tokens, self.max_dollars = max_steps, max_tokens, max_dollars
        self.steps = self.tokens = 0
        self.dollars = 0.0

    def charge(self, tokens, dollars):
        self.steps += 1
        self.tokens += tokens
        self.dollars += dollars
        return self.exceeded()

    def exceeded(self):
        """Return the first exceeded limit name, or None."""
        if self.steps > self.max_steps:
            return "steps"
        if self.tokens > self.max_tokens:
            return "tokens"
        if self.dollars > self.max_dollars:
            return "dollars"
        return None


def call_key(tool, args):
    return tool + ":" + json.dumps(args, sort_keys=True)


def detect_loop(calls, window=6, repeats=3):
    """True if any identical (tool, args) call appears `repeats` times in the last `window` calls."""
    recent = [call_key(t, a) for t, a in calls[-window:]]
    return any(recent.count(k) >= repeats for k in set(recent))


def detect_pingpong(calls, cycles=3):
    """True if the last 2*cycles calls alternate between exactly two distinct calls (A B A B A B)."""
    if len(calls) < 2 * cycles:
        return False
    keys = [call_key(t, a) for t, a in calls[-2 * cycles:]]
    a, b = keys[0], keys[1]
    return a != b and all(keys[i] == (a if i % 2 == 0 else b) for i in range(len(keys)))


def run_agent(script, budget, window=6, repeats=3):
    """Replay a scripted list of (tool, args, tokens, dollars). Returns (stop_reason, steps_done)."""
    calls = []
    for tool, args, tokens, dollars in script:
        hit = budget.charge(tokens, dollars)
        if hit:
            return f"budget:{hit}", len(calls)
        calls.append((tool, args))
        if detect_loop(calls, window, repeats):
            return "loop:repeat", len(calls)
        if detect_pingpong(calls):
            return "loop:pingpong", len(calls)
    return "finished", len(calls)


if __name__ == "__main__":
    same = [("search", {"q": "refund policy"}, 1500, 0.01)] * 30
    print("same call forever :", run_agent(same, Budget()))
    pp = [("read", {"f": "a"}, 1500, 0.01), ("edit", {"f": "a"}, 1500, 0.01)] * 15
    print("ping-pong         :", run_agent(pp, Budget()))
    varied = [("search", {"q": f"query {i}"}, 1500, 0.01) for i in range(30)]
    print("endless variety   :", run_agent(varied, Budget(max_steps=20)))
    pricey = [("big", {"i": i}, 20_000, 0.30) for i in range(10)]
    print("expensive steps   :", run_agent(pricey, Budget()))
    ok = [("search", {"q": "a"}, 1000, 0.01), ("read", {"f": "a"}, 1000, 0.01), ("answer", {}, 500, 0.01)]
    print("normal run        :", run_agent(ok, Budget()))
