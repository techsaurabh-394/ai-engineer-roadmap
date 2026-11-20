import ae_budget as m


def test_budget_order_and_none():
    b = m.Budget(max_steps=2, max_tokens=100, max_dollars=1)
    assert b.charge(10, 0.1) is None
    assert b.charge(10, 0.1) is None
    assert b.charge(10, 0.1) == "steps"


def test_tokens_and_dollars():
    assert m.Budget(max_tokens=50).charge(60, 0) == "tokens"
    assert m.Budget(max_dollars=0.5).charge(1, 0.6) == "dollars"


def test_repeat_loop_detected_with_arg_order_ignored():
    calls = [("s", {"a": 1, "b": 2}), ("s", {"b": 2, "a": 1}), ("s", {"a": 1, "b": 2})]
    assert m.detect_loop(calls)
    assert not m.detect_loop(calls[:2])


def test_pingpong():
    a, b = ("read", {"f": 1}), ("edit", {"f": 1})
    assert m.detect_pingpong([a, b] * 3)
    assert not m.detect_pingpong([a, b] * 2)
    assert not m.detect_pingpong([a, a, a, a, a, a])


def test_run_agent_reasons():
    same = [("search", {"q": "x"}, 100, 0.01)] * 30
    assert m.run_agent(same, m.Budget()) == ("loop:repeat", 3)
    varied = [("s", {"q": i}, 100, 0.01) for i in range(30)]
    assert m.run_agent(varied, m.Budget(max_steps=20))[0] == "budget:steps"
    assert m.run_agent([("a", {}, 1, 0.0)], m.Budget()) == ("finished", 1)
