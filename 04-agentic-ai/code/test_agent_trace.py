import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from agent_trace import SCENARIOS, calculator, run_trace  # noqa: E402


def test_calculator_basics_and_precedence():
    assert calculator("12*7+5") == "89"
    assert calculator("2+3*4") == "14"
    assert calculator("(2+3)*4") == "20"
    assert calculator("7/2") == "3.5"
    assert calculator("-3+5") == "2"


def test_calculator_rejects_code_and_bad_input():
    assert calculator("__import__('os')").startswith("Error: only numbers")
    assert calculator("").startswith("Error")
    assert calculator("10/0") == "Error: division by zero"
    assert calculator("2+").startswith("Error")
    assert calculator("(2+3").startswith("Error")


def test_happy_path_finishes_and_saves_note():
    sc = SCENARIOS["math_note"]
    r = run_trace(sc["script"], sc["goal"], 6, sc["approve"])
    assert not r["stopped"] and r["steps_used"] == 3 and r["notes"] == ["12*7+5 = 89"]
    assert r["final"].startswith("12*7+5 is 89")
    kinds = [e["type"] for e in r["events"]]
    assert kinds == ["user", "model", "tool", "model", "approval", "tool", "model", "final"]


def test_denied_approval_blocks_the_tool():
    sc = SCENARIOS["math_note"]
    r = run_trace(sc["script"], sc["goal"], 6, sc["approve"], {"save_note": False})
    assert r["notes"] == []
    denied = [e for e in r["events"] if e["type"] == "tool" and e["status"] == "denied"]
    assert len(denied) == 1 and "denied" in denied[0]["output"]


def test_step_cap_stops_a_loop():
    sc = SCENARIOS["loop_forever"]
    r = run_trace(sc["script"], sc["goal"], 4)
    assert r["stopped"] and r["steps_used"] == 4 and r["final"] is None
    assert sum(1 for e in r["events"] if e["type"] == "tool") == 4
    assert r["events"][-1]["type"] == "stopped"


def test_unknown_tool_and_tool_error_are_results_not_crashes():
    r = run_trace(SCENARIOS["bad_tool"]["script"], "x", 6)
    assert r["events"][2]["status"] == "error" and "unknown tool send_email" in r["events"][2]["output"]
    assert not r["stopped"]
    r = run_trace(SCENARIOS["tool_error"]["script"], "x", 6)
    outs = [e["output"] for e in r["events"] if e["type"] == "tool"]
    assert outs == ["Error: division by zero", "5"]


def test_missing_argument_is_reported():
    r = run_trace([{"tool_calls": [{"name": "calculator", "input": {}}]}, {"text": "done"}], "g", 3)
    assert r["events"][2]["output"] == "Error: missing argument 'expression'"


def test_message_count_grows_two_per_tool_round():
    r = run_trace(SCENARIOS["country_math"]["script"], "g", 6)
    assert r["messages"] == 1 + 2 + 2 + 1


def test_malformed_numbers_and_overflow_do_not_crash():
    assert calculator("1.2.3") == "Error: bad number"
    assert calculator(".") == "Error: bad number"
    assert calculator("9" * 200 + "*" + "9" * 200) == "inf"
