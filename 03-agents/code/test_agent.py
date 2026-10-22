import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402


def setup_function():
    agent.NOTES.clear()


def test_calculator_rejects_code():
    assert agent.calculator("__import__('os')").startswith("Error")
    assert agent.calculator("12*7+5") == "89"


def test_agent_runs_tools_then_answers():
    llm = agent.ScriptedLLM([
        {"tool_calls": [{"name": "calculator", "input": {"expression": "2+2"}}]},
        {"tool_calls": [{"name": "save_note", "input": {"text": "4"}}]},
        {"text": "done"},
    ])
    assert agent.run_agent(llm, "goal", verbose=False) == "done"
    assert agent.NOTES == ["4"]


def test_agent_stops_at_max_steps():
    llm = agent.ScriptedLLM([{"tool_calls": [{"name": "calculator", "input": {"expression": "1+1"}}]}])
    assert agent.run_agent(llm, "loop forever", max_steps=3, verbose=False).startswith("Stopped")


def test_unknown_tool_is_reported_not_crashed():
    llm = agent.ScriptedLLM([{"tool_calls": [{"name": "nope", "input": {}}]}, {"text": "ok"}])
    assert agent.run_agent(llm, "x", verbose=False) == "ok"
