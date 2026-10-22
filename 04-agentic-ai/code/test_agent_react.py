import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent_react as ar  # noqa: E402


def test_tool_specs_are_plain_json_schema():
    specs = ar.tool_specs()
    names = {s["name"] for s in specs}
    assert names == {"lookup", "multiply"}
    for s in specs:
        assert s["input_schema"]["type"] == "object" and "fn" not in s


def test_validate_args():
    schema = ar.TOOLS["multiply"]["schema"]
    assert ar.validate_args(schema, {"a": 2, "b": 3.5}) is None
    assert "missing" in ar.validate_args(schema, {"a": 2})
    assert "must be number" in ar.validate_args(schema, {"a": "2", "b": 3})
    assert "unexpected" in ar.validate_args(schema, {"a": 1, "b": 2, "c": 3})
    assert "must be number" in ar.validate_args(schema, {"a": True, "b": 2})


def test_dispatch_runs_and_reports_errors_as_text():
    assert ar.dispatch("multiply", {"a": 6, "b": 7}) == "42"
    assert ar.dispatch("nope", {}).startswith("Error: unknown tool")
    assert ar.dispatch("lookup", {}).startswith("Error: missing")


def test_parse_variants():
    assert ar.parse("Thought: x\nFinal Answer: 42") == ("final", "42")
    assert ar.parse('Action: lookup\nAction Input: {"topic": "japan"}') == ("action", "lookup", {"topic": "japan"})
    assert ar.parse("Action: lookup\nAction Input: {bad json}")[0] == "error"
    assert ar.parse("I am just chatting")[0] == "error"


def test_react_loop_end_to_end():
    llm = ar.ScriptedLLM(ar.DEMO_SCRIPT)
    answer, trace = ar.run_react(llm, "double France?")
    assert "136" in answer
    assert [t["type"] for t in trace] == ["action", "action", "final"]
    assert trace[1]["observation"] == "136"
    # the observation from step 1 is fed back into the prompt for step 2
    assert "Observation: France has a population" in llm.prompts[1]


def test_react_recovers_from_bad_format():
    llm = ar.ScriptedLLM(["no format here", "Final Answer: ok"])
    answer, trace = ar.run_react(llm, "q")
    assert answer == "ok" and trace[0]["type"] == "parse_error"
    assert "Observation: Error" in llm.prompts[1]


def test_react_stops_at_max_steps():
    looping = ['Action: lookup\nAction Input: {"topic": "japan"}']
    answer, trace = ar.run_react(ar.ScriptedLLM(looping), "q", max_steps=3)
    assert answer.startswith("Stopped") and len(trace) == 3
