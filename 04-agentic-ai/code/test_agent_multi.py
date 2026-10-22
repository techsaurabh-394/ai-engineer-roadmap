import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent_multi as amu  # noqa: E402


def yes(tool, preview):
    return True


def no(tool, preview):
    return False


def test_planner_adds_email_step_only_when_asked():
    assert [s["tool"] for s in amu.planner("agents")] == ["write_text"]
    assert [s["tool"] for s in amu.planner("agents, email it")] == ["write_text", "send_email"]


def test_critic_reflection_revises_draft():
    res = amu.run_pipeline("agents", yes)
    assert res["status"] == "done"
    assert "Example:" in res["result"]
    assert any(line.startswith("critic:") for line in res["log"])


def test_human_approval_gate_blocks_risky_action():
    res = amu.run_pipeline("agents and email it", no)
    assert res["status"] == "rejected"
    assert "EMAIL SENT" not in res["result"]
    assert "human rejected send_email" in res["log"]


def test_human_approval_allows_and_critic_runs_before_send():
    seen = []

    def spy(tool, preview):
        seen.append(preview)
        return True

    res = amu.run_pipeline("agents and email it", spy)
    assert res["status"] == "done" and res["result"].startswith("EMAIL SENT")
    assert "Example:" in seen[0]  # the human saw the already-revised draft


def test_guardrail_allowlist():
    res = amu.run_pipeline("agents and email it", yes, allowed_tools=("write_text", "add_example"))
    assert res["status"] == "blocked"
    assert "not on the allowlist" in res["log"][-1]


def test_guardrail_max_steps():
    res = amu.run_pipeline("agents", yes, max_steps=1)
    assert res["status"] == "blocked" and "max_steps" in res["log"][-1]


def test_guardrails_unit():
    g = amu.Guardrails(["a"], max_steps=2, risky_tools=["a"])
    g.check("a")
    assert g.needs_approval("a")
    try:
        g.check("b")
    except amu.GuardrailError:
        pass
    else:
        raise AssertionError("expected allowlist block")
