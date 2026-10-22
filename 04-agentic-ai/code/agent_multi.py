"""Multi-agent pipeline: planner -> worker -> critic (reflection), with a human approval gate
and guardrails (tool allowlist + max steps). Standard library only, fully deterministic.

  planner   splits a goal into steps
  worker    executes each step with an allowed tool
  critic    checks the draft; if it finds problems, the worker revises (reflection loop)
  approver  a human (or a test stub) must approve risky actions before they run
  Guardrails blocks tools not on the allowlist and stops runaway loops

Run:  python agent_multi.py
"""


class GuardrailError(Exception):
    pass


class Guardrails:
    def __init__(self, allowed_tools, max_steps=10, risky_tools=()):
        self.allowed = set(allowed_tools)
        self.risky = set(risky_tools)
        self.max_steps = max_steps
        self.steps = 0

    def check(self, tool):
        self.steps += 1
        if self.steps > self.max_steps:
            raise GuardrailError(f"max_steps={self.max_steps} exceeded")
        if tool not in self.allowed:
            raise GuardrailError(f"tool '{tool}' is not on the allowlist")

    def needs_approval(self, tool):
        return tool in self.risky


# ---------- tools ----------
def write_text(topic):
    return f"Draft about {topic}."


def add_example(text):
    return text + " Example: a support bot that looks up orders."


def send_email(text):
    return f"EMAIL SENT: {text}"


TOOLS = {"write_text": write_text, "add_example": add_example, "send_email": send_email}


# ---------- agents ----------
def planner(goal):
    """A real planner would be an LLM call returning JSON. This one is rule-based."""
    plan = [{"tool": "write_text", "arg": goal}]
    if "email" in goal.lower():
        plan.append({"tool": "send_email", "arg": None})  # arg None = use the current draft
    return plan


def worker(step, draft, guard, approver, log):
    tool = step["tool"]
    guard.check(tool)
    if guard.needs_approval(tool):
        preview = draft if step["arg"] is None else step["arg"]
        if not approver(tool, preview):
            log.append(f"human rejected {tool}")
            return draft, False
        log.append(f"human approved {tool}")
    arg = draft if step["arg"] is None else step["arg"]
    out = TOOLS[tool](arg)
    log.append(f"{tool} -> {out}")
    return out, True


def critic(draft):
    """Return a list of problems. Empty list means the draft passes."""
    problems = []
    if "Example:" not in draft:
        problems.append("needs a concrete example")
    if len(draft) > 300:
        problems.append("too long")
    return problems


def run_pipeline(goal, approver, allowed_tools=("write_text", "add_example", "send_email"),
                 risky_tools=("send_email",), max_steps=10, max_revisions=2):
    guard = Guardrails(allowed_tools, max_steps, risky_tools)
    log, draft = [], ""
    try:
        plan = planner(goal)
        log.append(f"plan: {[s['tool'] for s in plan]}")
        for step in plan:
            if step["tool"] in risky_tools:
                # reflection happens before anything irreversible goes out
                for _ in range(max_revisions):
                    problems = critic(draft)
                    if not problems:
                        break
                    log.append(f"critic: {problems}")
                    draft, _ = worker({"tool": "add_example", "arg": None}, draft, guard, approver, log)
            draft, ok = worker(step, draft, guard, approver, log)
            if not ok:
                return {"status": "rejected", "result": draft, "log": log}
        problems = critic(draft) if "EMAIL SENT" not in draft else []
        for _ in range(max_revisions):
            if not problems:
                break
            log.append(f"critic: {problems}")
            draft, _ = worker({"tool": "add_example", "arg": None}, draft, guard, approver, log)
            problems = critic(draft)
        return {"status": "done" if not problems else "needs_work", "result": draft, "log": log}
    except GuardrailError as e:
        log.append(f"guardrail: {e}")
        return {"status": "blocked", "result": draft, "log": log}


def console_approver(tool, preview):
    return input(f"Allow {tool}? Preview: {preview!r} [y/N] ").strip().lower() == "y"


if __name__ == "__main__":
    def auto_yes(tool, preview):
        print(f"[approval requested] {tool}: {preview!r} -> approved (auto, demo)")
        return True

    for goal in ("agents", "agents and email it"):
        res = run_pipeline(goal, auto_yes)
        print(f"\ngoal={goal!r} status={res['status']}")
        for line in res["log"]:
            print("  ", line)
