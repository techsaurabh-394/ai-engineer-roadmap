"""The agent loop from agent.py, but every step is recorded as an event (standard library only).

Why a second file: agent.py prints as it goes. The site's "Agent loop lab" needs the same loop as DATA, so it can
step forward and backward through it, show what is in the conversation at each point, and ask a human before a
risky tool runs. A JavaScript port of this file runs in the browser; tests compare the two.

The model is SCRIPTED (it replays fixed replies), exactly like ScriptedLLM in agent.py. A real model decides
for itself; the loop around it is the same.

Run:  python agent_trace.py
"""
import json

# ------------------------------------------------------------------ tools
FACTS = {
    "france": "France has a population of about 68 million. Capital: Paris.",
    "japan": "Japan has a population of about 124 million. Capital: Tokyo.",
}

TOOL_SPECS = [
    {"name": "calculator", "description": "Evaluate an arithmetic expression such as '12*7+5'.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
     "risk": "safe"},
    {"name": "lookup", "description": "Look up a short fact sheet about a country.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
     "risk": "safe"},
    {"name": "save_note", "description": "Save a short text note (changes state, so a person may want to approve it).",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
     "risk": "writes"},
]


class _Parser:
    """Recursive-descent arithmetic: numbers, + - * /, unary minus and parentheses. No eval()."""

    def __init__(self, text):
        self.s, self.i = text, 0

    def peek(self):
        while self.i < len(self.s) and self.s[self.i] == " ":
            self.i += 1
        return self.s[self.i] if self.i < len(self.s) else ""

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.s[self.i]
            self.i += 1
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.s[self.i]
            self.i += 1
            r = self.factor()
            if op == "/" and r == 0:
                raise ZeroDivisionError("division by zero")
            v = v * r if op == "*" else v / r
        return v

    def factor(self):
        c = self.peek()
        if c == "-":
            self.i += 1
            return -self.factor()
        if c == "(":
            self.i += 1
            v = self.expr()
            if self.peek() != ")":
                raise ValueError("missing )")
            self.i += 1
            return v
        j = self.i
        while j < len(self.s) and (self.s[j].isdigit() or self.s[j] == "."):
            j += 1
        if j == self.i:
            raise ValueError("expected a number")
        token = self.s[self.i:j]
        if token.count(".") > 1 or token == ".":
            raise ValueError("bad number")
        v = float(token)
        self.i = j
        return v


def _fmt(v):
    if v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def calculator(expression):
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    p = _Parser(expression)
    try:
        v = p.expr()
        if p.peek() != "":
            raise ValueError("unexpected character")
        return _fmt(v)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def lookup(topic):
    return FACTS.get(topic.lower(), f"No entry for '{topic}'.")


# ------------------------------------------------------------------ scripted scenarios
SCENARIOS = {
    "math_note": {
        "title": "Calculate, then save a note",
        "goal": "What is 12*7+5? Save the answer as a note.",
        "approve": ["save_note"],
        "script": [
            {"text": "I will use the calculator.", "tool_calls": [{"name": "calculator", "input": {"expression": "12*7+5"}}]},
            {"text": "Now I will save the result.", "tool_calls": [{"name": "save_note", "input": {"text": "12*7+5 = 89"}}]},
            {"text": "12*7+5 is 89, and I saved it as a note."},
        ],
    },
    "country_math": {
        "title": "Look something up, then calculate",
        "goal": "France has about 68 million people. How many people are in three Frances?",
        "approve": [],
        "script": [
            {"text": "I need the population first.", "tool_calls": [{"name": "lookup", "input": {"topic": "france"}}]},
            {"text": "The population is 68 million. Three times that:", "tool_calls": [{"name": "calculator", "input": {"expression": "68*3"}}]},
            {"text": "Three Frances would be about 204 million people."},
        ],
    },
    "bad_tool": {
        "title": "The model calls a tool that does not exist",
        "goal": "Email me the result of 6*7.",
        "approve": [],
        "script": [
            {"text": "I will send an email.", "tool_calls": [{"name": "send_email", "input": {"to": "me", "body": "6*7"}}]},
            {"text": "There is no email tool. I can still compute the number.", "tool_calls": [{"name": "calculator", "input": {"expression": "6*7"}}]},
            {"text": "6*7 is 42. I cannot send email with the tools I have."},
        ],
    },
    "tool_error": {
        "title": "A tool fails and the model recovers",
        "goal": "Split 10 into 2 equal parts.",
        "approve": [],
        "script": [
            {"text": "Divide 10 by 0 parts?", "tool_calls": [{"name": "calculator", "input": {"expression": "10/0"}}]},
            {"text": "That was a mistake. Two parts means divide by 2.", "tool_calls": [{"name": "calculator", "input": {"expression": "10/2"}}]},
            {"text": "Each part is 5."},
        ],
    },
    "loop_forever": {
        "title": "A confused model that never finishes",
        "goal": "Keep checking until you are sure.",
        "approve": [],
        "script": [
            {"text": "Let me check again.", "tool_calls": [{"name": "calculator", "input": {"expression": "1+1"}}]},
        ],
    },
}


# ------------------------------------------------------------------ the loop, as data
def run_trace(script, goal, max_steps=6, approve=(), decisions=None, specs=None):
    """Run the agent loop with a scripted model and return every event.

    approve   : tool names that need a person's OK before they run
    decisions : {tool name: True/False}, the person's answers (default: approve)
    """
    decisions = decisions or {}
    known = {t["name"]: t for t in (specs or TOOL_SPECS)}
    notes = []
    events = [{"type": "user", "text": goal}]
    messages = 1
    i = 0
    for step in range(1, max_steps + 1):
        item = script[min(i, len(script) - 1)]
        i += 1
        calls = [{"id": f"call_{i}_{n}", "name": c["name"], "input": c["input"]} for n, c in enumerate(item.get("tool_calls", []))]
        events.append({"type": "model", "step": step, "text": item.get("text", ""), "calls": calls})
        messages += 1
        if not calls:
            events.append({"type": "final", "step": step, "text": item.get("text", "")})
            return {"events": events, "final": item.get("text", ""), "steps_used": step, "notes": notes, "messages": messages,
                    "stopped": False}
        for c in calls:
            spec = known.get(c["name"])
            if spec is None:
                events.append({"type": "tool", "step": step, "id": c["id"], "name": c["name"], "input": c["input"],
                               "output": f"Error: unknown tool {c['name']}", "status": "error"})
                continue
            if c["name"] in approve:
                ok = bool(decisions.get(c["name"], True))
                events.append({"type": "approval", "step": step, "id": c["id"], "name": c["name"], "input": c["input"], "approved": ok})
                if not ok:
                    events.append({"type": "tool", "step": step, "id": c["id"], "name": c["name"], "input": c["input"],
                                   "output": "Error: a person denied this action.", "status": "denied"})
                    continue
            missing = [k for k in spec["input_schema"]["required"] if k not in c["input"]]
            if missing:
                out, status = f"Error: missing argument '{missing[0]}'", "error"
            elif c["name"] == "calculator":
                out = calculator(c["input"]["expression"])
                status = "error" if out.startswith("Error") else "ok"
            elif c["name"] == "lookup":
                out, status = lookup(c["input"]["topic"]), "ok"
            else:
                notes.append(c["input"]["text"])
                out, status = f"Saved note #{len(notes)}.", "ok"
            events.append({"type": "tool", "step": step, "id": c["id"], "name": c["name"], "input": c["input"], "output": out, "status": status})
        messages += 1
    events.append({"type": "stopped", "step": max_steps, "text": "Stopped: reached max steps without finishing."})
    return {"events": events, "final": None, "steps_used": max_steps, "notes": notes, "messages": messages, "stopped": True}


if __name__ == "__main__":
    for key, sc in SCENARIOS.items():
        r = run_trace(sc["script"], sc["goal"], 6, sc["approve"])
        print(f"{key}: {r['steps_used']} steps, {'STOPPED' if r['stopped'] else 'finished'}, {r['messages']} messages")
        for e in r["events"]:
            if e["type"] == "tool":
                print(f"   step {e['step']}: {e['name']}({json.dumps(e['input'])}) -> {e['output']}")
