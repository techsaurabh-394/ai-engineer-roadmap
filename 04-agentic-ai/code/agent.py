"""An AI agent in ~120 lines, no framework.

The loop: ask the model what to do -> run the tool it picks -> give it the result -> repeat.
Two LLM backends:
  ScriptedLLM    - fixed replies, so you can run and test the loop offline
  AnthropicLLM   - real Claude tool use (needs: pip install anthropic, ANTHROPIC_API_KEY)

Run offline demo:   python agent.py
Run with Claude:    ANTHROPIC_API_KEY=... python agent.py "What is 12*7+5? Then save it as a note."
"""
import json
import os
import sys

# ---------- tools: plain Python functions + a JSON schema the model reads ----------
NOTES: list[str] = []


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # safe: characters are whitelisted above
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def save_note(text: str) -> str:
    NOTES.append(text)
    return f"Saved note #{len(NOTES)}."


TOOLS = {"calculator": calculator, "save_note": save_note}
TOOL_SPECS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as '12*7+5'.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    {
        "name": "save_note",
        "description": "Save a short text note.",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
]


# ---------- the agent loop ----------
def run_agent(llm, user_goal: str, max_steps: int = 6, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": user_goal}]
    for step in range(1, max_steps + 1):
        reply = llm.step(messages, TOOL_SPECS)  # {"text": str, "tool_calls": [{"id","name","input"}]}
        messages.append({"role": "assistant", "content": reply["raw"]})
        if not reply["tool_calls"]:
            return reply["text"]  # model decided it is done
        results = []
        for call in reply["tool_calls"]:
            fn = TOOLS.get(call["name"])
            out = fn(**call["input"]) if fn else f"Error: unknown tool {call['name']}"
            if verbose:
                print(f"step {step}: {call['name']}({json.dumps(call['input'])}) -> {out}")
            results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out})
        messages.append({"role": "user", "content": results})
    return "Stopped: reached max steps without finishing."  # always cap the loop


# ---------- backends ----------
class ScriptedLLM:
    """Plays back a fixed script. Lets you test the loop with no API key."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def step(self, messages, tools):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        calls = [{"id": f"call_{self.i}_{n}", **c} for n, c in enumerate(item.get("tool_calls", []))]
        return {"text": item.get("text", ""), "tool_calls": calls, "raw": item.get("text", "")}


class AnthropicLLM:
    def __init__(self, model=None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def step(self, messages, tools):
        resp = self.client.messages.create(model=self.model, max_tokens=800, tools=tools, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "input": b.input} for b in resp.content if b.type == "tool_use"]
        return {"text": text, "tool_calls": calls, "raw": resp.content}


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:])
    if goal and os.getenv("ANTHROPIC_API_KEY"):
        print(run_agent(AnthropicLLM(), goal))
    else:
        print("(offline demo with a scripted model)")
        demo = ScriptedLLM([
            {"tool_calls": [{"name": "calculator", "input": {"expression": "12*7+5"}}]},
            {"tool_calls": [{"name": "save_note", "input": {"text": "12*7+5 = 89"}}]},
            {"text": "12*7+5 is 89, and I saved it as a note."},
        ])
        print(run_agent(demo, "What is 12*7+5? Save the answer as a note."))
