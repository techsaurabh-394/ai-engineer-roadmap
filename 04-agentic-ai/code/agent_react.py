"""ReAct (Reason + Act) loop with JSON tool schemas and function-calling dispatch. Standard library only.

The "LLM" here is scripted: it returns pre-written text so the loop runs offline and in tests.
A real model would generate the same kind of text (or structured tool calls) itself.

Text format the model writes each turn (from the ReAct paper's Thought / Action / Observation style):
    Thought: I need the population of France.
    Action: lookup
    Action Input: {"topic": "france"}
or, when done:
    Thought: I have what I need.
    Final Answer: ...

Run:  python agent_react.py
"""
import json
import re

# ---------- tools + JSON schemas (same shape as Anthropic "input_schema" / OpenAI "parameters") ----------
FACTS = {
    "france": "France has a population of about 68 million. Capital: Paris.",
    "japan": "Japan has a population of about 124 million. Capital: Tokyo.",
}


def lookup(topic: str) -> str:
    return FACTS.get(topic.lower(), f"No entry for '{topic}'.")


def multiply(a: float, b: float) -> str:
    result = a * b
    return str(int(result)) if float(result).is_integer() else str(result)


TOOLS = {
    "lookup": {
        "fn": lookup,
        "description": "Look up a short fact sheet about a country.",
        "schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
    "multiply": {
        "fn": multiply,
        "description": "Multiply two numbers.",
        "schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}

_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


def tool_specs():
    """What you would send to the model: name, description, JSON schema. No Python functions."""
    return [{"name": n, "description": t["description"], "input_schema": t["schema"]} for n, t in TOOLS.items()]


def validate_args(schema, args):
    """Tiny subset of JSON Schema validation: object, required, property types, no extras."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    props = schema.get("properties", {})
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties", True) is False:
                return f"unexpected argument '{key}'"
            continue
        expected = _JSON_TYPES[props[key]["type"]]
        if isinstance(value, bool) and props[key]["type"] in ("number", "integer"):
            return f"argument '{key}' must be {props[key]['type']}"
        if not isinstance(value, expected):
            return f"argument '{key}' must be {props[key]['type']}"
    return None


def dispatch(name, args):
    """Function-calling dispatch: validate, run, and always return a string (errors included)."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    problem = validate_args(tool["schema"], args)
    if problem:
        return f"Error: {problem}"
    try:
        return tool["fn"](**args)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# ---------- parsing the model's text ----------
_ACTION = re.compile(r"Action:\s*(\S+)\s*\nAction Input:\s*(\{.*\})", re.S)
_FINAL = re.compile(r"Final Answer:\s*(.*)", re.S)


def parse(text):
    """Return ("final", answer) or ("action", name, args) or ("error", message)."""
    final = _FINAL.search(text)
    if final:
        return ("final", final.group(1).strip())
    m = _ACTION.search(text)
    if not m:
        return ("error", "Could not find 'Action' or 'Final Answer'. Use the required format.")
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError as e:
        return ("error", f"Action Input is not valid JSON: {e.msg}")
    return ("action", m.group(1), args)


# ---------- the ReAct loop ----------
class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.i, self.prompts = replies, 0, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def run_react(llm, question, max_steps=5):
    """Returns (answer, trace). trace is a list of dicts so you can log or test each step."""
    transcript = f"Question: {question}\n"
    trace = []
    for _ in range(max_steps):
        text = llm(transcript)
        transcript += text.strip() + "\n"
        parsed = parse(text)
        if parsed[0] == "final":
            trace.append({"type": "final", "answer": parsed[1]})
            return parsed[1], trace
        if parsed[0] == "error":
            observation = f"Error: {parsed[1]}"
            trace.append({"type": "parse_error", "observation": observation})
        else:
            _, name, args = parsed
            observation = dispatch(name, args)
            trace.append({"type": "action", "tool": name, "args": args, "observation": observation})
        transcript += f"Observation: {observation}\n"
    return "Stopped: reached max steps without a final answer.", trace


DEMO_SCRIPT = [
    'Thought: I need the population of France.\nAction: lookup\nAction Input: {"topic": "france"}',
    'Thought: About 68 million. The question asks for double that.\nAction: multiply\nAction Input: {"a": 68, "b": 2}',
    "Thought: I have the result.\nFinal Answer: Double France's population is about 136 million.",
]

if __name__ == "__main__":
    print("tool specs sent to the model:")
    print(json.dumps(tool_specs(), indent=2))
    answer, trace = run_react(ScriptedLLM(DEMO_SCRIPT), "What is double the population of France?")
    for step in trace:
        print(step)
    print("\nanswer:", answer)
