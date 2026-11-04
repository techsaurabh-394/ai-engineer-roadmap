"""Get reliable structured data out of an LLM. Standard library only.

Free text is fragile. This module shows the safe pattern: extract the JSON,
validate it against a schema, and if it fails, send the errors back and try
again. The validator handles a small subset of JSON Schema, not the whole spec.
"""
import json


def extract_json(text):
    """Return the first JSON object found in text (prose or code fences), or None."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_ok(value, name):
    if name in ("integer", "number") and isinstance(value, bool):
        return False  # True is an int in Python, but not a JSON number
    return isinstance(value, _TYPES[name])


def validate(value, schema, path="$"):
    """Return a list of error strings. An empty list means the value is valid.

    Supports: type, required, properties, enum, minimum, maximum, items.
    """
    errors = []
    t = schema.get("type")
    if t and not _type_ok(value, t):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above maximum {schema['maximum']}")
    if t == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += validate(value[key], sub, f"{path}.{key}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def repair_loop(llm_fn, prompt, schema, max_tries=3):
    """Ask, validate, and feed errors back. Returns (data, tries, last_errors).

    data is None if every try failed. llm_fn takes a prompt string and returns text.
    """
    current = prompt
    errors = ["no JSON object found"]
    for tries in range(1, max_tries + 1):
        reply = llm_fn(current)
        data = extract_json(reply)
        errors = ["no JSON object found"] if data is None else validate(data, schema)
        if not errors:
            return data, tries, []
        current = (
            prompt
            + "\n\nYour last answer was invalid:\n"
            + "\n".join("- " + e for e in errors)
            + "\nReturn only a corrected JSON object."
        )
    return None, max_tries, errors


class ScriptedLLM:
    """Plays back fixed replies (the last one repeats) and records prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        i = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[i]


TICKET_SCHEMA = {
    "type": "object",
    "required": ["category", "priority", "tags"],
    "properties": {
        "category": {"type": "string", "enum": ["billing", "bug", "other"]},
        "priority": {"type": "integer", "minimum": 1, "maximum": 3},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


if __name__ == "__main__":
    prose = 'Sure! Here you go:\n```json\n{"category": "bug", "priority": 2, "tags": ["login"]}\n```\nHope it helps.'
    print("extracted:", extract_json(prose))

    bad = {"category": "refund", "priority": 7, "tags": ["ok", 5]}
    for e in validate(bad, TICKET_SCHEMA):
        print("error:", e)

    llm = ScriptedLLM([
        "I think it is a billing issue, priority high.",
        '{"category": "billing", "priority": 9, "tags": []}',
        '{"category": "billing", "priority": 2, "tags": ["invoice"]}',
    ])
    data, tries, errs = repair_loop(llm, "Classify: I was charged twice.", TICKET_SCHEMA, max_tries=3)
    print("result:", data, "| tries:", tries, "| errors:", errs)
    print("second prompt ended with:", llm.prompts[1].splitlines()[-1])
    print("third prompt had:", llm.prompts[2].splitlines()[-2])
