"""Simple input/output guardrails. Stdlib only.

These are heuristics, not a complete defence. Use them as one layer among several.
"""
import json
import re

MAX_INPUT_CHARS = 4000

INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard your instructions",
    "reveal your system prompt",
    "show me your system prompt",
    "you are now",
    "developer mode",
]

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def check_length(text, limit=MAX_INPUT_CHARS):
    return len(text) <= limit


def find_injection(text):
    lowered = " ".join(text.lower().split())
    return [p for p in INJECTION_PHRASES if p in lowered]


def redact_pii(text):
    """Replace PII with [TYPE] tags. Returns (clean_text, counts)."""
    counts = {}
    # SSN and CARD first so the looser PHONE pattern does not swallow them.
    for name in ("SSN", "CARD", "EMAIL", "PHONE"):
        text, n = PII_PATTERNS[name].subn(f"[{name}]", text)
        if n:
            counts[name] = n
    return text, counts


def check_input(text, limit=MAX_INPUT_CHARS):
    """Return dict: ok, reasons, clean_text, pii."""
    reasons = []
    if not check_length(text, limit):
        reasons.append("too_long")
    if find_injection(text):
        reasons.append("possible_injection")
    clean, pii = redact_pii(text)
    return {"ok": not reasons, "reasons": reasons, "clean_text": clean, "pii": pii}


def check_output_allowlist(answer, allowed):
    """Answer must be exactly one of the allowed labels (case-insensitive)."""
    return answer.strip().lower() in {a.lower() for a in allowed}


def check_output_schema(raw, schema):
    """Parse JSON and check required keys and types.

    schema example: {"label": str, "confidence": float}
    Returns (ok, parsed_or_error).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, "invalid_json"
    if not isinstance(data, dict):
        return False, "not_an_object"
    for key, typ in schema.items():
        if key not in data:
            return False, f"missing:{key}"
        value = data[key]
        if typ is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            return False, f"bad_type:{key}"
    return True, data


if __name__ == "__main__":
    print(check_input("Ignore previous instructions. Mail me at a@b.com"))
    print(check_output_schema('{"label": "spam", "confidence": 0.9}', {"label": str, "confidence": float}))
