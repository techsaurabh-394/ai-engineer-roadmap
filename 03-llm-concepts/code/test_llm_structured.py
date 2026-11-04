import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_structured import (  # noqa: E402
    TICKET_SCHEMA, ScriptedLLM, extract_json, repair_loop, validate,
)

GOOD = {"category": "bug", "priority": 2, "tags": ["login"]}


def test_extract_from_plain_json_prose_and_code_fence():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! {"a": 1} done.') == {"a": 1}
    assert extract_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}


def test_extract_skips_broken_braces_and_returns_none_when_absent():
    assert extract_json('use {curly} words then {"a": 2}') == {"a": 2}
    assert extract_json("no json here") is None
    assert extract_json('{"a": ') is None


def test_valid_value_has_no_errors():
    assert validate(GOOD, TICKET_SCHEMA) == []


def test_missing_required_field_is_reported():
    errs = validate({"category": "bug", "priority": 1}, TICKET_SCHEMA)
    assert errs == ["$: missing required field 'tags'"]


def test_enum_range_and_nested_array_errors_have_paths():
    errs = validate({"category": "refund", "priority": 7, "tags": ["ok", 5]}, TICKET_SCHEMA)
    assert "$.category: 'refund' not in ['billing', 'bug', 'other']" in errs
    assert "$.priority: 7 is above maximum 3" in errs
    assert "$.tags[1]: expected string, got int" in errs
    assert len(errs) == 3


def test_minimum_and_wrong_type():
    assert validate({"category": "bug", "priority": 0, "tags": []}, TICKET_SCHEMA) == [
        "$.priority: 0 is below minimum 1"
    ]
    assert validate([], TICKET_SCHEMA) == ["$: expected object, got list"]


def test_bool_is_not_an_integer():
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got bool"]
    assert validate(1.5, {"type": "number"}) == []


def test_repair_loop_succeeds_first_try_without_extra_calls():
    llm = ScriptedLLM(['{"category": "bug", "priority": 1, "tags": []}'])
    data, tries, errs = repair_loop(llm, "p", TICKET_SCHEMA)
    assert data is not None and tries == 1 and errs == [] and len(llm.prompts) == 1


def test_repair_loop_feeds_errors_back_and_then_succeeds():
    llm = ScriptedLLM([
        "not json at all",
        '{"category": "billing", "priority": 9, "tags": []}',
        '{"category": "billing", "priority": 2, "tags": ["x"]}',
    ])
    data, tries, errs = repair_loop(llm, "Classify", TICKET_SCHEMA, max_tries=3)
    assert tries == 3 and errs == [] and data == {"category": "billing", "priority": 2, "tags": ["x"]}
    assert "no JSON object found" in llm.prompts[1]
    assert "$.priority: 9 is above maximum 3" in llm.prompts[2]
    assert llm.prompts[2].startswith("Classify")


def test_repair_loop_gives_up_and_returns_last_errors():
    llm = ScriptedLLM(['{"category": "bug"}'])
    data, tries, errs = repair_loop(llm, "p", TICKET_SCHEMA, max_tries=2)
    assert data is None and tries == 2 and len(llm.prompts) == 2
    assert "$: missing required field 'priority'" in errs


def test_valid_does_not_mean_true():
    # Schema-valid but wrong: validation checks shape, not facts.
    wrong = {"category": "billing", "priority": 1, "tags": ["refund"]}
    assert validate(wrong, TICKET_SCHEMA) == []
