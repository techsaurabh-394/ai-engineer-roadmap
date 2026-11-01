import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_finetune_data import clean, split, to_jsonl  # noqa: E402


def row(p, c):
    return {"prompt": p, "completion": c}


def test_clean_drops_empty_and_duplicates_and_reports_where():
    rows = [row("a", "x"), row("a", "x"), row("b", ""), row("  ", "y"), row("c", "z")]
    kept, problems = clean(rows)
    assert kept == [row("a", "x"), row("c", "z")]
    assert problems == [(1, "duplicate"), (2, "empty field"), (3, "empty field")]


def test_same_prompt_with_different_answer_is_not_a_duplicate():
    kept, problems = clean([row("a", "x"), row("a", "y")])
    assert len(kept) == 2 and problems == []


def test_missing_keys_count_as_empty_instead_of_crashing():
    kept, problems = clean([{"prompt": "a"}, {"completion": "b"}])
    assert kept == [] and len(problems) == 2


def test_split_holds_out_rows_and_loses_none():
    rows = [row(str(i), "x") for i in range(10)]
    train, valid = split(rows, 0.2)
    assert len(valid) == 2 and len(train) == 8
    assert sorted(r["prompt"] for r in train + valid) == sorted(r["prompt"] for r in rows)
    assert not ({r["prompt"] for r in train} & {r["prompt"] for r in valid})


def test_split_is_repeatable_with_the_same_seed_and_differs_with_another():
    rows = [row(str(i), "x") for i in range(20)]
    assert split(rows, seed=1) == split(rows, seed=1)
    assert split(rows, seed=1) != split(rows, seed=2)


def test_split_of_one_row_keeps_it_for_training():
    train, valid = split([row("a", "x")])
    assert len(train) == 1 and valid == []


def test_split_does_not_change_the_input_list():
    rows = [row(str(i), "x") for i in range(5)]
    before = list(rows)
    split(rows)
    assert rows == before


def test_jsonl_is_one_json_object_per_line_and_keeps_unicode():
    out = to_jsonl([row("नमस्ते", "hi"), row("b", "c")])
    lines = out.split("\n")
    assert len(lines) == 2 and "नमस्ते" in lines[0]
