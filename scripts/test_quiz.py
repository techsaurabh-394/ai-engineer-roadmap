"""Tests for the quiz extractor and page builder (scripts/quiz_core.py, scripts/build_quiz.py)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import quiz_core as qc  # noqa: E402


def test_extracts_every_details_block_in_phase_readmes():
    for phase in qc.PHASES:
        text = (qc.ROOT / phase / "README.md").read_text(encoding="utf-8")
        assert len(qc.extract(qc.ROOT / phase / "README.md", phase)) == text.count("<details>"), phase


def test_total_matches_readmes():
    n = sum((qc.ROOT / p / "README.md").read_text(encoding="utf-8").count("<details>") for p in qc.PHASES)
    assert len(qc.build()) == n and n >= 75


def test_all_three_readme_shapes_parse():
    one = '**1. What is X?**\n<details><summary>Answer</summary>\nIt is `y`.\n</details>\n'
    two = '<details><summary><b>2. What is Z?</b></summary>\n\nBecause *w*.\n</details>\n'
    three = '3. **Why?**\n   <details><summary>Answer</summary>Since.</details>\n'
    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "README.md"
        p.write_text(one + two + three, encoding="utf-8")
        items = qc.extract(p, "01-rag-foundations")
    assert [i["q"] for i in items] == ["What is X?", "What is Z?", "Why?"]
    assert "<code>y</code>" in items[0]["a"] and "<em>w</em>" in items[1]["a"] and items[2]["a"] == "Since."


def test_no_empty_or_numbered_questions_and_html_is_safe():
    for it in qc.build():
        assert it["q"] and not re.match(r"^\d+\.", it["q"]) and "<" not in it["q"] and "**" not in it["q"]
        assert it["a"].strip()
        # only the tags we emit are allowed in answers
        assert set(re.findall(r"</?([a-z]+)", it["a"])) <= {"a", "code", "em", "br"}, it["q"]
        assert "**" not in it["a"] and "<details" not in it["a"] and "</details" not in it["a"]


def test_answer_links_are_absolute_and_point_at_real_topic_files():
    seen = 0
    for it in qc.build():
        for href in re.findall(r'href="([^"]+)"', it["a"]):
            assert href.startswith(qc.REPO + "/blob/main/"), href
            rel = href[len(qc.REPO + "/blob/main/"):]
            assert (qc.ROOT / rel).exists(), href
            seen += 1
    in_readmes = sum((qc.ROOT / p / "README.md").read_text(encoding="utf-8").count('href="topics/') for p in qc.PHASES)
    assert seen == in_readmes > 0, (seen, in_readmes)


def test_untrusted_text_is_escaped():
    out = qc._answer_html("<script>alert(1)</script> & `a<b`", "01-rag-foundations")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<code>a&lt;b</code>" in out


def test_built_page_embeds_valid_json_and_counts(tmp_path, monkeypatch):
    import build_quiz
    monkeypatch.setattr(build_quiz, "OUT", tmp_path / "quiz")
    build_quiz.main()
    page = (tmp_path / "quiz" / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const Q=(\[.*?\]),PH=", page, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</")) if m else []
    assert m is not None
    assert len(data) == len(qc.build())
    assert "__QUESTIONS__" not in page and "__REPO__" not in page and "__COUNT__" not in page
