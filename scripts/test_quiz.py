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


def test_built_site_embeds_every_question_with_an_animation(tmp_path, monkeypatch):
    import build_site
    monkeypatch.setattr(build_site, "OUT", tmp_path / "site")
    build_site.main()
    data = (tmp_path / "site" / "assets" / "data.js").read_text(encoding="utf-8")
    assert data.startswith("window.AIER_DATA=")
    d = json.loads(data[len("window.AIER_DATA="):].rstrip().rstrip(";"))
    assert len(d["quiz"]) == len(qc.build())
    names = {l["name"] for l in d["lessons"]}
    assert all(q["anims"] and set(q["anims"]) <= names for q in d["quiz"])
    page = (tmp_path / "site" / "quiz" / "index.html").read_text(encoding="utf-8")
    assert "__SCRIPTS__" not in page and "__REPO__" not in page and "__COUNT__" not in page
    assert 'rel="icon"' in page  # no favicon.ico 404 in the console
    assert (tmp_path / "site" / "assets" / "quiz.js").exists()


def test_quiz_links_cover_every_question_and_nothing_else():
    import build_quiz
    links = build_quiz.load_links()
    questions = [q["q"] for q in qc.build()]
    assert set(links) == set(questions), "update scripts/quiz_links.json: " + repr(set(links) ^ set(questions))
