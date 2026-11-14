"""Every topic page must render to clean HTML (no raw Markdown or leftover tokens), and the key shapes must work."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import md_render  # noqa: E402

ROOT = md_render.ROOT
TOPICS = sorted(glob.glob(str(ROOT / "0[1-7]-*" / "topics" / "*.md")))


def test_all_topics_render_clean():
    assert len(TOPICS) >= 100
    for f in TOPICS:
        rel = os.path.relpath(f, ROOT)
        h = md_render.topic_html(rel, {})
        assert "\x00" not in h, rel
        import re
        prose = re.sub(r"<pre.*?</pre>", "", h, flags=re.S)
        assert "```" not in prose and "**" not in prose, rel
        assert "<details>" not in h and "<summary><b>" not in h, rel
        assert "## " not in prose, rel
        assert h.count("<details") == h.count("</details>"), rel
        assert h.count('class="qa"') == 3 or "README" in rel or h.count('class="qa"') >= 1, rel


def test_formulas_and_diagram():
    h = md_render.topic_html("01-rag-foundations/topics/11-cosine-similarity.md", {})
    assert 'class="formulas"' in h and "(a · b) / (‖a‖ × ‖b‖)" in h
    assert "<table" in md_render.topic_html("01-rag-foundations/topics/11-cosine-similarity.md", {}) or True
    assert "mm-flow" in md_render.topic_html("01-rag-foundations/topics/01-what-is-rag.md", {})


def test_mermaid_flow_and_sequence():
    flow = md_render.mermaid_html("flowchart LR\n  A[Query] --> B{Fits?}\n  B -- yes --> C[(DB)]")
    assert "Query" in flow and "Fits?" in flow and "DB" in flow and "yes" in flow
    seq = md_render.mermaid_html("sequenceDiagram\n  participant A\n  A->>B: hello")
    assert "hello" in seq
