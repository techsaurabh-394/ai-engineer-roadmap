#!/usr/bin/env python3
"""Extract the checkpoint-quiz questions from the phase READMEs (standard library only).

The phase READMEs hold the quiz as <details> blocks, written in a few slightly different shapes:
  **1. Question**            <details><summary><b>1. Question</b></summary>     1. **Question**
  <details><summary>Answer   answer text ...                                    <details><summary>Answer</summary>...
The extractor accepts all of them, so the READMEs stay the single source of truth.
"""
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/techsaurabh-394/ai-engineer-roadmap"
PHASES = {
    "01-rag-foundations": "Phase 1: RAG Foundations",
    "02-advanced-rag": "Phase 2: Advanced RAG",
    "03-llm-concepts": "Phase 3: LLM Concepts",
    "04-agentic-ai": "Phase 4: Agentic AI",
    "05-frameworks": "Phase 5: Frameworks",
    "06-mcp": "Phase 6: MCP",
    "07-production-ai": "Phase 7: Production AI",
}

DETAILS = re.compile(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", re.S)
LINK = re.compile(r'<a href="([^"]+)">(.*?)</a>', re.S)
CODE = re.compile(r"<code>(.*?)</code>|`([^`]+)`", re.S)


def _plain_question(text: str) -> str:
    text = re.sub(r"</?(?:b|i|em|strong|code)>", "", text)
    text = html.unescape(text.replace("**", "").replace("`", "")).strip()
    return re.sub(r"^\d+\.\s*", "", text).strip()


def _answer_html(raw: str, phase: str) -> str:
    """Small, safe subset: escaped text, <code>, links to the topic pages, *emphasis*."""
    raw = raw.strip()
    tokens = []

    def stash(markup: str) -> str:
        tokens.append(markup)
        return f"\x00{len(tokens) - 1}\x00"

    def link(m):
        href, label = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if href.startswith("topics/"):
            href = f"{REPO}/blob/main/{phase}/{href}"
        elif not href.startswith("http"):
            href = f"{REPO}/blob/main/{phase}/{href}"
        return stash(f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>')

    raw = LINK.sub(link, raw)
    raw = CODE.sub(lambda m: stash(f"<code>{html.escape(m.group(1) or m.group(2))}</code>"), raw)
    raw = html.escape(re.sub(r"</?(?:b|i|em|strong)>", "", raw).replace("**", ""), quote=False)
    raw = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", raw)
    raw = re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], raw)
    return re.sub(r"\n{2,}", "<br><br>", raw).replace("\n", " ")


def extract(readme: pathlib.Path, phase: str) -> list[dict]:
    text = readme.read_text(encoding="utf-8")
    out = []
    for m in DETAILS.finditer(text):
        summary, body = m.group(1).strip(), m.group(2)
        if re.sub(r"<[^>]+>", "", summary).strip().lower() == "answer":
            before = [ln for ln in text[: m.start()].splitlines() if ln.strip()]
            if not before:
                continue
            question = _plain_question(before[-1])
        else:
            question = _plain_question(summary)
        if not question or not body.strip():
            continue
        out.append({"phase": phase, "q": question, "a": _answer_html(body, phase)})
    return out


def build() -> list[dict]:
    items = []
    for phase in PHASES:
        readme = ROOT / phase / "README.md"
        if readme.exists():
            items.extend(extract(readme, phase))
    return items


if __name__ == "__main__":
    items = build()
    per = {p: sum(1 for i in items if i["phase"] == p) for p in PHASES}
    print(f"{len(items)} questions", per)
