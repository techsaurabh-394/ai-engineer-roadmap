#!/usr/bin/env python3
"""The quiz page of the site (standard library only).

    python scripts/build_quiz.py      # builds the whole site into _site/, quiz included

Questions come from the phase READMEs (quiz_core.py). Answers are free text, so the quiz is
self-assessment: think, reveal, then mark "I knew it" or "Not yet". Progress and the spaced-repetition
schedule stay in the browser (localStorage); nothing is sent anywhere.

scripts/quiz_links.json says which animation explains each question. It is keyed by the question
text, so renaming a question in a README without updating it fails the tests.
"""
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quiz_core as qc  # noqa: E402

LINKS = pathlib.Path(__file__).resolve().parent / "quiz_links.json"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Engineer Roadmap: quiz</title>
<meta name="description" content="__COUNT__ self-check questions across 7 phases: RAG, LLMs, agents, MCP and production AI. Reveal the answer, mark yourself, see your weak topics. Free and open source.">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="__FAVICON__">
__THEME_BOOT__
<link rel="stylesheet" href="../assets/theme.css?v=__V__">
<link rel="stylesheet" href="../assets/quiz.css?v=__V__">
</head><body>
<a class="skip" href="#view">Skip to the quiz</a>
<header class="topbar">
<a class="brand" href="../"><span class="logo" aria-hidden="true">AI</span><span><span class="name-long">AI Engineer </span>Roadmap</span></a>
<nav class="topnav" aria-label="Sections"><a href="../">Lessons</a><a href="../#roadmap">Roadmap</a><a href="./" aria-current="page">Quiz</a><a href="__REPO__/blob/main/README.md#start-in-5-minutes">Start here</a></nav>
<span class="grow"></span>
<a class="icon-btn only-narrow" href="../">Lessons</a>
<button id="theme" class="icon-btn" type="button" aria-label="Change theme"></button>
<a class="icon-btn" href="__REPO__" rel="noopener"><span class="name-long">GitHub</span><span class="sr-only">repository</span></a>
</header>
<main class="quiz-main" id="main">
<div class="quiz-head"><h1>Check what stuck</h1>
<p>__COUNT__ questions from the seven phases. Say your answer, reveal the model answer, and mark yourself. Cards you miss come back sooner, and every card links to the animation that explains it.</p></div>
<nav class="tabs" id="tabs" aria-label="Quiz views"></nav>
<div class="filters" id="filters" role="group" aria-label="Filter by phase"></div>
<div id="view" tabindex="-1"><noscript><p>The quiz needs JavaScript. The same questions and answers are in each phase README in the <a href="__REPO__">GitHub repo</a>.</p></noscript></div>
<footer class="site"><span>Self-assessment only: answers are free text, so only you can judge them. Progress stays in this browser.</span>
<span>A wrong or unclear answer is a bug: <a href="__REPO__/issues" rel="noopener">open an issue</a>.</span></footer>
</main>
__SCRIPTS__
</body></html>
"""


def question_id(question: str) -> str:
    """A short stable id for a question, so saved progress survives questions being added or reordered."""
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]


def load_links() -> dict:
    return json.loads(LINKS.read_text(encoding="utf-8"))["links"]


def payload(anims) -> dict:
    """The quiz part of assets/data.js: the questions, each with the animations that explain it."""
    links, names = load_links(), {a["name"] for a in anims}
    items, per_anim = [], collections.Counter()
    for q in qc.build():
        if q["q"] not in links:
            raise SystemExit(f"scripts/quiz_links.json has no entry for the question {q['q']!r}")
        targets = links[q["q"]]
        bad = [n for n in targets if n not in names]
        if bad or not targets:
            raise SystemExit(f"scripts/quiz_links.json: {q['q']!r} points at unknown animations {bad}")
        per_anim.update(targets)
        items.append({"id": question_id(q["q"]), "phase": q["phase"], "q": q["q"], "a": q["a"], "anims": targets})
    if len({i["id"] for i in items}) != len(items):
        raise SystemExit("two quiz questions have the same text")
    return {"quiz": items, "quizFor": dict(per_anim)}


def write_page(out: pathlib.Path, data: dict, version: str) -> None:
    import build_site
    (out / "quiz").mkdir(parents=True, exist_ok=True)
    page = build_site.fill(PAGE, version, len(data["quiz"]), "../", "quiz.js")
    (out / "quiz" / "index.html").write_text(page, encoding="utf-8")
    print(f"quiz: {len(data['quiz'])} questions, each linked to an animation -> {out / 'quiz'}")


def main():
    import build_site
    build_site.main()


if __name__ == "__main__":
    main()
