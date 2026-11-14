#!/usr/bin/env python3
"""Build the GitHub Pages site into _site/ (standard library only).

    python scripts/build_site.py

What gets built:
  _site/index.html         the lessons viewer: home, one page per animation (#rag-flow ...), labs (#lab/sampling ...)
  _site/quiz/index.html    the quiz (scripts/build_quiz.py)
  _site/assets/            theme.css, the JavaScript from scripts/site/, and data.js
  _site/anim/*.svg         the animated SVGs (same files as assets/anim/)

The page skeleton is the PAGE string below. Styles and scripts live in scripts/site/ as plain files, so
the tests can run exactly the JavaScript the browser runs. data.js is generated here from the animation
specs, the topic pages and the demo code (scripts/site_data.py). No framework, no CDN, no build step.
"""
import hashlib
import html
import json
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_animations as g  # noqa: E402
import site_data  # noqa: E402
import roadmap_data  # noqa: E402
import md_render  # noqa: E402

ROOT = g.core.ROOT
OUT = ROOT / "_site"
SRC = pathlib.Path(__file__).resolve().parent / "site"
REPO = site_data.REPO
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' "
           "fill='%230F766E'/%3E%3Ctext x='16' y='22' font-size='15' text-anchor='middle' font-family='sans-serif' font-weight='700' "
           "fill='%23fff'%3EAI%3C/text%3E%3C/svg%3E")
# Runs before first paint so a saved Light/Dark choice does not flash the other theme.
THEME_BOOT = ("<script>try{var t=JSON.parse(localStorage.getItem('aier-theme'));"
              "if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>")
IC_MENU = '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>'
IC_SEARCH = '<svg class="ic sm" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg>'

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Engineer Roadmap: interactive lessons</title>
<meta name="description" content="__COUNT__ step-by-step lessons and hands-on labs for RAG, LLMs, agents, MCP and production AI. Free and open source.">
<meta property="og:title" content="AI Engineer Roadmap: interactive lessons">
<meta property="og:description" content="__COUNT__ step-by-step lessons and hands-on labs for RAG, LLMs, agents, MCP and production AI.">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="__FAVICON__">
__THEME_BOOT__
<link rel="stylesheet" href="assets/theme.css?v=__V__">
<link rel="stylesheet" href="assets/reading.css?v=__V__">
</head><body>
<a class="skip" href="#main">Skip to the content</a>
<header class="topbar">
<button id="menu" class="icon-btn only-narrow" type="button" aria-label="Open the list of lessons" aria-expanded="false" aria-controls="sidebar">__IC_MENU__</button>
<a class="brand" href="#"><span class="logo" aria-hidden="true">AI</span><span><span class="name-long">AI Engineer </span>Roadmap</span></a>
<nav class="topnav" aria-label="Sections"><a href="#" data-nav="lessons" aria-current="page">Lessons</a><a href="#roadmap" data-nav="roadmap">Roadmap</a><a href="quiz/">Quiz</a><a href="__REPO__/blob/main/README.md#start-in-5-minutes">Start here</a></nav>
<span class="grow"></span>
<div class="searchbox">__IC_SEARCH__<input id="q" type="search" placeholder="Search __COUNT__ lessons" aria-label="Search lessons" autocomplete="off"><kbd aria-hidden="true">/</kbd></div>
<a class="icon-btn only-narrow" href="quiz/">Quiz</a>
<button id="theme" class="icon-btn" type="button" aria-label="Change theme"></button>
<a class="icon-btn" href="__REPO__" rel="noopener"><span class="name-long">GitHub</span><span class="sr-only">repository</span></a>
</header>
<div class="layout">
<aside class="sidebar" id="sidebar" aria-label="All lessons">
<div class="mobile-search"><input id="q2" type="search" placeholder="Search __COUNT__ lessons" aria-label="Search lessons" autocomplete="off"></div>
<div id="nav"></div>
</aside>
<main id="main" tabindex="-1"><div class="wrap"><noscript><p>This page needs JavaScript to show the interactive lessons. Everything is also readable as plain text in the <a href="__REPO__">GitHub repo</a>.</p></noscript></div></main>
</div>
<div class="scrim" hidden></div>
<div class="transport-dock" id="dock" hidden></div>
__SCRIPTS__
</body></html>
"""


# ------------------------------------------------------------------ topic pages
def inline_md(text: str) -> str:
    """One line of topic Markdown -> a small safe HTML subset (code, strong, em). Links become their text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    tokens = []

    def stash(m):
        tokens.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], text)


def plain_md(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"[*`]", "", text)


def topic_info(rel: str) -> dict:
    """Title, level, reading time and the "In one line" summary of a topic page."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    m = re.search(r"\*\*Level:\*\* (\w+).*?\*\*Reading time:\*\* (\d+) min", text)
    if not m:
        raise SystemExit(f"{rel}: no '**Level:** ... **Reading time:** N min' line")
    one = re.search(r"^\*\*In one line:\*\*\s*(.+)$", text, re.M)
    return dict(path=rel, title=text.split("\n")[0].lstrip("# ").strip(), level=m.group(1), min=int(m.group(2)),
                one=inline_md(one.group(1).strip()) if one else "", plain=plain_md(one.group(1).strip()) if one else "",
                demos=site_data.topic_demos(text, rel.split("/")[0]))


def phases() -> list:
    out = []
    for pid, label in sorted(g.PHASES.items()):
        dirs = sorted(p.name for p in ROOT.glob(f"{pid}-*") if p.is_dir())
        out.append(dict(id=pid, n=int(pid), name=label.split(": ", 1)[1], dir=dirs[0] if dirs else ""))
    return out


def lessons(anims) -> list:
    out = []
    for a in anims:
        topics = [topic_info(t) for t in a["topics"]]
        demos = []
        for t in topics:
            demos += [d for d in t.pop("demos") if d not in demos]
        item = dict(name=a["name"], title=a["title"], sub=a["subtitle"], phase=a["phase"], kind=a["kind"], marks=a["marks"],
                    steps=[[s["label"], s["sub"], s["c1"], s["c2"]] for s in a["steps"]], topics=topics, demos=demos)
        if a["kind"] == "bars":
            item["labels"] = a["labels"]
            item["frames"] = [dict(name=f[0], probs=f[1], keep=f[2]) for f in a["frames"]]
        out.append(item)
    out.sort(key=lambda m: (m["phase"], m["topics"][0]["path"]))
    for m in out:
        for t in m["topics"]:
            t["frag"] = frag_name(t["path"])
    return out


def frag_name(path: str) -> str:
    return path.replace("/topics/", "__").removesuffix(".md")


def write_topic_fragments(out: pathlib.Path, items: list) -> int:
    """One HTML fragment per topic page (the full lesson text), fetched when a lesson is opened."""
    lesson_of = {}
    for m in items:
        for t in m["topics"]:
            lesson_of.setdefault(t["path"], m["name"])
    d = out / "topic"
    d.mkdir(parents=True, exist_ok=True)
    seen = set()
    for m in items:
        for t in m["topics"]:
            if t["path"] not in seen:
                seen.add(t["path"])
                (d / (t["frag"] + ".html")).write_text(md_render.topic_html(t["path"], lesson_of), encoding="utf-8")
    return len(seen)


def site_payload(anims, extra=None) -> dict:
    items = lessons(anims)
    names = {m["name"] for m in items}
    for lab, meta in site_data.LAB_META.items():
        missing = [n for n in meta["lessons"] if n not in names]
        if missing:
            raise SystemExit(f"lab {lab!r} is attached to lessons that do not exist: {missing}")
    demos = site_data.demo_outputs()
    used = {d for m in items for d in m["demos"]}
    road = dict(steps=[list(s) for s in roadmap_data.STEPS], after=[list(a) for a in roadmap_data.AFTER],
                phases={k: v for k, v in roadmap_data.PHASES.items()})
    data: dict = dict(repo=REPO, phases=phases(), lessons=items, roadmap=road,
                labs=dict(meta=site_data.LAB_META, data=site_data.labs()),
                demos={k: v for k, v in demos.items() if k in used},
                counts=dict(lessons=len(items), questions=0))
    data.update(extra or {})
    data["counts"]["questions"] = len(data.get("quiz", []))
    return data


# ------------------------------------------------------------------ assets
def asset_files() -> list:
    """Source files copied to _site/assets/, in the order the page loads the scripts."""
    js = [SRC / "core" / "py.js", SRC / "shared.js"]
    js += sorted(p for p in (SRC / "core").glob("*.js") if p.name != "py.js")
    js += sorted((SRC / "labs").glob("*.js"))
    return [SRC / "theme.css"] + js


def copy_assets(out: pathlib.Path) -> str:
    """Copy scripts/site/ into out/assets/ and return a short content hash for cache busting."""
    digest = hashlib.sha256()
    for src in asset_files() + [SRC / "app.js", SRC / "quiz.js", SRC / "quiz.css", SRC / "reading.css"]:
        if not src.exists():
            continue
        dst = out / "assets" / src.relative_to(SRC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest.update(src.read_bytes())
    return digest.hexdigest()[:10]


def script_tags(prefix: str, version: str, entry: str) -> str:
    files = ["data.js"] + [p.relative_to(SRC).as_posix() for p in asset_files() if p.suffix == ".js"] + [entry]
    return "\n".join(f'<script defer src="{prefix}assets/{f}?v={version}"></script>' for f in files)


def write_data(out: pathlib.Path, data: dict) -> str:
    text = "window.AIER_DATA=" + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n"
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "data.js").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def fill(page: str, version: str, count: int, prefix: str, entry: str) -> str:
    return (page.replace("__SCRIPTS__", script_tags(prefix, version, entry)).replace("__THEME_BOOT__", THEME_BOOT)
            .replace("__FAVICON__", FAVICON).replace("__IC_MENU__", IC_MENU).replace("__IC_SEARCH__", IC_SEARCH)
            .replace("__REPO__", REPO).replace("__COUNT__", str(count)).replace("__V__", version))


def main():
    anims = g.build()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "anim").mkdir(parents=True)
    for a in anims:
        (OUT / "anim" / f"{a['name']}.svg").write_text(a["svg"], encoding="utf-8")
    import build_quiz  # the quiz adds its questions to the shared data file
    data = site_payload(anims, build_quiz.payload(anims))
    n_frag = write_topic_fragments(OUT, data["lessons"])
    version = copy_assets(OUT) + write_data(OUT, data)[:4]
    (OUT / "index.html").write_text(fill(PAGE, version, len(data["lessons"]), "", "app.js"), encoding="utf-8")
    (OUT / ".nojekyll").write_text("")
    print(f"site: {len(data['lessons'])} lessons, {len(data['labs']['meta'])} labs, {len(data['demos'])} demo outputs, {n_frag} full lesson texts -> {OUT}")
    build_quiz.write_page(OUT, data, version)


if __name__ == "__main__":
    main()
