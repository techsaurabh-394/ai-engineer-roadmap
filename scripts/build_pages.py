#!/usr/bin/env python3
"""Static, indexable article pages for every topic (standard library only).

Called at the end of scripts/build_site.py. The interactive viewer uses hash links (#rag-flow), which search
engines treat as one page. This step writes one plain HTML page per topic so each lesson can be found,
shared and cited on its own:

  _site/learn/index.html               every topic, grouped by phase
  _site/learn/<slug>/index.html        one article per topic Markdown file, with its animations
  _site/img/*.svg                      diagrams from assets/ that the articles use
  _site/sitemap.xml                    only when SITE_URL is set (sitemaps need absolute URLs);
                                       submit it in Google Search Console

Optional environment variables (all can be left unset; the site then builds exactly as before plus /learn/):
  SITE_URL            public address of the site, e.g. https://mernstackdev.com/ai-roadmap
                      adds canonical links, Open Graph URLs, JSON-LD and sitemap.xml
  PARENT_SITE_NAME    e.g. MernStackDev    } adds a link back to the main site in the header
  PARENT_SITE_URL     e.g. https://mernstackdev.com
  ADSENSE_CLIENT      e.g. ca-pub-1234567890123456: adds the AdSense script to article pages only,
                      never to the interactive lessons or the quiz
"""
import html
import json
import os
import pathlib
import re
import shutil
import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES_PREFIX = "https://techsaurabh-394.github.io/ai-engineer-roadmap/"


def env(name: str) -> str:
    return os.environ.get(name, "").strip().rstrip("/") if name in ("SITE_URL", "PARENT_SITE_URL") else os.environ.get(name, "").strip()


# ------------------------------------------------------------------ slugs and links
def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text).strip("-")


def topic_slug(path: pathlib.Path, taken: set) -> str:
    base = re.sub(r"^\d+-", "", path.stem)
    slug = base if base not in taken else f"{base}-{path.parent.parent.name[:2]}"
    taken.add(slug)
    return slug


class Linker:
    """Rewrites links and images in topic Markdown so they work on the built site."""

    def __init__(self, repo: str, slugs: dict, out: pathlib.Path):
        self.repo, self.slugs, self.out = repo, slugs, out
        self.copied = set()

    def href(self, url: str, src: pathlib.Path) -> str:
        if url.startswith("#") or url.startswith("mailto:"):
            return url
        if url.startswith(PAGES_PREFIX):
            return "../../" + url[len(PAGES_PREFIX):]
        if re.match(r"^[a-z]+://", url):
            return url
        path, _, anchor = url.partition("#")
        target = (src.parent / path).resolve()
        try:
            rel = target.relative_to(ROOT).as_posix()
        except ValueError:
            return self.repo
        if rel in self.slugs:
            return f"../{self.slugs[rel]}/" + (f"#{anchor}" if anchor else "")
        kind = "tree" if target.is_dir() else "blob"
        return f"{self.repo}/{kind}/main/{rel}" + (f"#{anchor}" if anchor else "")

    def img(self, url: str, src: pathlib.Path) -> str:
        if re.match(r"^[a-z]+://", url):
            return url
        target = (src.parent / url).resolve()
        try:
            rel = target.relative_to(ROOT).as_posix()
        except ValueError:
            return ""
        if rel.startswith("assets/anim/"):
            return "../../anim/" + target.name
        if rel.startswith("assets/") and target.exists():
            dst = self.out / "img" / target.name
            if target.name not in self.copied:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, dst)
                self.copied.add(target.name)
            return "../../img/" + target.name
        return f"{self.repo}/raw/main/{rel}"


# ------------------------------------------------------------------ Markdown
def inline(text: str, link) -> str:
    """Inline Markdown -> HTML: code, links, bold, italics. Everything else is escaped."""
    stash = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    def fmt(t: str) -> str:
        # placeholders (\x00n\x00) pass through escaping and emphasis untouched
        t = html.escape(t, quote=False)
        t = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", t)  # allows *italics* inside
        t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
        return re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", t)

    text = re.sub(r"`([^`]+)`", lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)

    def a(m):
        label, url = m.group(1), m.group(2).strip()
        target = link(url)
        ext = ' rel="noopener"' if re.match(r"^[a-z]+://", target) and "mernstackdev.com" not in target else ""
        return keep(f'<a href="{html.escape(target, quote=True)}"{ext}>{fmt(label)}</a>')

    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", a, text)
    text = fmt(text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def render(md: str, link, img) -> tuple:
    """Markdown -> (title, html, toc). Covers what the topic pages use: headings, paragraphs, lists,
    code fences, tables, block quotes and images."""
    lines = md.replace("\r\n", "\n").split("\n")
    out, toc, title = [], [], ""
    i, ids = 0, set()

    def heading_id(text):
        base = slugify(text) or "section"
        hid, n = base, 2
        while hid in ids:
            hid, n = f"{base}-{n}", n + 1
        ids.add(hid)
        return hid

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--"):
            i += 1
            continue
        fence = re.match(r"^(```+|~~~+)\s*([\w+-]*)", stripped)
        if fence:
            mark, lang, buf = fence.group(1), fence.group(2), []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(mark):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        h = re.match(r"^(#{1,4})\s+(.+?)\s*#*$", stripped)
        if h:
            level, text = len(h.group(1)), h.group(2)
            if level == 1 and not title:
                title = re.sub(r"[*`]", "", text)
            else:
                level = max(level, 2)
                hid = heading_id(text)
                if level == 2:
                    toc.append((hid, re.sub(r"<[^>]+>", "", inline(text, link))))
                out.append(f'<h{level} id="{hid}">{inline(text, link)}</h{level}>')
            i += 1
            continue
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            text = " ".join(b for b in buf if b)
            cls = ' class="meta"' if "**Level:**" in text else ""
            out.append(f"<blockquote{cls}><p>{inline(text, link)}</p></blockquote>")
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            head = cells(lines[i])
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            t = ["<div class=\"table-scroll\"><table><thead><tr>"]
            t += [f"<th>{inline(c, link)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c, link)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        image = re.match(r"^!\[([^\]]*)\]\(([^)\s]+)\)\s*$", stripped)
        if image:
            src = img(image.group(2))
            if src:
                alt = html.escape(image.group(1), quote=True)
                out.append(f'<figure><img src="{html.escape(src, quote=True)}" alt="{alt}" loading="lazy" decoding="async">'
                           f"<figcaption>{html.escape(image.group(1))}</figcaption></figure>")
            i += 1
            continue
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            out.append(render_list(lines, i, link))
            i = render_list.end
            continue
        buf = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", ">", "```", "~~~", "|", "<!--")) or re.match(r"^([-*+]|\d+\.)\s+", nxt)
                    or re.match(r"^!\[", nxt)):
                break
            buf.append(nxt)
            i += 1
        out.append(f"<p>{inline(' '.join(buf), link)}</p>")
    return title, "\n".join(out), toc


def render_list(lines, start, link):
    """Ordered and unordered lists, nested by indentation. Sets render_list.end to the next line index."""
    def indent(s):
        return len(s) - len(s.lstrip(" "))

    def walk(i, base):
        ordered = bool(re.match(r"^\s*\d+\.\s", lines[i]))
        tag = "ol" if ordered else "ul"
        items = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                nxt = next((l for l in lines[i + 1:] if l.strip()), "")
                if re.match(r"^\s*([-*+]|\d+\.)\s+", nxt) and indent(nxt) >= base:
                    i += 1
                    continue
                break
            ind = indent(line)
            m = re.match(r"^\s*([-*+]|\d+\.)\s+(.*)$", line)
            if m and ind == base:
                items.append([inline(m.group(2), link), ""])
                i += 1
            elif m and ind > base and items:
                sub, i = walk(i, ind)
                items[-1][1] += sub
            elif not m and ind > base and items:
                items[-1][0] += " " + inline(line.strip(), link)
                i += 1
            else:
                break
        body = "".join(f"<li>{text}{sub}</li>" for text, sub in items)
        return f"<{tag}>{body}</{tag}>", i

    html_out, end = walk(start, indent(lines[start]))
    render_list.end = end
    return html_out


# ------------------------------------------------------------------ page template
CSS = """
.article-top{display:flex;align-items:center;gap:var(--s4);padding:var(--s3) var(--s5);border-bottom:1px solid var(--line);background:var(--surface);flex-wrap:wrap}
.article-top a{color:var(--text-2);text-decoration:none;font-weight:600;font-size:var(--fs-sm)}
.article-top a:hover{color:var(--accent)}
.article-top .home{display:flex;align-items:center;gap:var(--s2);color:var(--text);font-size:var(--fs-md)}
.article-top .home .logo{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:7px;background:var(--accent);color:var(--accent-ink);font-size:12px}
.article-top .grow{flex:1}
.article{max-width:760px;margin:0 auto;padding:var(--s6) var(--s5) var(--s8);font-size:17px;line-height:1.7;color:var(--text)}
.crumbs{font-size:var(--fs-sm);color:var(--muted);margin-bottom:var(--s4)}
.crumbs a{color:var(--muted)}
.article h1{font-size:clamp(28px,4vw,var(--fs-3xl));line-height:1.15;margin:0 0 var(--s4)}
.article h2{font-size:var(--fs-xl);margin:var(--s7) 0 var(--s3);line-height:1.25}
.article h3{font-size:var(--fs-lg);margin:var(--s6) 0 var(--s2)}
.article a{color:var(--accent)}
.article p,.article ul,.article ol{margin:0 0 var(--s4)}
.article li{margin-bottom:var(--s2)}
.article blockquote{margin:0 0 var(--s5);padding:var(--s3) var(--s4);border-left:4px solid var(--accent);background:var(--accent-soft);border-radius:var(--r1)}
.article blockquote p{margin:0}
.article blockquote.meta{font-size:var(--fs-sm)}
.article code{font-family:var(--mono);font-size:.88em;background:var(--surface-2);padding:.1em .35em;border-radius:5px}
.article pre{background:var(--stage-bg);border:1px solid var(--line);border-radius:var(--r2);padding:var(--s4);overflow-x:auto;font-size:14px;line-height:1.55}
.article pre code{background:none;padding:0}
.article figure{margin:var(--s5) 0;padding:var(--s3);background:var(--stage-bg);border:1px solid var(--line);border-radius:var(--r2)}
.article figure img{display:block;width:100%;height:auto}
.article figcaption{font-size:var(--fs-sm);color:var(--muted);margin-top:var(--s2)}
.table-scroll{overflow-x:auto;margin:0 0 var(--s5);border:1px solid var(--line);border-radius:var(--r2)}
.article table{border-collapse:collapse;width:100%;font-size:15px}
.article th,.article td{padding:var(--s2) var(--s3);border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.article th{background:var(--surface-2)}
.cta{display:flex;flex-wrap:wrap;gap:var(--s3);margin:var(--s5) 0}
.cta a{display:inline-flex;align-items:center;min-height:var(--tap);padding:0 var(--s4);border-radius:var(--r1);font-weight:600;text-decoration:none;border:1px solid var(--line-strong);color:var(--text)}
.cta a.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.toc{background:var(--surface);border:1px solid var(--line);border-radius:var(--r2);padding:var(--s3) var(--s4);margin:0 0 var(--s5)}
.toc summary{cursor:pointer;font-weight:600}
.toc ol{margin:var(--s2) 0 0}
.pager{display:grid;grid-template-columns:1fr 1fr;gap:var(--s3);margin-top:var(--s7)}
.pager a{display:block;padding:var(--s3) var(--s4);border:1px solid var(--line);border-radius:var(--r2);text-decoration:none;background:var(--surface);color:var(--text)}
.pager a span{display:block;font-size:var(--fs-xs);color:var(--muted)}
.pager .next{text-align:right}
.phase{margin:var(--s7) 0 var(--s3)}
.lesson-list{list-style:none;padding:0}
.lesson-list li{padding:var(--s3) 0;border-bottom:1px solid var(--line);margin:0}
.lesson-list a{font-weight:600;text-decoration:none}
.lesson-list p{margin:var(--s1) 0 0;color:var(--text-2);font-size:15px}
.lesson-list small{color:var(--muted)}
@media (max-width:560px){.pager{grid-template-columns:1fr}.pager .next{text-align:left}}
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="color-scheme" content="light dark">
{canonical}
<meta property="og:type" content="{og_type}">
<meta property="og:title" content="{og_title}">
<meta property="og:description" content="{desc}">
{og_url}<link rel="icon" href="{favicon}">
{theme_boot}
<link rel="stylesheet" href="{root}assets/theme.css?v={version}">
<style>{css}</style>
{jsonld}{ads}</head><body>
<a class="skip" href="#main">Skip to the content</a>
<header class="article-top">
<a class="home" href="{root}"><span class="logo" aria-hidden="true">AI</span>AI Engineer Roadmap</a>
<span class="grow"></span>
<a href="{root}learn/">All lessons</a>
<a href="{root}">Interactive</a>
<a href="{root}quiz/">Quiz</a>
{parent}<a href="{repo}" rel="noopener">GitHub</a>
</header>
<main id="main" class="article">
{body}
</main>
<script src="{root}assets/shared.js?v={version}" defer></script>
</body></html>
"""


def jsonld(obj) -> str:
    return '<script type="application/ld+json">' + json.dumps(obj, ensure_ascii=False).replace("</", "<\\/") + "</script>\n"


def page(**kw) -> str:
    site, parent_name, parent_url, ads = env("SITE_URL"), env("PARENT_SITE_NAME"), env("PARENT_SITE_URL"), env("ADSENSE_CLIENT")
    url = f"{site}/{kw['path']}" if site else ""
    kw.setdefault("og_type", "article")
    return PAGE.format(
        title=html.escape(kw["title"]), desc=html.escape(kw["desc"], quote=True), og_title=html.escape(kw["og_title"], quote=True),
        og_type=kw["og_type"],
        canonical=f'<link rel="canonical" href="{url}">' if url else "",
        og_url=f'<meta property="og:url" content="{url}">\n' if url else "",
        favicon=kw["favicon"], theme_boot=kw["theme_boot"], root=kw["root"], version=kw["version"], css=CSS,
        jsonld="".join(jsonld(j) for j in kw.get("ld", [])) if site else "",
        ads=(f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={html.escape(ads)}" '
             'crossorigin="anonymous"></script>\n') if ads and kw.get("allow_ads") else "",
        parent=f'<a href="{html.escape(parent_url)}">{html.escape(parent_name or "Main site")}</a>\n' if parent_url else "",
        repo=kw["repo"], body=kw["body"],
    )


# ------------------------------------------------------------------ build
def write(out: pathlib.Path, data: dict, version: str, favicon: str, theme_boot: str) -> int:
    repo, site = data["repo"], env("SITE_URL")
    phases = {p["id"]: p for p in data["phases"]}
    files = sorted(ROOT.glob("[01][0-9]-*/topics/*.md"))
    taken, slugs = set(), {}
    for f in files:
        slugs[f.relative_to(ROOT).as_posix()] = topic_slug(f, taken)

    anims_for = {}
    for lesson in data["lessons"]:
        for t in lesson["topics"]:
            anims_for.setdefault(t["path"], []).append(lesson)
    info = {t["path"]: t for lesson in data["lessons"] for t in lesson["topics"]}

    linker = Linker(repo, slugs, out)
    learn = out / "learn"
    if learn.exists():
        shutil.rmtree(learn)
    articles = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        md = f.read_text(encoding="utf-8")
        title, body, toc = render(md, lambda u, f=f: linker.href(u, f), lambda u, f=f: linker.img(u, f))
        pid = f.parent.parent.name[:2]
        one = info.get(rel, {}).get("plain") or re.sub(r"[*`]", "", (re.search(r"^\*\*In one line:\*\*\s*(.+)$", md, re.M) or [None, title])[1])
        level = info.get(rel, {}).get("level", "")
        minutes = info.get(rel, {}).get("min", 0)
        articles.append(dict(rel=rel, slug=slugs[rel], title=title or slugs[rel], body=body, toc=toc, phase=pid,
                             one=one.strip(), level=level, min=minutes, lessons=anims_for.get(rel, [])))

    by_phase = {}
    for a in articles:
        by_phase.setdefault(a["phase"], []).append(a)

    common = dict(favicon=favicon, theme_boot=theme_boot, version=version, repo=repo)
    for pid, items in by_phase.items():
        for n, a in enumerate(items):
            phase = phases.get(pid, {"name": f"Phase {int(pid)}", "n": int(pid)})
            cta = "".join(f'<a class="primary" href="../../#{l["name"]}">Play the interactive lesson: {html.escape(l["title"])}</a>'
                          for l in a["lessons"][:2])
            cta += f'<a href="{repo}/blob/main/{a["rel"]}" rel="noopener">Read on GitHub</a>'
            toc = ""
            if len(a["toc"]) >= 4:
                toc = ('<details class="toc"><summary>On this page</summary><ol>'
                       + "".join(f'<li><a href="#{hid}">{text}</a></li>' for hid, text in a["toc"]) + "</ol></details>")
            prev_a, next_a = (items[n - 1] if n else None), (items[n + 1] if n + 1 < len(items) else None)
            pager = '<nav class="pager" aria-label="More lessons in this phase">'
            pager += (f'<a href="../{prev_a["slug"]}/"><span>Previous</span>{html.escape(prev_a["title"])}</a>' if prev_a else "<span></span>")
            pager += (f'<a class="next" href="../{next_a["slug"]}/"><span>Next</span>{html.escape(next_a["title"])}</a>' if next_a else "<span></span>")
            pager += "</nav>"
            crumbs = (f'<nav class="crumbs" aria-label="Breadcrumb"><a href="../../">AI Engineer Roadmap</a> / '
                      f'<a href="../#phase-{pid}">Phase {phase["n"]}: {html.escape(phase["name"])}</a></nav>')
            body = f"{crumbs}<h1>{html.escape(a['title'])}</h1>{cta and '<div class=cta>' + cta + '</div>'}{toc}{a['body']}{pager}"
            body = body.replace("<div class=cta>", '<div class="cta">')
            path = f"learn/{a['slug']}/"
            ld = []
            if site:
                ld = [{
                    "@context": "https://schema.org", "@type": "TechArticle", "headline": a["title"][:110],
                    "description": a["one"], "url": f"{site}/{path}", "inLanguage": "en",
                    "proficiencyLevel": a["level"] or None, "timeRequired": f"PT{a['min']}M" if a["min"] else None,
                    "isPartOf": {"@type": "Course", "name": "AI Engineer Roadmap", "url": f"{site}/"},
                    "license": f"{repo}/blob/main/LICENSE",
                }, {
                    "@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
                        {"@type": "ListItem", "position": 1, "name": "AI Engineer Roadmap", "item": f"{site}/"},
                        {"@type": "ListItem", "position": 2, "name": f"Phase {phase['n']}: {phase['name']}", "item": f"{site}/learn/"},
                        {"@type": "ListItem", "position": 3, "name": a["title"], "item": f"{site}/{path}"},
                    ]}]
                ld[0] = {k: v for k, v in ld[0].items() if v is not None}
            target = out / path
            target.mkdir(parents=True, exist_ok=True)
            (target / "index.html").write_text(page(
                title=f"{a['title']} | AI Engineer Roadmap", og_title=a["title"], desc=a["one"][:300], path=path,
                root="../../", body=body, ld=ld, allow_ads=True, **common), encoding="utf-8")

    sections = []
    for pid in sorted(by_phase):
        phase = phases.get(pid, {"name": f"Phase {int(pid)}", "n": int(pid)})
        rows = []
        for a in by_phase[pid]:
            meta = f' <small>{html.escape(a["level"])} · {a["min"]} min</small>' if a["min"] else ""
            rows.append(f'<li><a href="{a["slug"]}/">{html.escape(a["title"])}</a>{meta}<p>{html.escape(a["one"])}</p></li>')
        rows = "".join(rows)
        sections.append(f'<h2 class="phase" id="phase-{pid}">Phase {phase["n"]}: {html.escape(phase["name"])}</h2><ul class="lesson-list">{rows}</ul>')
    intro = (f"<h1>All {len(articles)} AI engineering lessons</h1>"
             "<p>Every lesson in the AI Engineer Roadmap as a readable article: RAG, embeddings, LLM internals, agents, "
             "MCP and production AI, from beginner to advanced. Each one links to its interactive, step-by-step animation.</p>"
             '<div class="cta"><a class="primary" href="../">Open the interactive roadmap</a><a href="../quiz/">Test yourself with the quiz</a></div>')
    ld_index = [{"@context": "https://schema.org", "@type": "Course", "name": "AI Engineer Roadmap",
                 "description": f"{len(articles)} free lessons on RAG, LLMs, agents, MCP and production AI.",
                 "url": f"{site}/", "isAccessibleForFree": True,
                 "provider": {"@type": "Organization", "name": env("PARENT_SITE_NAME") or "AI Engineer Roadmap",
                              "url": env("PARENT_SITE_URL") or repo},
                 "hasCourseInstance": {"@type": "CourseInstance", "courseMode": "Online", "courseWorkload": "PT40H"}}] if site else []
    (learn / "index.html").write_text(page(
        title="All AI Engineering Lessons: RAG, LLMs, Agents and MCP | AI Engineer Roadmap", og_title="All AI engineering lessons",
        desc=f"{len(articles)} free, beginner-friendly lessons on RAG, embeddings, LLMs, agents, MCP and production AI, each with an animated diagram.",
        path="learn/", root="../", body=intro + "".join(sections), ld=ld_index, allow_ads=True, og_type="website", **common),
        encoding="utf-8")

    if site:
        today = datetime.date.today().isoformat()
        urls = ["", "quiz/", "learn/"] + [f"learn/{a['slug']}/" for a in articles]
        xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
        xml += [f"<url><loc>{html.escape(site)}/{u}</loc><lastmod>{today}</lastmod></url>" for u in urls]
        xml.append("</urlset>")
        (out / "sitemap.xml").write_text("\n".join(xml) + "\n", encoding="utf-8")
        (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {site}/sitemap.xml\n", encoding="utf-8")
    return len(articles)


def add_head_tags(page_html: str, path: str, extra_nav: bool = True) -> str:
    """Canonical link for the existing index and quiz pages, plus a visible link to the article list."""
    site = env("SITE_URL")
    if site:
        page_html = page_html.replace("</head>", f'<link rel="canonical" href="{site}/{path}">\n'
                                                  f'<meta property="og:url" content="{site}/{path}">\n</head>', 1)
    ads = env("ADSENSE_CLIENT")
    if ads and "adsbygoogle.js" not in page_html:
        page_html = page_html.replace("</head>", f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={html.escape(ads)}"\n     crossorigin="anonymous"></script>\n</head>', 1)
    parent_url, parent_name = env("PARENT_SITE_URL"), env("PARENT_SITE_NAME") or "Main site"
    if parent_url:
        page_html = page_html.replace('<span class="grow"></span>',
                                      f'<a class="parent-link" href="{html.escape(parent_url)}" rel="noopener">&larr; {html.escape(parent_name)}</a><span class="grow"></span>', 1)
    return page_html
