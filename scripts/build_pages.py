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
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seo  # noqa: E402
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
.article-foot{max-width:760px;margin:0 auto;padding:var(--s5);color:var(--muted);font-size:var(--fs-sm);border-top:1px solid var(--line)}
.related{margin:var(--s6) 0}.related h2{font-size:var(--fs-lg)!important;margin:0 0 var(--s3)!important}
.answer-first{font-size:1.08em;font-weight:500;border-left:4px solid var(--accent);padding:var(--s2) var(--s4);background:var(--accent-soft);border-radius:var(--r1);margin:0 0 var(--s5)}
.meta-line{color:var(--muted);font-size:var(--fs-sm);margin:0 0 var(--s4)}
.glossary dt{font-weight:700;margin-top:var(--s4);scroll-margin-top:72px}.glossary dd{margin:2px 0 0;color:var(--text)}.glossary dt:target{background:var(--accent-soft);border-radius:var(--r1)}
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
{og_url}{extra_head}<link rel="icon" href="{favicon}">
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
<footer class="article-foot">{foot}</footer>
<script src="{root}assets/shared.js?v={version}" defer></script>
</body></html>
"""


def jsonld(obj) -> str:
    return '<script type="application/ld+json">' + json.dumps(obj, ensure_ascii=False).replace("</", "<\\/") + "</script>\n"


def page(**kw) -> str:
    site, parent_name, parent_url, ads = env("SITE_URL"), env("PARENT_SITE_NAME"), env("PARENT_SITE_URL"), env("ADSENSE_CLIENT")
    url = f"{site}/{kw['path']}" if site else ""
    kw.setdefault("og_type", "article")
    kw.setdefault("extra_head", "")
    return PAGE.format(
        title=html.escape(kw["title"]), desc=html.escape(kw["desc"], quote=True), og_title=html.escape(kw["og_title"], quote=True),
        og_type=kw["og_type"],
        canonical=f'<link rel="canonical" href="{url}">' if url else "",
        og_url=f'<meta property="og:url" content="{url}">\n' if url else "",
        extra_head=kw["extra_head"],
        favicon=kw["favicon"], theme_boot=kw["theme_boot"], root=kw["root"], version=kw["version"], css=CSS,
        jsonld="".join(jsonld(j) for j in kw.get("ld", [])) if site else "",
        ads=(f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={html.escape(ads)}" '
             'crossorigin="anonymous"></script>\n') if ads and kw.get("allow_ads") else "",
        parent=f'<a href="{html.escape(parent_url)}">{html.escape(parent_name or "Main site")}</a>\n' if parent_url else "",
        repo=kw["repo"], body=kw["body"], foot=kw.get("foot", ""),
    )


# ------------------------------------------------------------------ build
def phase_label(pid: str, phases: dict) -> dict:
    p = phases.get(pid, {"name": f"Phase {int(pid)}", "n": int(pid)})
    return {"n": p["n"], "name": p["name"], "hub": f"learn/phase-{p['n']}/"}


def write(out: pathlib.Path, data: dict, version: str, favicon: str, theme_boot: str) -> int:
    repo, site = data["repo"], env("SITE_URL")
    parent_name, parent_url = env("PARENT_SITE_NAME"), env("PARENT_SITE_URL")
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
    (out / "img").mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "docs" / "social-preview.png", out / seo.OG_IMAGE)
    articles = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        md = f.read_text(encoding="utf-8")
        title, body, toc = render(md, lambda u, f=f: linker.href(u, f), lambda u, f=f: linker.img(u, f))
        pid = f.parent.parent.name[:2]
        one = info.get(rel, {}).get("plain") or re.sub(r"[*`]", "", (re.search(r"^\*\*In one line:\*\*\s*(.+)$", md, re.M) or [None, title])[1])
        level = info.get(rel, {}).get("level", "")
        minutes = info.get(rel, {}).get("min", 0)
        pub, mod = seo.file_dates(rel)
        articles.append(dict(rel=rel, slug=slugs[rel], title=title or slugs[rel], body=body, toc=toc, phase=pid,
                             one=one.strip(), level=level, min=minutes, lessons=anims_for.get(rel, []),
                             faq=seo.faq_pairs(md), words=len(seo.plain(md).split()), pub=pub, mod=mod))

    by_phase = {}
    for a in articles:
        by_phase.setdefault(a["phase"], []).append(a)
    by_slug = {a["slug"]: a for a in articles}
    lessons_total = len(data["lessons"])
    labs_total = len(data["labs"]["meta"])
    summary = (f"A free, open-source, interactive course on AI engineering: {len(articles)} lessons with animations, {labs_total} hands-on labs, "
               "runnable Python demos and quiz cards covering ML basics, RAG, LLM internals, agents, MCP, production AI, fine-tuning and multimodal AI.")
    common = dict(favicon=favicon, theme_boot=theme_boot, version=version, repo=repo)

    def related(a, items, n):
        """Up to 4 links: the lessons the page already points at, then neighbours in the phase."""
        picked, seen = [], {a["slug"]}
        for m in re.finditer(r'href="\.\./([a-z0-9-]+)/', a["body"]):
            x = by_slug.get(m.group(1))
            if x and x["slug"] not in seen and len(picked) < 4:
                seen.add(x["slug"]); picked.append(x)
        for k in (n + 1, n - 1, n + 2, n - 2):
            if 0 <= k < len(items) and items[k]["slug"] not in seen and len(picked) < 4:
                seen.add(items[k]["slug"]); picked.append(items[k])
        if not picked:
            return ""
        return ('<aside class="related" aria-label="Related lessons"><h2>Related lessons</h2><ul class="lesson-list">'
                + "".join(f'<li><a href="../{x["slug"]}/">{html.escape(x["title"])}</a><p>{html.escape(seo.trim(x["one"], 140))}</p></li>' for x in picked)
                + "</ul></aside>")

    foot = (f'<p>{html.escape(seo.SITE_NAME)} is free and open source (MIT) by <a href="{seo.AUTHOR["url"]}" rel="noopener">{seo.AUTHOR["name"]}</a>. '
            f'<a href="{repo}/issues" rel="noopener">Report a mistake</a> · <a href="{repo}" rel="noopener">Source on GitHub</a>.</p>')

    for pid, items in by_phase.items():
        phase = phase_label(pid, phases)
        for n, a in enumerate(items):
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
            crumbs = (f'<nav class="crumbs" aria-label="Breadcrumb"><a href="../../">{seo.SITE_NAME}</a> / '
                      f'<a href="../phase-{phase["n"]}/">Phase {phase["n"]}: {html.escape(phase["name"])}</a></nav>')
            meta_line = (f'<p class="meta-line">{html.escape(a["level"] or "Lesson")} · {a["min"]} min read · '
                         f'Updated <time datetime="{a["mod"]}">{a["mod"]}</time> · By {seo.AUTHOR["name"]}</p>') if a["min"] else ""
            answer = f'<p class="answer-first"><strong>In short:</strong> {html.escape(seo.plain(a["one"]))}</p>' if a["one"] else ""
            body_html = a["body"]
            if answer:   # the 'In short' box already says this; drop the identical 'In one line' paragraph that follows
                body_html = re.sub(r"<p><strong>In one line:</strong>.*?</p>", "", body_html, count=1, flags=re.S)
            body = (f"{crumbs}<h1>{html.escape(a['title'])}</h1>{meta_line}{answer}{cta and '<div class=cta>' + cta + '</div>'}{toc}"
                    f"{body_html}{related(a, items, n)}{pager}")
            body = body.replace("<div class=cta>", '<div class="cta">')
            path = f"learn/{a['slug']}/"
            title_t = seo.page_title(a["title"])
            desc = seo.trim(a["one"])
            ld = seo.article_ld(site, path, a, phase, repo, parent_name, parent_url, a["words"], a["faq"], a["pub"], a["mod"]) if site else []
            target = out / path
            target.mkdir(parents=True, exist_ok=True)
            (target / "index.html").write_text(page(
                title=title_t, og_title=a["title"], desc=desc, path=path, root="../../", body=body, ld=ld, allow_ads=True, foot=foot,
                extra_head=seo.head_meta(site, path, a["title"], desc, "article", a["pub"], a["mod"], f"Phase {phase['n']}: {phase['name']}"), **common),
                encoding="utf-8")

    # ---- one hub page per phase: a crawlable list with a short intro, linked from the index and from every lesson
    road = data.get("roadmap", {}).get("phases", {})
    for pid, items in by_phase.items():
        phase = phase_label(pid, phases)
        info_p = road.get(pid, {})
        mins = sum(x["min"] for x in items)
        blurb = info_p.get("why") or f"{len(items)} lessons on {phase['name']}."
        rows = "".join(f'<li><a href="../{x["slug"]}/">{html.escape(x["title"])}</a> <small>{html.escape(x["level"])} · {x["min"]} min</small>'
                       f'<p>{html.escape(seo.trim(x["one"], 220))}</p></li>' for x in items)
        outcomes = "".join(f"<li>{html.escape(o)}</li>" for o in info_p.get("outcomes", []))
        bits = [f'<nav class="crumbs" aria-label="Breadcrumb"><a href="../../">{seo.SITE_NAME}</a> / <a href="../">All lessons</a></nav>',
                f'<h1>Phase {phase["n"]}: {html.escape(phase["name"])}</h1>',
                f'<p class="meta-line">{len(items)} lessons · about {mins} minutes of reading</p>']
        if info_p.get("tagline"):
            bits.append(f'<p class="answer-first"><strong>{html.escape(info_p["tagline"])}.</strong> {html.escape(blurb)}</p>')
        if outcomes:
            bits.append(f"<h2>After this phase you can</h2><ul>{outcomes}</ul>")
        if info_p.get("needs"):
            bits.append(f'<p><strong>Before you start:</strong> {html.escape(info_p["needs"])}</p>')
        bits.append(f'<h2>Lessons in this phase</h2><ul class="lesson-list">{rows}</ul>')
        bits.append(f'<div class="cta"><a class="primary" href="../../#{items[0]["lessons"][0]["name"] if items[0]["lessons"] else ""}">Start with the interactive lesson</a>'
                    f'<a href="../../quiz/">Take the quiz</a></div>')
        path = phase["hub"]
        d = seo.trim(f"{info_p.get('tagline', phase['name'])}. {blurb}", 158)
        ttl = seo.page_title(f"Phase {phase['n']}: {phase['name']} lessons")
        (out / path).mkdir(parents=True, exist_ok=True)
        (out / path / "index.html").write_text(page(
            title=ttl, og_title=f"Phase {phase['n']}: {phase['name']}", desc=d, path=path, root="../../", body="".join(bits),
            ld=seo.hub_ld(site, path, phase, items, repo, parent_name, parent_url, d) if site else [], allow_ads=True, og_type="website", foot=foot,
            extra_head=seo.head_meta(site, path, f"Phase {phase['n']}: {phase['name']}", d, "website"), **common), encoding="utf-8")

    # ---- glossary: every term as a DefinedTerm (good for 'what is X' questions)
    gloss_md = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8") if (ROOT / "docs" / "glossary.md").exists() else ""
    terms = seo.glossary_terms(gloss_md)
    gloss_path = None
    if terms:
        gloss_path = "glossary/"
        rows = "".join(f'<dt id="{slugify(t)}">{html.escape(t)}</dt><dd>{html.escape(d)}</dd>' for t, d in terms)
        d = f"{len(terms)} AI engineering terms in plain English: token, embedding, RAG, agent, MCP, LoRA, attention and more."
        (out / "glossary").mkdir(exist_ok=True)
        (out / "glossary" / "index.html").write_text(page(
            title=seo.page_title("AI Engineering Glossary: plain-English definitions"), og_title="AI engineering glossary", desc=d, path=gloss_path,
            root="../", body=('<nav class="crumbs" aria-label="Breadcrumb"><a href="../">' + seo.SITE_NAME + '</a></nav><h1>AI engineering glossary</h1>'
                              f'<p class="answer-first">{html.escape(d)}</p><dl class="glossary">{rows}</dl>'),
            ld=seo.glossary_ld(site, gloss_path, terms, repo, parent_name, parent_url) if site else [], allow_ads=True, og_type="website", foot=foot,
            extra_head=seo.head_meta(site, gloss_path, "AI engineering glossary", d, "website"), **common), encoding="utf-8")

    # ---- the index of all lessons
    sections = []
    for pid in sorted(by_phase):
        phase = phase_label(pid, phases)
        rows = []
        for a in by_phase[pid]:
            meta = f' <small>{html.escape(a["level"])} · {a["min"]} min</small>' if a["min"] else ""
            rows.append(f'<li><a href="{a["slug"]}/">{html.escape(a["title"])}</a>{meta}<p>{html.escape(a["one"])}</p></li>')
        sections.append(f'<h2 class="phase" id="phase-{pid}"><a href="phase-{phase["n"]}/">Phase {phase["n"]}: {html.escape(phase["name"])}</a></h2>'
                        f'<ul class="lesson-list">{"".join(rows)}</ul>')
    intro = (f"<h1>All {len(articles)} AI engineering lessons</h1>"
             "<p>Every lesson in the AI Engineer Roadmap as a readable article: machine-learning basics, RAG, embeddings, LLM internals, agents, "
             "MCP, production AI, fine-tuning and multimodal AI, from beginner to advanced. Each one links to its interactive, step-by-step animation.</p>"
             '<div class="cta"><a class="primary" href="../">Open the interactive roadmap</a><a href="../quiz/">Test yourself with the quiz</a>'
             + ('<a href="../glossary/">Glossary</a>' if terms else "") + "</div>")
    ld_index = ([{"@context": "https://schema.org", "@type": "CollectionPage", "@id": f"{site}/learn/", "url": f"{site}/learn/", "name": "All AI engineering lessons",
                  "inLanguage": "en", "isPartOf": {"@type": "Course", "name": seo.SITE_NAME, "url": f"{site}/"},
                  "mainEntity": {"@type": "ItemList", "numberOfItems": len(articles), "itemListElement": [
                      {"@type": "ListItem", "position": i + 1, "name": a["title"], "url": f"{site}/learn/{a['slug']}/"} for i, a in enumerate(articles)]}}]
                if site else [])
    d = (f"{len(articles)} free lessons on machine learning basics, RAG, LLMs, agents, MCP, production AI, fine-tuning and multimodal AI, "
         "each with an animated diagram.")
    (learn / "index.html").write_text(page(
        title="All AI Engineering Lessons: RAG, LLMs, Agents, MCP | " + seo.SITE_NAME, og_title="All AI engineering lessons", desc=seo.trim(d, 158),
        path="learn/", root="../", body=intro + "".join(sections), ld=ld_index, allow_ads=True, og_type="website", foot=foot,
        extra_head=seo.head_meta(site, "learn/", "All AI engineering lessons", seo.trim(d, 158), "website"), **common), encoding="utf-8")

    if site:
        urls = [("", None, "1.0", "weekly"), ("learn/", None, "0.9", "weekly"), ("quiz/", None, "0.6", "monthly")]
        urls += [(f"learn/phase-{phase_label(pid, phases)['n']}/", None, "0.8", "weekly") for pid in sorted(by_phase)]
        if gloss_path:
            urls.append((gloss_path, None, "0.7", "monthly"))
        urls += [(f"learn/{a['slug']}/", a["mod"], "0.7", "monthly") for a in articles]
        today = datetime.date.today().isoformat()
        xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">']
        for u, last, prio, freq in urls:
            img = f"<image:image><image:loc>{html.escape(site)}/{seo.OG_IMAGE}</image:loc></image:image>" if u == "" else ""
            xml.append(f"<url><loc>{html.escape(site)}/{u}</loc><lastmod>{last or today}</lastmod><changefreq>{freq}</changefreq><priority>{prio}</priority>{img}</url>")
        xml.append("</urlset>")
        (out / "sitemap.xml").write_text("\n".join(xml) + "\n", encoding="utf-8")
        (out / "robots.txt").write_text(seo.robots(site), encoding="utf-8")
        sects = [("Start here", [(seo.SITE_NAME + ": interactive course", f"{site}/", "The interactive viewer: animations, labs, progress tracking"),
                                 ("All lessons", f"{site}/learn/", f"Index of all {len(articles)} lessons"), ("Quiz", f"{site}/quiz/", "Self-check flash cards for every phase")]
                                + ([("Glossary", f"{site}/glossary/", f"{len(terms)} AI engineering terms in plain English")] if terms else []))]
        for pid in sorted(by_phase):
            phase = phase_label(pid, phases)
            sects.append((f"Phase {phase['n']}: {phase['name']}", [(x["title"], f"{site}/learn/{x['slug']}/", seo.trim(x["one"], 200)) for x in by_phase[pid]]))
        (out / "llms.txt").write_text(seo.llms_txt(site, summary, sects, [("Source code and demos", repo, "Runnable Python demos with tests, MIT license")]), encoding="utf-8")
        (out / "llms-full.txt").write_text(seo.llms_full(site, summary, articles), encoding="utf-8")
    return len(articles)


def add_head_tags(page_html: str, path: str, extra_nav: bool = True, data: dict = None) -> str:
    """Canonical, Open Graph, Twitter card, JSON-LD and the parent-site link for the interactive home page and the quiz."""
    site = env("SITE_URL")
    parent_url, parent_name = env("PARENT_SITE_URL"), env("PARENT_SITE_NAME") or "Main site"
    if site:
        m = re.search(r'<meta name="description" content="([^"]*)"', page_html)
        t = re.search(r"<title>([^<]*)</title>", page_html)
        desc = html.unescape(m.group(1)) if m else seo.SITE_NAME
        title = html.unescape(t.group(1)) if t else seo.SITE_NAME
        if path == "":   # a keyword-bearing title and a description that fits in a search result
            title = "AI Engineer Roadmap: learn RAG, LLMs, agents, MCP and production AI"
            desc = ("Free interactive AI engineering course: animated lessons, hands-on labs and quizzes on RAG, LLM internals, agents, MCP, "
                    "fine-tuning and multimodal AI.")
            page_html = re.sub(r"<title>[^<]*</title>", f"<title>{html.escape(title)}</title>", page_html, count=1)
            page_html = re.sub(r'<meta name="description" content="[^"]*"', f'<meta name="description" content="{html.escape(desc, quote=True)}"', page_html, count=1)
            page_html = re.sub(r'<meta property="og:title" content="[^"]*"', f'<meta property="og:title" content="{html.escape(title, quote=True)}"', page_html, count=1)
        extra = (f'<link rel="canonical" href="{site}/{path}">\n<meta property="og:url" content="{site}/{path}">\n'
                 f'<meta property="og:type" content="website">\n' + seo.head_meta(site, path, title, desc, "website"))
        if path == "" and data:
            extra += "".join(jsonld(j) for j in seo.home_ld(site, data["repo"], env("PARENT_SITE_NAME"), parent_url, data["phases"],
                                                            len(data["lessons"]), len(data["labs"]["meta"])))
        page_html = page_html.replace("</head>", extra + "</head>", 1)
        # the interactive page is script-rendered: give crawlers and no-JS readers the real lesson list too
        if path == "" and data:
            links = "".join(f'<li><a href="learn/{a_slug}/">{html.escape(a_title)}</a></li>' for a_slug, a_title in data.get("_crawl", []))
            if links:
                page_html = re.sub(r"</p></noscript>", lambda m: f"</p><h1>AI Engineer Roadmap</h1><p>{html.escape(desc)}</p><ul>{links}</ul></noscript>", page_html, count=1)
    ads = env("ADSENSE_CLIENT")
    if ads and "adsbygoogle.js" not in page_html:
        page_html = page_html.replace("</head>", f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={html.escape(ads)}"\n     crossorigin="anonymous"></script>\n</head>', 1)
    if parent_url:
        page_html = page_html.replace('<span class="grow"></span>',
                                      f'<a class="parent-link" href="{html.escape(parent_url)}" rel="noopener">&larr; {html.escape(parent_name)}</a><span class="grow"></span>', 1)
    return page_html
