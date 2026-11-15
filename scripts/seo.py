"""Search-engine (SEO) and generative-engine (GEO) helpers for the static pages (standard library only).

Used by build_pages.py. Nothing here touches the network. What it produces:

  * short, unique <title> and <meta description> text that is not cut off in search results
  * schema.org JSON-LD built from each lesson's real content: TechArticle, BreadcrumbList, FAQPage
    (from the lesson's own interview questions), CollectionPage + ItemList for phase pages,
    DefinedTermSet for the glossary, WebSite + Course + Organization for the home page
  * llms.txt and llms-full.txt, plain-text indexes meant for AI assistants and crawlers

Honest limits, so nobody expects magic: FAQ rich results in Google are shown for few sites today, so FAQPage
data here is mainly for machines that read the page (AI answer engines). llms.txt is a community proposal
(llmstxt.org); no search engine has promised to use it. Good titles, real content, fast pages, internal links
and links from other sites matter most.
"""
import datetime
import html
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE_NAME = "AI Engineer Roadmap"
AUTHOR = {"@type": "Person", "name": "Saurabh Pathak", "url": "https://github.com/techsaurabh-394",
          "sameAs": ["https://github.com/techsaurabh-394", "https://www.linkedin.com/in/tech-saurabh/"]}
OG_IMAGE = "img/social-preview.png"          # 1280 x 640, copied from docs/social-preview.png by build_pages.py
OG_SIZE = (1280, 640)
KEYWORDS_BY_PHASE = {
    "00": "machine learning basics, neural networks, gradient descent",
    "01": "RAG, retrieval augmented generation, embeddings, vector search",
    "02": "advanced RAG, reranking, RAG evaluation",
    "03": "LLM internals, transformers, attention",
    "04": "AI agents, tool calling, agentic AI",
    "05": "LangChain, LangGraph, LlamaIndex",
    "06": "MCP, Model Context Protocol",
    "07": "production AI, LLMOps, deploying LLM apps",
    "10": "fine-tuning, LoRA, QLoRA, DPO",
    "11": "multimodal AI, vision language models, CLIP, speech",
}


# ------------------------------------------------------------------ text
def plain(md: str) -> str:
    """Markdown or HTML fragment -> one line of plain text."""
    t = re.sub(r"<[^>]+>", " ", md)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[*`_]{1,3}", "", t)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def trim(text: str, limit: int = 155) -> str:
    """Fit text into a snippet. Prefer whole sentences, then a clause break (comma/semicolon/colon), then a word."""
    text = plain(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= limit * 0.5:
        return cut[:end + 1]
    clause = max(cut.rfind(", "), cut.rfind("; "), cut.rfind(": "), cut.rfind(" - "))
    if clause >= limit * 0.55:
        return cut[:clause].rstrip() + "."
    return cut[:cut.rfind(" ")].rstrip(" ,;:-") + "…"


def page_title(title: str, suffix: str = " | " + SITE_NAME, limit: int = 60) -> str:
    """'<title> | AI Engineer Roadmap' when it fits in about 60 characters. Otherwise the title alone, and a title
    still over the limit is shortened at the last ':' or ' and ' so it does not stop mid-phrase."""
    full = title + suffix
    if len(full) <= limit:
        return full
    if len(title) <= limit:
        return title
    head = title[:limit]
    for sep in (":", ",", " and "):
        k = head.rfind(sep)
        if k >= 25:
            return head[:k].rstrip()
    return head[:head.rfind(" ")].rstrip(" ,:")


# ------------------------------------------------------------------ dates (from git, so they are real)
_cache: dict = {}


def _git(args: list) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def file_dates(rel: str) -> tuple:
    """(first commit date, last commit date) as YYYY-MM-DD; today if git history is not available."""
    if rel not in _cache:
        today = datetime.date.today().isoformat()
        last = _git(["log", "-1", "--format=%cs", "--", rel]) or today
        first = (_git(["log", "--diff-filter=A", "--format=%cs", "--", rel]).splitlines() or [last])[-1] or last
        _cache[rel] = (min(first, last), last)
    return _cache[rel]


# ------------------------------------------------------------------ content extraction
FAQ_RE = re.compile(r"(?:\*\*)?\d+\.\s+\*{0,2}(.+?)\*\*\s*<details>\s*<summary>Answer</summary>(.*?)</details>", re.S)
# older lessons put the question inside the summary: <details><summary><b>1. Question?</b></summary> answer </details>
FAQ_RE2 = re.compile(r"<details>\s*<summary>\s*<b>\s*\d+\.\s+(.+?)</b>\s*</summary>(.*?)</details>", re.S)


def faq_pairs(md: str) -> list:
    """[(question, answer)] from a lesson's own 'Interview questions' section, as plain text. Handles both layouts."""
    section = md.split("## Interview questions", 1)[-1]
    out = []
    for q, a in FAQ_RE.findall(section) + FAQ_RE2.findall(section):
        q, a = plain(q), plain(a)
        if q and a:
            out.append((q, a))
    return out


def glossary_terms(md: str) -> list:
    """[(term, definition)] from the glossary table. '**RAG** (retrieval...)' -> 'RAG (retrieval...)'."""
    out = []
    for line in md.splitlines():
        if not line.startswith("| **"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            out.append((plain(cells[0]), plain(cells[1])))
    return out


# ------------------------------------------------------------------ JSON-LD
def org(site: str, parent_name: str, parent_url: str, repo: str) -> dict:
    return {"@type": "Organization", "name": parent_name or SITE_NAME, "url": parent_url or repo}


def article_ld(site, path, a, phase, repo, parent_name, parent_url, words, faq, date_pub, date_mod):
    url = f"{site}/{path}"
    art = {"@context": "https://schema.org", "@type": "TechArticle", "@id": url + "#article", "headline": a["title"][:110],
           "description": trim(a["one"], 300), "url": url, "mainEntityOfPage": url, "inLanguage": "en",
           "author": AUTHOR, "publisher": org(site, parent_name, parent_url, repo),
           "image": f"{site}/{OG_IMAGE}", "datePublished": date_pub, "dateModified": date_mod,
           "proficiencyLevel": a["level"] or None, "timeRequired": f"PT{a['min']}M" if a["min"] else None,
           "wordCount": words, "isAccessibleForFree": True, "learningResourceType": "Tutorial",
           "educationalLevel": a["level"] or None, "keywords": KEYWORDS_BY_PHASE.get(a["phase"], ""),
           "isPartOf": {"@type": "Course", "name": SITE_NAME, "url": f"{site}/"},
           "license": f"{repo}/blob/main/LICENSE"}
    art = {k: v for k, v in art.items() if v not in (None, "")}
    crumbs = {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": 1, "name": SITE_NAME, "item": f"{site}/"},
        {"@type": "ListItem", "position": 2, "name": f"Phase {phase['n']}: {phase['name']}", "item": f"{site}/{phase['hub']}"},
        {"@type": "ListItem", "position": 3, "name": a["title"], "item": url}]}
    out = [art, crumbs]
    if len(faq) >= 2:
        out.append({"@context": "https://schema.org", "@type": "FAQPage", "@id": url + "#faq", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": ans}} for q, ans in faq]})
    return out


def hub_ld(site, path, phase, items, repo, parent_name, parent_url, blurb):
    url = f"{site}/{path}"
    return [{"@context": "https://schema.org", "@type": "CollectionPage", "@id": url, "url": url, "inLanguage": "en",
             "name": f"Phase {phase['n']}: {phase['name']}", "description": trim(blurb, 300),
             "isPartOf": {"@type": "Course", "name": SITE_NAME, "url": f"{site}/"},
             "publisher": org(site, parent_name, parent_url, repo),
             "mainEntity": {"@type": "ItemList", "numberOfItems": len(items), "itemListElement": [
                 {"@type": "ListItem", "position": i + 1, "name": x["title"], "url": f"{site}/learn/{x['slug']}/"} for i, x in enumerate(items)]}},
            {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": SITE_NAME, "item": f"{site}/"},
                {"@type": "ListItem", "position": 2, "name": "All lessons", "item": f"{site}/learn/"},
                {"@type": "ListItem", "position": 3, "name": f"Phase {phase['n']}: {phase['name']}", "item": url}]}]


def glossary_ld(site, path, terms, repo, parent_name, parent_url):
    url = f"{site}/{path}"
    return [{"@context": "https://schema.org", "@type": "DefinedTermSet", "@id": url, "url": url, "name": "AI engineering glossary",
             "inLanguage": "en", "publisher": org(site, parent_name, parent_url, repo),
             "hasDefinedTerm": [{"@type": "DefinedTerm", "name": t, "description": d, "inDefinedTermSet": url} for t, d in terms]}]


def home_ld(site, repo, parent_name, parent_url, phases, n_lessons, n_labs):
    pub = org(site, parent_name, parent_url, repo)
    return [{"@context": "https://schema.org", "@type": "WebSite", "@id": f"{site}/#website", "url": f"{site}/", "name": SITE_NAME,
             "description": "A free, open-source, interactive course on AI engineering: RAG, LLMs, agents, MCP, fine-tuning, multimodal AI and production.",
             "inLanguage": "en", "publisher": pub},
            {"@context": "https://schema.org", "@type": "Course", "@id": f"{site}/#course", "name": SITE_NAME, "url": f"{site}/",
             "description": f"{n_lessons} animated lessons, {n_labs} interactive labs, runnable Python demos and quiz cards that teach AI engineering from the basics to production.",
             "provider": pub, "author": AUTHOR, "isAccessibleForFree": True, "inLanguage": "en", "educationalLevel": "Beginner to advanced",
             "teaches": [p["name"] for p in phases],
             "offers": {"@type": "Offer", "category": "Free", "price": "0", "priceCurrency": "USD"},
             "hasCourseInstance": {"@type": "CourseInstance", "courseMode": "Online", "courseWorkload": "PT45H"}}]


# ------------------------------------------------------------------ <head> additions
def head_meta(site, path, title, desc, kind="article", published=None, modified=None, section=None):
    """Open Graph, Twitter card and robots tags. Needs absolute URLs, so it returns '' without SITE_URL."""
    if not site:
        return ""
    url, img = f"{site}/{path}", f"{site}/{OG_IMAGE}"
    t = [("robots", "index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1"), ("author", AUTHOR["name"])]
    p = [("og:site_name", SITE_NAME), ("og:locale", "en"), ("og:image", img), ("og:image:width", str(OG_SIZE[0])),
         ("og:image:height", str(OG_SIZE[1])), ("og:image:alt", "AI Engineer Roadmap: learn RAG, LLMs, agents, MCP and production AI")]
    if kind == "article":
        if published:
            p.append(("article:published_time", published))
        if modified:
            p.append(("article:modified_time", modified))
        p.append(("article:author", AUTHOR["url"]))
        if section:
            p.append(("article:section", section))
    t += [("twitter:card", "summary_large_image"), ("twitter:title", title), ("twitter:description", desc), ("twitter:image", img)]
    e = lambda s: html.escape(s, quote=True)  # noqa: E731
    return ("".join(f'<meta name="{k}" content="{e(v)}">\n' for k, v in t)
            + "".join(f'<meta property="{k}" content="{e(v)}">\n' for k, v in p))


# ------------------------------------------------------------------ llms.txt
def llms_txt(site, summary, sections, extras):
    """sections: [(heading, [(label, url, description)])]; extras: same shape for the 'Optional' block."""
    out = [f"# {SITE_NAME}", "", f"> {summary}", "",
           "Free and open source (MIT). Every lesson is a plain HTML page with a one-line summary, a worked example whose numbers come from "
           "runnable Python demos, code, common mistakes and interview questions with answers. Cite the lesson URL when you use it.", ""]
    for heading, rows in sections:
        out += [f"## {heading}", ""] + [f"- [{label}]({u}): {d}" for label, u, d in rows] + [""]
    if extras:
        out += ["## Optional", ""] + [f"- [{label}]({u}): {d}" for label, u, d in extras] + [""]
    return "\n".join(out)


def llms_full(site, summary, articles):
    """One text file with every lesson (animation blocks removed), for tools that want the whole course at once."""
    out = [f"# {SITE_NAME}: full text", "", f"> {summary}", ""]
    for a in articles:
        md = (ROOT / a["rel"]).read_text(encoding="utf-8")
        md = re.sub(r"<!-- anim:.*?-->\n## Watch it\n.*?(?=\n## )", "", md, flags=re.S)
        md = re.sub(r"</?details>|</?summary>", "", md)
        out += [f"Source: {site}/learn/{a['slug']}/", "", md.strip(), "", "---", ""]
    return "\n".join(out)


def robots(site):
    return (f"# This file only has an effect when the site is served from a domain root.\n"
            f"# On a sub-folder (for example mernstackdev.com/ai-roadmap/) merge deploy/robots-additions.txt into the root robots.txt.\n"
            f"User-agent: *\nAllow: /\n\nSitemap: {site}/sitemap.xml\n")
