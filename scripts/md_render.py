#!/usr/bin/env python3
"""Topic Markdown -> safe HTML for the Pages site (standard library only).

Handles exactly what the topic pages use: headings, paragraphs, nested lists, tables, fenced code,
mermaid diagrams (drawn as plain HTML boxes, no CDN), blockquotes, <details> answers, images and links.

    python scripts/md_render.py 01-rag-foundations/topics/11-cosine-similarity.md   # prints the HTML
"""
import html
import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/techsaurabh-394/ai-engineer-roadmap"
MATH = re.compile(r"[√Σ‖·×^²≈]|\w\s*=\s*\S|/")


class Ctx:
    """Where we are (for resolving relative links) and which topic files have a lesson page."""

    def __init__(self, path: str, lesson_of: dict):
        self.path, self.lesson_of = path, lesson_of
        self.formulas = []


def link_target(href: str, ctx: Ctx) -> str:
    if re.match(r"^(https?:|mailto:|#)", href):
        return href
    base, _, frag = href.partition("#")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(ctx.path), base))
    if target in ctx.lesson_of:
        return "#" + ctx.lesson_of[target]
    if (ROOT / target).is_dir():
        return f"{REPO}/tree/main/{target}"
    if (ROOT / target).exists():
        return f"{REPO}/blob/main/{target}"
    return REPO


def inline(text: str, ctx: Ctx) -> str:
    """Inline Markdown -> HTML. Code spans and links are stashed first so their text is not re-processed."""
    stash = []

    def keep(s):
        stash.append(s)
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: keep(f"<code>{html.escape(m.group(1), quote=False)}</code>"), text)

    text = re.sub(r"<details>\s*<summary>(.*?)</summary>",
                  lambda m: keep(f'<details class="qa"><summary>{html.escape(m.group(1), quote=False)}</summary><div>'), text)
    text = text.replace("</details>", keep("</div></details>"))
    def link(m):
        href = link_target(m.group(2).strip(), ctx)
        ext = href.startswith("http")
        rel = ' rel="noopener" target="_blank"' if ext else ""
        return keep(f'<a href="{html.escape(href)}"{rel}>{html.escape(m.group(1), quote=False)}</a>')

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = re.sub(r"(?<![\"'>(])(https?://[^\s<)]+[^\s<).,;])",
                  lambda m: keep(f'<a href="{html.escape(m.group(1))}" rel="noopener" target="_blank">{html.escape(m.group(1), quote=False)}</a>'), text)
    text = html.escape(text, quote=False)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    for _ in range(3):  # a link label may itself hold a stashed code span
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def is_formula(c: str) -> bool:
    """A code span that states a rule (cos = a·b / ..., softmax(x) = ...) rather than assigning a sample value."""
    if "=" not in c or len(c) > 90 or re.match(r"^\w+\s*=\s*[\[(\d-]", c):
        return False
    return bool(re.search(r"[√Σ‖·×/²^∑]|\w\(.*\)\s*=|=\s*\w+\(", c)) and c.count("=") <= 2


# ------------------------------------------------------------------ mermaid -> HTML
NODE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(?:(\[\(|\[\[|\(\(|\[|\(|\{)(.*?)(\)\]|\]\]|\)\)|\]|\)|\}))?\s*$")


def _node(tok: str, names: dict) -> str:
    m = NODE.match(tok.strip())
    if not m:
        return tok.strip()
    nid, _, label, _ = m.groups()
    if label:
        names[nid] = label.strip().strip('"')
    return nid


def mermaid_html(src: str) -> str:
    lines = [l.rstrip() for l in src.strip("\n").split("\n")]
    head = lines[0].strip()
    esc = lambda s: html.escape(s, quote=False)  # noqa: E731
    if head.startswith("sequenceDiagram"):
        out = ['<ol class="mm-seq">']
        for l in lines[1:]:
            m = re.match(r"\s*(\w+)\s*(-->>|->>|-->|->)\s*(\w+)\s*:\s*(.*)", l)
            if m:
                a, arrow, b, msg = m.groups()
                out.append(f'<li><span class="who">{esc(a)}</span><span class="arr">{"⇢" if "--" in arrow else "→"}</span>'
                           f'<span class="who">{esc(b)}</span><span class="msg">{esc(msg)}</span></li>')
        return "".join(out) + "</ol>"
    names: dict = {}
    groups = [("", [])]
    for l in lines[1:]:
        s = l.strip()
        m = re.match(r"subgraph\s+(.*)", s)
        if m:
            groups.append((m.group(1).strip().strip('"'), []))
            continue
        if s == "end":
            groups.append(("", []))
            continue
        if not s or s.startswith("%%") or s.startswith(("style ", "classDef ", "class ")):
            continue
        parts = re.split(r"\s*(?:--\s*([^>]*?)\s*-->|-->\|([^|]*)\||-->|==>)\s*", s)
        # re.split with 2 capture groups: [n0, lab1, lab2, n1, lab1, lab2, n2 ...]
        nodes, labels = [], []
        i = 0
        while i < len(parts):
            nodes.append(_node(parts[i], names))
            if i + 2 < len(parts):
                labels.append((parts[i + 1] or parts[i + 2] or "").strip())
            i += 3
        for a, b, lab in zip(nodes, nodes[1:], labels):
            groups[-1][1].append((a, lab, b))
    out = ['<div class="mm-flow">']
    for title, edges in groups:
        if not edges:
            continue
        out.append('<div class="mm-group">')
        if title:
            out.append(f'<p class="mm-title">{esc(title)}</p>')
        out.append("<ul>")
        chains: list = []  # [(first box, [(label, next box), ...])]: consecutive edges share one line
        for a, lab, b in edges:
            if chains and chains[-1][1][-1][1] == a:
                chains[-1][1].append((lab, b))
            else:
                chains.append((a, [(lab, b)]))
        for first, rest in chains:
            row = f'<span class="box">{esc(names.get(first, first))}</span>'
            for lab, b in rest:
                arrow = f'<span class="arr">→{(" <i>" + esc(lab) + "</i> →") if lab else ""}</span>'
                row += f'{arrow}<span class="box">{esc(names.get(b, b))}</span>'
            out.append(f"<li>{row}</li>")
        out.append("</ul></div>")
    return "".join(out) + "</div>"


# ------------------------------------------------------------------ blocks
def render(md: str, ctx: Ctx) -> str:
    lines = md.split("\n")
    out, i, section = [], 0, ""
    para: list = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para), ctx)}</p>")
            para.clear()

    def table(rows):
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        head, body = cells[0], cells[2:]
        t = "<div class=\"tbl\"><table><thead><tr>" + "".join(f"<th>{inline(c, ctx)}</th>" for c in head) + "</tr></thead><tbody>"
        for r in body:
            t += "<tr>" + "".join(f"<td>{inline(c, ctx)}</td>" for c in r) + "</tr>"
        return t + "</tbody></table></div>"

    def list_block(i):
        """Nested -/1. lists by indentation. Returns (html, next_i)."""
        items = []  # (indent, ordered, text)
        while i < len(lines):
            m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
            if m:
                items.append([len(m.group(1)), m.group(2)[0].isdigit(), m.group(3)])
                i += 1
            elif lines[i].strip() and re.match(r"^\s{2,}\S", lines[i]) and items:
                items[-1][2] += " " + lines[i].strip()
                i += 1
            else:
                break

        def build(pos, indent):
            ordered = items[pos][1]
            tag = "ol" if ordered else "ul"
            s = f"<{tag}>"
            while pos < len(items) and items[pos][0] >= indent:
                if items[pos][0] > indent:
                    pos_sub, sub = build(pos, items[pos][0])
                    s = s[: -len("</li>")] + sub + "</li>" if s.endswith("</li>") else s + sub
                    pos = pos_sub
                    continue
                s += f"<li>{inline(items[pos][2], ctx)}</li>"
                pos += 1
            return pos, s + f"</{tag}>"

        _, h = build(0, items[0][0])
        return h, i

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            lang = line[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            code = "\n".join(buf)
            if lang == "mermaid":
                out.append(f'<figure class="diagram"><figcaption>Diagram</figcaption>{mermaid_html(code)}</figure>')
            else:
                label = {"python": "Python", "text": "Diagram / text", "bash": "Shell", "json": "JSON", "": "Code"}.get(lang, lang or "Code")
                out.append(f'<div class="code" data-lang="{html.escape(lang or "text")}"><div class="code-head"><span>{html.escape(label)}</span>'
                           f'<button type="button" class="copy" aria-label="Copy this code">Copy</button></div>'
                           f'<pre tabindex="0"><code>{html.escape(code, quote=False)}</code></pre></div>')
            continue
        if line.strip().startswith("<!--"):
            flush()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            lvl, title = len(m.group(1)), m.group(2).strip()
            if lvl == 1:
                i += 1
                continue
            section = title
            sid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{lvl + 1} id="sec-{sid}">{inline(title, ctx)}</h{lvl + 1}>')
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        if line.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            if buf and buf[0].startswith("**Level:**"):
                continue
            out.append(f"<blockquote>{inline(' '.join(buf), ctx)}</blockquote>")
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            flush()
            h, i = list_block(i)
            out.append(h)
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            flush()
            i += 1
            continue  # the animation is already the hero of the page
        if line.strip().startswith("<details"):
            flush()
            buf = []
            while i < len(lines):
                buf.append(lines[i])
                i += 1
                if "</details>" in lines[i - 1]:
                    break
            blob = " ".join(b.strip() for b in buf)
            sm = re.search(r"<summary>(.*?)</summary>", blob)
            summ = sm.group(1) if sm else "Answer"
            body = re.sub(r"</?details>|<summary>.*?</summary>", "", blob).strip()
            out.append(f'<details class="qa"><summary>{inline(re.sub(chr(60) + "[^>]+>", "", summ), ctx)}</summary><div>{inline(body, ctx)}</div></details>')
            continue
        if line.startswith("<summary>") or line.strip() in ("</details>", "<details>"):
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,})$", line.strip()):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def topic_html(path: str, lesson_of: dict) -> str:
    """The whole lesson as an <article>: contents list, 'Key formulas' box, then the page body."""
    text = (ROOT / path).read_text(encoding="utf-8")
    text = re.sub(r"<!-- anim:.*?-->\n## Watch it\n.*?(?=\n## )", "", text, flags=re.S)  # the animation is the hero
    ctx = Ctx(path, lesson_of)
    body = render(text, ctx)
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    prose = re.sub(r"### Worked example.*?(?=\n##)", "", prose, flags=re.S)
    for c in re.findall(r"`([^`\n]+)`", prose.split("## Interview questions")[0]):
        if is_formula(c) and c not in ctx.formulas:
            ctx.formulas.append(c)
    # promote Prev/Next footer line into nothing: the site has its own lesson navigation
    body = re.sub(r"<p><strong>Prev:</strong>.*?</p>\s*$", "", body, flags=re.S)
    toc = re.findall(r'<h2 id="(sec-[^"]+)">(.*?)</h2>', body)
    nav = ""
    if toc:
        nav = '<nav class="toc" aria-label="On this page"><p class="k">On this page</p><ol>' + "".join(
            f'<li><a href="#{a}" data-jump="{a}">{re.sub(r"<[^>]+>", "", b)}</a></li>' for a, b in toc) + "</ol></nav>"
    formulas = ""
    if ctx.formulas:
        formulas = ('<aside class="formulas"><p class="k">Key formulas</p><ul>'
                    + "".join(f"<li><code>{html.escape(f, quote=False)}</code></li>" for f in ctx.formulas[:6]) + "</ul></aside>")
    return f'<article class="reading">{nav}{formulas}{body}</article>'


if __name__ == "__main__":
    print(topic_html(sys.argv[1], {}))
