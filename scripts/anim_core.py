"""Shared engine for the animated explainers (standard library only).

Specs call P() (pipeline), L() (loop) or B() (bar chart) to register an
animation. gen_animations.py renders the registry to assets/anim/*.svg,
docs/animations.md and the manifest used by scripts/build_site.py.
"""
from __future__ import annotations

import glob
import math
import pathlib
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parent.parent
BG, CARD, LINE, TEXT, MUTED = "#0D1117", "#161B22", "#30363D", "#E6EDF3", "#9DA7B3"
COLORS = {"teal": "#2DD4BF", "blue": "#6EA8FE", "amber": "#F5B042", "pink": "#F472B6"}
FONT = "-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "SFMono-Regular,Menlo,Consolas,monospace"
SLOT = 2.4          # seconds per step (pipelines and loops)
BAR_SLOT = 3.0      # seconds per frame (bar charts)

REGISTRY: dict[str, dict] = {}


# ----------------------------------------------------------------- topics
def topic_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for f in sorted(glob.glob(str(ROOT / "0[1-7]*/topics/*.md"))):
        rel = pathlib.Path(f).relative_to(ROOT).as_posix()
        stem = rel.rsplit("/", 1)[1][:-3]
        if stem in idx:
            idx[stem] = ""  # ambiguous
        else:
            idx[stem] = rel
    return idx


def _resolve(topic, name):
    idx = topic_index()
    if topic is None:
        topic = name if name in idx else ()
    if isinstance(topic, str):
        topic = (topic,)
    out = []
    for t in topic:
        if "/" in t:
            rel = t if t.endswith(".md") else t + ".md"
        else:
            rel = idx.get(t, "")
            if not rel:
                raise SystemExit(f"topic {t!r} not found or ambiguous (animation {name})")
        if not (ROOT / rel).exists():
            raise SystemExit(f"topic file missing: {rel}")
        out.append(rel)
    return out


def _parse(text: str, cols: int):
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != cols:
            raise SystemExit(f"bad step (need {cols} fields): {line!r}")
        rows.append(parts)
    return rows


def _register(name, **kw):
    if name in REGISTRY:
        raise SystemExit(f"duplicate animation name {name}")
    REGISTRY[name] = dict(name=name, **kw)


def P(name, title, subtitle, steps="", color="teal", topic=None, custom=False, **_):
    if custom:
        _register(name, kind="custom", title=title, subtitle=subtitle, color=color, topics=_resolve(topic, name))
        return
    rows = _parse(steps, 4)
    _register(name, kind="pipeline", title=title, subtitle=subtitle, color=color, topics=_resolve(topic, name),
              steps=[dict(label=r[0], sub=r[1], c1=r[2], c2=r[3]) for r in rows])


def L(name, title, subtitle, steps="", color="teal", topic=None, **_):
    rows = _parse(steps, 3)
    _register(name, kind="loop", title=title, subtitle=subtitle, color=color, topics=_resolve(topic, name),
              steps=[dict(label=r[0], sub="", c1=r[1], c2=r[2]) for r in rows])


def B(name, title, subtitle, labels, frames, color="teal", topic=None):
    """Bar chart. frames: (frame name, probs, keep mask or None, caption 1, caption 2)."""
    _register(name, kind="bars", title=title, subtitle=subtitle, color=color, topics=_resolve(topic, name),
              labels=labels, frames=frames,
              steps=[dict(label=f[0], sub="", c1=f[3], c2=f[4]) for f in frames])


# ----------------------------------------------------------- svg helpers
def head(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n<title id="t">{escape(title)}</title>\n'
            f'<desc id="d">{escape(desc)}</desc>\n'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" rx="16" fill="{BG}"/>\n')


def txt(x, y, s, size=15, fill=TEXT, weight=400, anchor="middle", mono=False):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{fam}>{escape(s)}</text>\n')


def _f(v):
    return f"{v:.4g}"


def slot_anim(attr, on, off, i, n, dur, fade=0.012):
    a, b = i / n, (i + 1) / n
    pts = {}
    for k in [0, max(a - fade, 0), a, b, min(b + fade, 1), 1]:
        # decide on/off from the exact time, BEFORE rounding the key
        pts[round(k, 4)] = on if a - 1e-9 <= k <= b + 1e-9 else off
    kt = sorted(pts)
    return (f'<animate attributeName="{attr}" dur="{dur}s" repeatCount="indefinite" '
            f'keyTimes="{";".join(map(str, kt))}" values="{";".join(pts[k] for k in kt)}"/>')


def hold_anim(attr, values, n, dur, fade=0.18):
    """Smoothly move `attr` through one value per slot (hold, then glide at the slot start)."""
    kt, vs = [0.0], [values[-1]]
    for i, v in enumerate(values):
        a, b = i / n, (i + 1) / n
        kt += [a + (b - a) * fade, b]
        vs += [v, v]
    kt[-1] = 1.0
    kt = [round(k, 4) for k in kt]
    return (f'<animate attributeName="{attr}" dur="{dur}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{";".join(map(str, kt))}" values="{";".join(vs)}"/>')


def caption_block(cx, y, lines, i, n, dur, size=15):
    out = ""
    for j, line in enumerate(lines):
        col, wt = (TEXT, 700) if j == 0 else (MUTED, 400)
        out += (f'<text x="{cx}" y="{y + j * 24}" font-size="{size}" fill="{col}" font-weight="{wt}" '
                f'text-anchor="middle" opacity="0">{escape(line)}{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')
    return out


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def packet(px, py, n, dur, r=9, color="#F5B042"):
    kt, xs, ys = [0.0], [px[0]], [py[0]]
    for i in range(n):
        a, b = i / n, (i + 1) / n
        if i > 0:
            kt.append(a); xs.append(px[i - 1]); ys.append(py[i - 1])
            kt.append(a + (b - a) * 0.35)
        else:
            kt.append(a + (b - a) * 0.05)
        xs.append(px[i]); ys.append(py[i])
    kt.append(1.0); xs.append(px[-1]); ys.append(py[-1])
    last, fixed = -1, []
    for k in kt:
        k = max(k, last); fixed.append(round(min(k, 1), 4)); last = k
    ktxt = ";".join(map(str, fixed))
    fx = ";".join(_f(v) for v in xs)
    fy = ";".join(_f(v) for v in ys)
    return (f'<circle r="{r}" fill="{color}" cx="{_f(xs[0])}" cy="{_f(ys[0])}">'
            f'<animate attributeName="cx" dur="{dur}s" repeatCount="indefinite" keyTimes="{ktxt}" values="{fx}"/>'
            f'<animate attributeName="cy" dur="{dur}s" repeatCount="indefinite" keyTimes="{ktxt}" values="{fy}"/>'
            f'</circle>\n')


def step_counter(x, y, i, n, dur):
    return (f'<text x="{x}" y="{y}" font-size="12" fill="{MUTED}" text-anchor="end" opacity="0">Step {i + 1} of {n}'
            f'{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')


# ----------------------------------------------------------------- lint
def lint(a):
    errs = []
    if a["kind"] in ("pipeline", "loop", "bars"):
        n = len(a["steps"])
        if not 3 <= n <= 6:
            errs.append(f"{a['name']}: {n} steps (need 3 to 6)")
    if a["kind"] == "pipeline":
        n = len(a["steps"]); bw = 118 if n >= 6 else 128
        for s in a["steps"]:
            if len(s["label"]) * 9 > bw - 14: errs.append(f"{a['name']}: label too long {s['label']!r}")
            if len(s["sub"]) * 6.8 > bw - 14: errs.append(f"{a['name']}: sub too long {s['sub']!r}")
            if len(s["c1"]) > 96: errs.append(f"{a['name']}: c1 too long ({len(s['c1'])}) {s['c1']!r}")
            if len(s["c2"]) > 108: errs.append(f"{a['name']}: c2 too long ({len(s['c2'])}) {s['c2']!r}")
    if a["kind"] == "loop":
        for s in a["steps"]:
            if len(s["label"]) > 14: errs.append(f"{a['name']}: loop label too long {s['label']!r}")
    if a["kind"] == "bars":
        for s in a["steps"]:
            if len(s["c1"]) > 96 or len(s["c2"]) > 108: errs.append(f"{a['name']}: caption too long {s['c1']!r}")
    return errs


# --------------------------------------------------------------- renderers
def render_pipeline(a):
    steps, n = a["steps"], len(a["steps"])
    color = COLORS[a["color"]]
    W, H = 880, 330
    dur = round(n * SLOT, 2)
    bw = 118 if n >= 6 else 128
    gap = (W - 60 - n * bw) / max(n - 1, 1)
    xs = [30 + i * (bw + gap) for i in range(n)]
    cy = 150
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    for i in range(n - 1):
        s += (f'<line x1="{_f(xs[i] + bw + 2)}" y1="{cy}" x2="{_f(xs[i + 1] - 4)}" y2="{cy}" stroke="{LINE}" '
              f'stroke-width="2.5" marker-end="url(#ah)"/>\n')
    for i, st in enumerate(steps):
        x = xs[i]
        s += (f'<rect x="{_f(x)}" y="{cy - 46}" width="{bw}" height="92" rx="12" fill="{CARD}" stroke="{LINE}" stroke-width="2">'
              f'{slot_anim("stroke", color, LINE, i, n, dur)}</rect>\n')
        s += f'<circle cx="{_f(x + 18)}" cy="{cy - 46}" r="13" fill="{color}"/>' + txt(_f(x + 18), cy - 41, str(i + 1), 14, BG, 700)
        s += txt(_f(x + bw / 2), cy - 4, st["label"], 16, TEXT, 700) + txt(_f(x + bw / 2), cy + 20, st["sub"], 12.5, MUTED)
    s += packet([x + bw / 2 for x in xs], [cy - 62] * n, n, dur)
    s += f'<rect x="30" y="226" width="{W - 60}" height="84" rx="12" fill="{CARD}" stroke="{LINE}"/>\n'
    for i, st in enumerate(steps):
        s += caption_block(W / 2, 258, [st["c1"], st["c2"]], i, n, dur) + step_counter(W - 44, 300, i, n, dur)
    return s + "</svg>\n", dur, [round((i + 0.5) * SLOT, 2) for i in range(n)]


def render_loop(a):
    steps, n = a["steps"], len(a["steps"])
    color = COLORS[a["color"]]
    W, H = 880, 420
    cx, cy, rx, ry = 440, 240, 250, 105
    dur = round(n * SLOT, 2)
    pts = [(cx + rx * math.cos(-math.pi / 2 + 2 * math.pi * i / n), cy + ry * math.sin(-math.pi / 2 + 2 * math.pi * i / n)) for i in range(n)]
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        s += f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{LINE}" stroke-width="2.5" marker-end="url(#ah)"/>\n'
    bw, bh = 150, 50
    for i, st in enumerate(steps):
        x, y = pts[i]
        s += (f'<rect x="{x - bw / 2:.0f}" y="{y - bh / 2:.0f}" width="{bw}" height="{bh}" rx="12" fill="{CARD}" stroke="{LINE}" stroke-width="2">'
              f'{slot_anim("stroke", color, LINE, i, n, dur)}</rect>\n')
        s += txt(round(x), round(y + 6), st["label"], 16, TEXT, 700)
    s += packet([p[0] for p in pts], [p[1] - 36 for p in pts], n, dur)
    for i, st in enumerate(steps):
        lines = [(wrap(st["c1"], 40), 700, TEXT), (wrap(st["c2"], 44), 400, MUTED)]
        total = sum(len(l[0]) for l in lines)
        y = cy + 5 - (total - 1) * 11
        for ls, wt, col in lines:
            for ln in ls:
                s += (f'<text x="{cx}" y="{y}" font-size="15" fill="{col}" font-weight="{wt}" text-anchor="middle" opacity="0">'
                      f'{escape(ln)}{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')
                y += 22
    return s + "</svg>\n", dur, [round((i + 0.5) * SLOT, 2) for i in range(n)]


def render_bars(a):
    frames, labels, n, m = a["frames"], a["labels"], len(a["frames"]), len(a["labels"])
    color = COLORS[a["color"]]
    W, H = 880, 360
    dur = round(n * BAR_SLOT, 2)
    bw, gap = 90, 36
    x0 = (W - (m * bw + (m - 1) * gap)) / 2
    base, scale = 238, 150
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    s += f'<line x1="{_f(x0 - 20)}" y1="{base}" x2="{_f(x0 + m * bw + (m - 1) * gap + 20)}" y2="{base}" stroke="{LINE}" stroke-width="2"/>\n'
    for j, lab in enumerate(labels):
        x = x0 + j * (bw + gap)
        hs = [fr[1][j] * scale for fr in frames]
        ys = [base - h for h in hs]
        fills = [color if (fr[2] is None or fr[2][j]) else LINE for fr in frames]
        s += (f'<rect x="{_f(x)}" y="{_f(ys[-1])}" width="{bw}" height="{_f(max(hs[-1], 1))}" rx="6" fill="{fills[-1]}">'
              + hold_anim("height", [_f(max(h, 1)) for h in hs], n, dur)
              + hold_anim("y", [_f(y) for y in ys], n, dur)
              + hold_anim("fill", fills, n, dur) + '</rect>\n')
        s += txt(_f(x + bw / 2), base + 20, lab, 14, TEXT, 700)
        for i, fr in enumerate(frames):
            keep = fr[2] is None or fr[2][j]
            s += (f'<text x="{_f(x + bw / 2)}" y="{_f(base - hs[i] - 8)}" font-size="14" fill="{TEXT if keep else MUTED}" '
                  f'font-weight="700" text-anchor="middle" opacity="0">{fr[1][j]:.2f}'
                  f'{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')
    for i, fr in enumerate(frames):
        s += (f'<text x="40" y="100" font-size="18" fill="{color}" font-weight="700" opacity="0">{escape(fr[0])}'
              f'{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')
    s += f'<rect x="30" y="272" width="{W - 60}" height="78" rx="12" fill="{CARD}" stroke="{LINE}"/>\n'
    for i, fr in enumerate(frames):
        s += caption_block(W / 2, 303, [fr[3], fr[4]], i, n, dur) + step_counter(W - 44, 342, i, n, dur)
    return s + "</svg>\n", dur, [round((i + 0.5) * BAR_SLOT, 2) for i in range(n)]


# ---------------------------------------------------------------- custom
def _chunking(a):
    W, H, dur = 880, 360, 12
    words = "Multi-stage builds compile code in one stage and copy only the final artifact into a small runtime image".split()[:15]
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    bw, gap, x0 = 52, 3, 30
    wx = lambda i: x0 + i * (bw + gap)
    s += txt(30, 100, "Document", 13, MUTED, 700, "start")
    for i, w in enumerate(words):
        s += f'<rect x="{wx(i)}" y="108" width="{bw}" height="30" rx="6" fill="{CARD}" stroke="{LINE}"/>' + txt(wx(i) + bw / 2, 128, w[:7], 11, TEXT)
    chunks = [(0, 6, COLORS["teal"], "Chunk 1"), (5, 11, COLORS["blue"], "Chunk 2"), (10, 15, COLORS["pink"], "Chunk 3")]
    for k, (lo, hi, col, lab) in enumerate(chunks):
        y, t0 = 180 + k * 52, 0.12 + k * 0.22
        s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
              f'keyTimes="0;{t0:.2f};{t0 + 0.04:.2f};0.9;0.96;1" values="0;0;1;1;0;0"/>')
        s += txt(30, y - 8, f"{lab}  (words {lo + 1} to {hi})", 13, col, 700, "start")
        for i in range(lo, hi):
            ov = k > 0 and i < chunks[k - 1][1]
            s += (f'<rect x="{wx(i)}" y="{y}" width="{bw}" height="30" rx="6" fill="{CARD}" stroke="{COLORS["amber"] if ov else col}" '
                  f'stroke-width="{3 if ov else 1.5}"/>' + txt(wx(i) + bw / 2, y + 20, words[i][:7], 11, TEXT))
        s += "</g>\n"
    s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.66;0.7;0.9;0.96;1" values="0;0;1;1;0;0"/>'
          + txt(W / 2, 346, "Amber outline = overlap: the shared words keep context across the cut.", 14, COLORS["amber"], 700) + "</g>\n")
    steps = [
        dict(label="The text", sub="", c1="1. Start with one long text", c2="Cutting it badly would split ideas in half."),
        dict(label="Chunk 1", sub="", c1="2. Chunk 1 takes the first 6 words", c2="A small piece the search can match on its own."),
        dict(label="Chunk 2", sub="", c1="3. Chunk 2 starts 1 word early, so it overlaps", c2="The shared word is the amber outline."),
        dict(label="Chunk 3", sub="", c1="4. Chunk 3 overlaps chunk 2 the same way", c2="Every cut point sits inside two chunks."),
        dict(label="Why", sub="", c1="5. The overlap keeps context across each cut", c2="The price: a few more chunks to store and search."),
    ]
    return s + "</svg>\n", dur, [0.6, 2.5, 5.0, 7.8, 10.0], steps


def _cosine(a):
    W, H, n, dur = 880, 380, 5, 15
    ox, oy, L_ = 260, 300, 190
    angles = [10, 45, 90, 135, 180]
    notes = ["almost the same meaning", "related", "unrelated", "leaning opposite", "opposite meaning"]
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    s += f'<line x1="40" y1="{oy}" x2="480" y2="{oy}" stroke="{LINE}" stroke-dasharray="4 4"/>\n'
    s += f'<line x1="{ox}" y1="{oy}" x2="{ox + L_}" y2="{oy}" stroke="{COLORS["teal"]}" stroke-width="4" stroke-linecap="round"/>\n'
    s += txt(ox + L_ + 8, oy - 8, "query", 14, COLORS["teal"], 700, "start")
    s += (f'<g><line x1="{ox}" y1="{oy}" x2="{ox + L_}" y2="{oy}" stroke="{COLORS["amber"]}" stroke-width="4" stroke-linecap="round"/>'
          + txt(ox + L_ + 8, oy - 8, "chunk", 14, COLORS["amber"], 700, "start")
          + hold_anim("transform", [f"{-x} {ox} {oy}" for x in angles], n, dur).replace(
              'attributeName="transform"', 'attributeName="transform" type="rotate"').replace("<animate ", "<animateTransform ") .replace("</animate>", "</animateTransform>")
          + '</g>\n')
    s += f'<rect x="540" y="110" width="310" height="210" rx="14" fill="{CARD}" stroke="{LINE}"/>\n'
    s += txt(695, 142, "score = cos(angle)", 16, TEXT, 700)
    steps = []
    for i, (ang, note) in enumerate(zip(angles, notes)):
        c = math.cos(math.radians(ang))
        s += (f'<g opacity="0">{slot_anim("opacity", "1", "0", i, n, dur, 0.02)}'
              + txt(695, 205, f"{c:+.2f}", 46, COLORS["amber"], 700) + txt(695, 238, f"angle {ang} degrees", 14, MUTED)
              + txt(695, 276, note, 17, TEXT, 700) + "</g>\n")
        steps.append(dict(label=f"{ang} degrees", sub="", c1=f"{i + 1}. Angle {ang} degrees: score {c:+.2f}", c2=note[0].upper() + note[1:] + "."))
    s += txt(695, 302, "1 = same direction, 0 = unrelated, -1 = opposite", 12.5, MUTED)
    return s + "</svg>\n", dur, [round((i + 0.5) * dur / n, 2) for i in range(n)], steps


def _attention(a):
    W, H, dur = 880, 380, 10
    toks = ["The", "cat", "sat", "on", "the", "mat"]
    wts = [0.05, 0.45, 0.15, 0.05, 0.05, 0.25]
    s = head(W, H, a["title"], a["subtitle"])
    s += txt(W / 2, 40, a["title"], 22, TEXT, 700) + txt(W / 2, 64, a["subtitle"], 14, MUTED)
    xs = [90 + i * 140 for i in range(6)]
    qx = xs[2]
    for i, (x, t) in enumerate(zip(xs, toks)):
        q = i == 2
        s += (f'<rect x="{x - 48}" y="256" width="96" height="44" rx="10" fill="{CARD}" stroke="{COLORS["amber"] if q else LINE}" '
              f'stroke-width="{3 if q else 2}"/>' + txt(x, 285, t, 18, TEXT, 700))
    s += txt(qx, 324, "query (asking)", 13, COLORS["amber"], 700)
    fade = lambda t0: (f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
                       f'keyTimes="0;{t0};{t0 + 0.05:.2f};0.9;0.96;1" values="0;0;1;1;0;0"/>')
    for i, (x, w) in enumerate(zip(xs, wts)):
        sw = f"{max(w * 14, 1.5):.1f}"
        if i == 2:
            s += f'<path d="M{x - 24},256 C{x - 60},200 {x + 60},200 {x + 24},256" fill="none" stroke="{COLORS["teal"]}" stroke-width="{sw}" stroke-linecap="round" opacity="0">{fade(0.15)}</path>\n'
        else:
            lift = 120 + abs(x - qx) * 0.12
            s += f'<path d="M{qx},256 Q{(x + qx) / 2},{256 - lift:.0f} {x},256" fill="none" stroke="{COLORS["teal"]}" stroke-width="{sw}" stroke-linecap="round" opacity="0">{fade(0.15)}</path>\n'
        s += (f'<g opacity="0">{fade(0.3)}<rect x="{x - 24}" y="124" width="48" height="26" rx="8" fill="{BG}" stroke="{COLORS["teal"]}"/>'
              + txt(x, 142, f"{w:.2f}", 14, COLORS["teal"], 700) + "</g>\n")
    s += f'<g opacity="0">{fade(0.5)}' + txt(W / 2, 98, "Weights add up to 1.00. 'sat' mostly looks at 'cat' (who sat) and 'mat' (where).", 15, COLORS["amber"], 700) + "</g>\n"
    steps = [
        dict(label="Pick a word", sub="", c1="1. Take one word as the question: 'sat'", c2="It asks: which other words help me?"),
        dict(label="Look", sub="", c1="2. It looks at every word in the sentence", c2="Thicker arcs mean more attention."),
        dict(label="Weights", sub="", c1="3. Each word gets a weight, and they add up to 1.00", c2="'cat' gets 0.45 and 'mat' gets 0.25."),
        dict(label="Meaning", sub="", c1="4. 'sat' mostly looks at 'cat' (who sat) and 'mat' (where)", c2="Its new meaning is a weighted mix of those words."),
    ]
    return s + "</svg>\n", dur, [1.0, 2.8, 4.5, 7.0], steps


CUSTOM = {"chunking": _chunking, "cosine-similarity": _cosine, "attention": _attention}


def render(a):
    """Return (svg, duration, marks, steps)."""
    k = a["kind"]
    if k == "pipeline":
        svg, dur, marks = render_pipeline(a)
    elif k == "loop":
        svg, dur, marks = render_loop(a)
    elif k == "bars":
        svg, dur, marks = render_bars(a)
    else:
        svg, dur, marks, steps = CUSTOM[a["name"]](a)
        return svg, dur, marks, steps
    return svg, dur, marks, a["steps"]
