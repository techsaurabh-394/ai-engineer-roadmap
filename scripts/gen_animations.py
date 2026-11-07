#!/usr/bin/env python3
"""Generate the animated explainer SVGs in assets/anim/.

Pure standard library. The SVGs use SMIL (<animate>), which GitHub renders
inline in README and topic pages, and every browser plays natively.
Run:  python scripts/gen_animations.py
"""
from __future__ import annotations

import math
import pathlib
from xml.sax.saxutils import escape

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "anim"
BG, CARD, LINE, TEAL, TEXT, MUTED, AMBER, BLUE, PINK = (
    "#0D1117", "#161B22", "#30363D", "#2DD4BF", "#E6EDF3", "#9DA7B3", "#F5B042", "#6EA8FE", "#F472B6")
FONT = "-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "SFMono-Regular,Menlo,Consolas,monospace"


def head(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n<title id="t">{escape(title)}</title>\n'
            f'<desc id="d">{escape(desc)}</desc>\n'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>\n'
            f'<rect width="{w}" height="{h}" rx="16" fill="{BG}"/>\n')


def txt(x, y, s, size=15, fill=TEXT, weight=400, anchor="middle", mono=False, extra=""):
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{fam}{extra}>{escape(s)}</text>\n')


def slot_anim(attr, on, off, i, n, dur, fade=0.012):
    """Animate `attr` to `on` during slot i of n, else `off`. Looping."""
    a, b = i / n, (i + 1) / n
    kt = [0, max(a - fade, 0), a, b, min(b + fade, 1), 1]
    kt = sorted(set(round(k, 4) for k in kt))
    vals = []
    for k in kt:
        inside = a - 1e-9 <= k <= b + 1e-9
        vals.append(on if inside else off)
    return (f'<animate attributeName="{attr}" dur="{dur}s" repeatCount="indefinite" '
            f'keyTimes="{";".join(map(str, kt))}" values="{";".join(vals)}"/>')


def caption_block(cx, y, lines, i, n, dur, size=15):
    out = ""
    for j, line in enumerate(lines):
        col = TEXT if j == 0 else MUTED
        wt = 700 if j == 0 else 400
        out += (f'<text x="{cx}" y="{y + j * 24}" font-size="{size}" fill="{col}" font-weight="{wt}" '
                f'text-anchor="middle" opacity="0">{escape(line)}{slot_anim("opacity", "1", "0", i, n, dur)}</text>\n')
    return out


def packet(path_x, path_y, n, dur, r=9, color=AMBER):
    """A dot that hops through points; moves during first 35% of each slot."""
    kt, xs, ys = [0.0], [path_x[0]], [path_y[0]]
    for i in range(n):
        a, b = i / n, (i + 1) / n
        move_end = a + (b - a) * 0.35
        if i > 0:
            kt += [a]; xs += [path_x[i - 1]]; ys += [path_y[i - 1]]
        kt += [move_end if i > 0 else a + (b - a) * 0.05]
        xs += [path_x[i]]; ys += [path_y[i]]
    kt += [1.0]; xs += [path_x[-1]]; ys += [path_y[-1]]
    # keyTimes must be non-decreasing
    last = -1; fixed = []
    for k in kt:
        k = max(k, last); fixed.append(round(min(k, 1), 4)); last = k
    ktxt = ";".join(map(str, fixed))
    return (f'<circle r="{r}" fill="{color}" cx="{xs[0]}" cy="{ys[0]}">'
            f'<animate attributeName="cx" dur="{dur}s" repeatCount="indefinite" keyTimes="{ktxt}" values="{";".join(map(str, xs))}"/>'
            f'<animate attributeName="cy" dur="{dur}s" repeatCount="indefinite" keyTimes="{ktxt}" values="{";".join(map(str, ys))}"/>'
            f'</circle>\n')


def pipeline(name, title, subtitle, stages, color=TEAL):
    """stages: list of (label, sublabel, caption_line1, caption_line2)."""
    n = len(stages)
    W, H = 880, 330
    slot = 2.4
    dur = round(n * slot, 2)
    bw = 118 if n >= 6 else 128
    gap = (W - 60 - n * bw) / max(n - 1, 1)
    xs = [30 + i * (bw + gap) for i in range(n)]
    cy = 150
    s = head(W, H, title, subtitle)
    s += txt(W / 2, 40, title, 22, TEXT, 700)
    s += txt(W / 2, 64, subtitle, 14, MUTED)
    for i in range(n - 1):
        x1, x2 = xs[i] + bw, xs[i + 1]
        s += (f'<line x1="{x1 + 2}" y1="{cy}" x2="{x2 - 4}" y2="{cy}" stroke="{LINE}" stroke-width="2.5" marker-end="url(#ah)"/>\n')
    for i, (lab, sub, *_c) in enumerate(stages):
        x = xs[i]
        s += (f'<rect x="{x}" y="{cy - 46}" width="{bw}" height="92" rx="12" fill="{CARD}" stroke="{LINE}" stroke-width="2">'
              f'{slot_anim("stroke", color, LINE, i, n, dur)}</rect>\n')
        s += (f'<circle cx="{x + 18}" cy="{cy - 46}" r="13" fill="{color}"/>'
              + txt(x + 18, cy - 41, str(i + 1), 14, BG, 700))
        s += txt(x + bw / 2, cy - 4, lab, 16, TEXT, 700)
        s += txt(x + bw / 2, cy + 20, sub, 12.5, MUTED)
    s += packet([x + bw / 2 for x in xs], [cy - 62] * n, n, dur)
    s += f'<rect x="30" y="226" width="{W - 60}" height="84" rx="12" fill="{CARD}" stroke="{LINE}"/>\n'
    for i, (_l, _s, c1, c2) in enumerate(stages):
        s += caption_block(W / 2, 258, [c1, c2], i, n, dur)
    s += "</svg>\n"
    (OUT / f"{name}.svg").write_text(s)


def loop(name, title, subtitle, nodes, color=TEAL):
    """Circular loop. nodes: list of (label, caption1, caption2)."""
    n = len(nodes)
    W, H = 880, 420
    cx, cy, rx, ry = 440, 240, 250, 105
    slot = 2.4
    dur = round(n * slot, 2)
    pts = []
    for i in range(n):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pts.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))
    s = head(W, H, title, subtitle)
    s += txt(W / 2, 40, title, 22, TEXT, 700)
    s += txt(W / 2, 64, subtitle, 14, MUTED)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        s += f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{LINE}" stroke-width="2.5" marker-end="url(#ah)" opacity="0.9"/>\n'
    bw, bh = 150, 50
    for i, (lab, *_c) in enumerate(nodes):
        x, y = pts[i]
        s += (f'<rect x="{x - bw / 2:.0f}" y="{y - bh / 2:.0f}" width="{bw}" height="{bh}" rx="12" fill="{CARD}" stroke="{LINE}" stroke-width="2">'
              f'{slot_anim("stroke", color, LINE, i, n, dur)}</rect>\n')
        s += txt(round(x), round(y + 6), lab, 16, TEXT, 700)
    px = [p[0] for p in pts]; py = [p[1] - 36 for p in pts]
    s += packet(px, py, n, dur, 9)
    for i, (_l, c1, c2) in enumerate(nodes):
        s += caption_block(cx, cy - 8, [c1, c2], i, n, dur, 15)
    s += "</svg>\n"
    (OUT / f"{name}.svg").write_text(s)


def chunking():
    W, H = 880, 360
    dur = 12
    words = "Multi-stage builds compile code in one stage and copy only the final artifact into a small runtime image".split()
    words = words[:15]
    s = head(W, H, "Chunking with overlap", "A long text is cut into overlapping pieces so no idea is split in half.")
    s += txt(W / 2, 40, "Chunking with overlap", 22, TEXT, 700)
    s += txt(W / 2, 64, "A long text is cut into overlapping pieces so no idea is split in half.", 14, MUTED)
    bw, gap, x0 = 52, 3, 30
    def wx(i): return x0 + i * (bw + gap)
    s += txt(30, 100, "Document", 13, MUTED, 700, "start")
    for i, w in enumerate(words):
        s += f'<rect x="{wx(i)}" y="108" width="{bw}" height="30" rx="6" fill="{CARD}" stroke="{LINE}"/>' + txt(wx(i) + bw / 2, 128, w[:7], 11, TEXT)
    chunks = [(0, 6, TEAL, "Chunk 1"), (5, 11, BLUE, "Chunk 2"), (10, 15, PINK, "Chunk 3")]
    for k, (a, b, col, lab) in enumerate(chunks):
        y = 180 + k * 52
        t0 = 0.12 + k * 0.22
        s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
              f'keyTimes="0;{t0};{t0 + 0.04};0.9;0.96;1" values="0;0;1;1;0;0"/>')
        s += txt(30, y - 8, f"{lab}  (words {a + 1} to {b})", 13, col, 700, "start")
        for i in range(a, b):
            overlap = (k > 0 and i < chunks[k - 1][1]) 
            stroke = AMBER if overlap else col
            s += f'<rect x="{wx(i)}" y="{y}" width="{bw}" height="30" rx="6" fill="{CARD}" stroke="{stroke}" stroke-width="{3 if overlap else 1.5}"/>' + txt(wx(i) + bw / 2, y + 20, words[i][:7], 11, TEXT)
        s += "</g>\n"
    s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.66;0.7;0.9;0.96;1" values="0;0;1;1;0;0"/>'
          + txt(W / 2, 346, "Amber outline = overlap: the shared words keep context across the cut.", 14, AMBER, 700) + "</g>\n")
    s += "</svg>\n"
    (OUT / "chunking.svg").write_text(s)


def cosine():
    W, H = 880, 380
    dur = 12
    ox, oy, L = 260, 300, 190
    angles = [10, 45, 90, 135, 180]
    cosv = [math.cos(math.radians(a)) for a in angles]
    notes = ["almost the same meaning", "related", "unrelated", "leaning opposite", "opposite meaning"]
    s = head(W, H, "Cosine similarity", "The angle between two embedding vectors tells you how similar their meaning is.")
    s += txt(W / 2, 40, "Cosine similarity", 22, TEXT, 700)
    s += txt(W / 2, 64, "The angle between two embedding vectors tells you how similar their meaning is.", 14, MUTED)
    s += f'<line x1="40" y1="{oy}" x2="480" y2="{oy}" stroke="{LINE}" stroke-dasharray="4 4"/>\n'
    s += f'<line x1="{ox}" y1="{oy}" x2="{ox + L}" y2="{oy}" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/>\n'
    s += txt(ox + L + 8, oy - 8, "query", 14, TEAL, 700, "start")
    n = len(angles)
    vals = ";".join(f"{-a} {ox} {oy}" for a in angles)
    kt = ";".join(str(round(i / (n - 1), 4)) for i in range(n))
    s += (f'<g><line x1="{ox}" y1="{oy}" x2="{ox + L}" y2="{oy}" stroke="{AMBER}" stroke-width="4" stroke-linecap="round"/>'
          + txt(ox + L + 8, oy - 8, "chunk", 14, AMBER, 700, "start")
          + f'<animateTransform attributeName="transform" type="rotate" dur="{dur}s" repeatCount="indefinite" calcMode="linear" keyTimes="{kt}" values="{vals}"/></g>\n')
    s += f'<rect x="540" y="110" width="310" height="210" rx="14" fill="{CARD}" stroke="{LINE}"/>\n'
    s += txt(695, 142, "score = cos(angle)", 16, TEXT, 700)
    for i, (a, c, note) in enumerate(zip(angles, cosv, notes)):
        s += (f'<g opacity="0">{slot_anim("opacity", "1", "0", i, n, dur, 0.02)}'
              + txt(695, 205, f"{c:+.2f}", 46, AMBER, 700)
              + txt(695, 238, f"angle {a} degrees", 14, MUTED)
              + txt(695, 276, note, 17, TEXT, 700) + "</g>\n")
    s += txt(695, 302, "1 = same direction, 0 = unrelated, -1 = opposite", 12.5, MUTED)
    s += "</svg>\n"
    (OUT / "cosine-similarity.svg").write_text(s)


def attention():
    W, H = 880, 380
    dur = 10
    toks = ["The", "cat", "sat", "on", "the", "mat"]
    wts = [0.05, 0.45, 0.15, 0.05, 0.05, 0.25]
    s = head(W, H, "Attention: who should this word look at?", "The word 'sat' scores every word. Higher weight = more attention.")
    s += txt(W / 2, 40, "Attention: who should this word look at?", 22, TEXT, 700)
    s += txt(W / 2, 64, "The word 'sat' scores every word. Higher weight = more attention.", 14, MUTED)
    xs = [90 + i * 140 for i in range(6)]
    qx = xs[2]
    for i, (x, t) in enumerate(zip(xs, toks)):
        is_q = i == 2
        s += (f'<rect x="{x - 48}" y="256" width="96" height="44" rx="10" fill="{CARD}" stroke="{AMBER if is_q else LINE}" stroke-width="{3 if is_q else 2}"/>'
              + txt(x, 285, t, 18, TEXT, 700))
    s += txt(qx, 324, "query (asking)", 13, AMBER, 700)
    for i, (x, w) in enumerate(zip(xs, wts)):
        if i == 2:
            s += (f'<path d="M{x - 24},256 C{x - 60},200 {x + 60},200 {x + 24},256" fill="none" stroke="{TEAL}" stroke-width="{max(w * 14, 1.5):.1f}" stroke-linecap="round" opacity="0">'
                  f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.15;0.2;0.9;0.96;1" values="0;0;1;1;0;0"/></path>\n')
        else:
            mid = (x + qx) / 2
            lift = 120 + abs(x - qx) * 0.12
            s += (f'<path d="M{qx},256 Q{mid},{256 - lift:.0f} {x},256" fill="none" stroke="{TEAL}" stroke-width="{max(w * 14, 1.5):.1f}" stroke-linecap="round" opacity="0">'
                  f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.15;0.2;0.9;0.96;1" values="0;0;1;1;0;0"/></path>\n')
        s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.3;0.36;0.9;0.96;1" values="0;0;1;1;0;0"/>'
              + f'<rect x="{x - 24}" y="124" width="48" height="26" rx="8" fill="{BG}" stroke="{TEAL}"/>'
              + txt(x, 142, f"{w:.2f}", 14, TEAL, 700) + "</g>\n")
    s += (f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="0;0.5;0.56;0.9;0.96;1" values="0;0;1;1;0;0"/>'
          + txt(W / 2, 98, "Weights add up to 1.00. 'sat' mostly looks at 'cat' (who sat) and 'mat' (where).", 15, AMBER, 700) + "</g>\n")
    s += "</svg>\n"
    (OUT / "attention.svg").write_text(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline("rag-flow", "RAG, step by step", "Watch one question travel through the whole pipeline.", [
        ("Documents", "your files", "1. Start with your own documents", "PDFs, wiki pages, docs. The model has never seen these."),
        ("Chunk", "small pieces", "2. Cut them into small chunks", "Small pieces are easier to match to a question."),
        ("Embed", "text to numbers", "3. Turn each chunk into a vector", "Similar meaning gives similar numbers."),
        ("Store", "vector database", "4. Save the vectors in a vector database", "Done once, ahead of time."),
        ("Retrieve", "top matches", "5. Embed the question and find the closest chunks", "Cosine similarity picks the best 3 to 5."),
        ("Generate", "answer + cites", "6. Give the chunks to the LLM with the question", "It answers from them and cites the source."),
    ])
    pipeline("llm-flow", "From text to answer inside an LLM", "What happens to your prompt, in order.", [
        ("Text", "your prompt", "1. You type a prompt", "Just plain text."),
        ("Tokens", "word pieces", "2. Text is split into tokens", "Tokens are word pieces, each with an ID number."),
        ("Embeddings", "ID to vector", "3. Each token becomes a vector", "A list of numbers that carries meaning."),
        ("Attention", "who matters", "4. Attention mixes in context", "Every token looks at the others to find what matters."),
        ("Next token", "probabilities", "5. The model scores every possible next token", "It picks one, adds it, and repeats."),
    ], BLUE)
    pipeline("tool-calling-flow", "Tool calling, step by step", "The model asks, your code acts.", [
        ("Question", "from the user", "1. User asks something the model cannot know", "For example: what is the weather in Delhi?"),
        ("Model", "picks a tool", "2. The model replies with a tool request", "It names the tool and its inputs. It does not run anything."),
        ("Your code", "runs the tool", "3. YOUR code runs the function", "This is where you check permissions and limits."),
        ("Result", "sent back", "4. The result goes back to the model", "As a new message in the conversation."),
        ("Answer", "in plain words", "5. The model writes the final answer", "Using the tool result as facts."),
    ], AMBER)
    pipeline("mcp-flow", "MCP: how an AI app uses a tool server", "One standard way to plug tools into any AI app.", [
        ("Host app", "Claude, IDE...", "1. The host app starts an MCP client", "The host is the AI app you use."),
        ("Connect", "handshake", "2. Client and server agree on features", "They exchange name, version and capabilities."),
        ("List", "tools/list", "3. The client asks: what can you do?", "The server returns tools with descriptions and schemas."),
        ("Call", "tools/call", "4. The model picks a tool and the client calls it", "Inputs follow the tool's JSON schema."),
        ("Result", "content back", "5. The server returns the result", "The model reads it and continues."),
    ], PINK)
    pipeline("hybrid-search-flow", "Hybrid search", "Keyword search and meaning search, merged.", [
        ("Query", "user question", "1. One question comes in", "It is sent down two paths at once."),
        ("BM25", "exact words", "2. Keyword path: BM25 finds exact word matches", "Great for names, codes and rare terms."),
        ("Dense", "meaning", "3. Meaning path: embeddings find similar ideas", "Great for paraphrases."),
        ("Merge", "RRF", "4. Reciprocal Rank Fusion merges both rankings", "A chunk ranked high in both rises to the top."),
        ("Top-k", "to the LLM", "5. The top chunks go to the model", "You get the strengths of both."),
    ], TEAL)
    pipeline("fine-tuning-flow", "Fine-tuning, step by step", "Teach a base model your style or task.", [
        ("Data", "examples", "1. Collect example inputs and ideal outputs", "Quality beats quantity."),
        ("Clean", "check it", "2. Clean and split into train and test", "Remove duplicates and wrong labels."),
        ("Train", "LoRA / full", "3. Train: nudge the weights toward your examples", "LoRA trains a tiny add-on, so it is cheap."),
        ("Evaluate", "on test set", "4. Test on examples the model never saw", "Compare with the base model."),
        ("Deploy", "serve it", "5. Ship it, then keep monitoring", "Retrain when your data changes."),
    ], BLUE)
    loop("react-loop", "ReAct loop: Think, Act, Observe", "Like a detective: each clue changes the next step.", [
        ("Thought", "1. Think: what do I need?", "Example: I need France's population."),
        ("Action", "2. Act: choose a tool + input", "Example: lookup({topic: france})."),
        ("Observation", "3. Observe: your code runs the tool", "Result: about 68 million."),
        ("Final Answer", "4. Enough facts? Write the answer", "If not, loop again (up to a max number of steps)."),
    ], TEAL)
    loop("agent-loop-anim", "The agent loop", "An agent is a model + tools + memory in a loop.", [
        ("Goal", "1. Receive the goal", "For example: book the cheapest train."),
        ("Plan", "2. Decide the next step", "The model picks a tool or answers."),
        ("Act", "3. Run the tool", "Your code executes it safely."),
        ("Observe", "4. Read the result", "Add it to memory, then check if the goal is met."),
    ], AMBER)
    loop("eval-loop", "The evaluation loop", "Measure, change one thing, measure again.", [
        ("Test set", "1. Keep a fixed set of questions", "With known good answers."),
        ("Run", "2. Run your system on all of them", "Log every answer and every source."),
        ("Score", "3. Score: accuracy, faithfulness, hit@k", "Numbers, not feelings."),
        ("Improve", "4. Change ONE thing, then re-run", "Keep it only if the score goes up."),
    ], PINK)
    chunking(); cosine(); attention()
    print("wrote", len(list(OUT.glob("*.svg"))), "svgs to", OUT)


if __name__ == "__main__":
    main()
