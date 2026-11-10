#!/usr/bin/env python3
"""Everything the site's labs show, taken from the repo's own demo code (standard library only).

Nothing here is typed by hand: tables, weights, sample texts and printed output are read from
the modules in <phase>/code/ at build time, so the pages cannot drift away from the demos.

    python scripts/site_data.py        # prints a short summary
"""
import contextlib
import importlib.util
import io
import os
import pathlib
import re
import runpy
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/techsaurabh-394/ai-engineer-roadmap"
P1 = ROOT / "01-rag-foundations" / "code"
P3 = ROOT / "03-llm-concepts" / "code"


# Which lessons each lab sits under. Names are animation names (the #deep-link of a lesson).
LAB_META = {
    "sampling": dict(title="Sampling lab", phase="03", verb="Sliders",
                     blurb="Move temperature, top-k and top-p and watch the next-token odds reshape.",
                     lessons=["temperature", "top-k", "top-p", "12-beam-search"]),
    "attention": dict(title="Attention explorer", phase="03", verb="Hover and toggle",
                      blurb="Pick a word, see which other words it looks at, then open up the Q, K and V numbers behind it.",
                      lessons=["attention", "22-qkv-explained-from-scratch", "03-multi-head-attention"]),
    "toyllm": dict(title="Toy LLM, end to end", phase="03", verb="Build a sentence",
                   blurb="Build a sentence from 8 words and follow it through every stage of a tiny transformer to the next word.",
                   lessons=["llm-flow", "01-transformer-architecture", "05-decoder-only-models"]),
    "chunking": dict(title="Chunking lab", phase="01", verb="Sliders",
                     blurb="Paste a paragraph, change chunk size and overlap, and see exactly where the cuts land.",
                     lessons=["chunking", "05-recursive-chunking", "06-semantic-chunking", "08-sliding-window-chunking"]),
    "similarity": dict(title="Similarity lab", phase="01", verb="Drag",
                       blurb="Drag two vectors and watch cosine, dot product and distance change, and sometimes disagree.",
                       lessons=["cosine-similarity", "12-euclidean-distance", "13-dot-product", "02-embeddings"]),
    "search": dict(title="Search lab", phase="01", verb="Type a query",
                   blurb="Run one query through keyword search (BM25), a toy vector search and the hybrid of both.",
                   lessons=["hybrid-search-flow", "17-bm25", "18-sparse-retrieval", "19-dense-retrieval", "20-metadata-filtering"]),
}


def load(path: pathlib.Path):
    """Import a demo module by file path (the code folders are not packages)."""
    name = "aier_demo_" + path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main_globals(path: pathlib.Path) -> dict:
    """Run a demo as a script (its `if __name__ == "__main__"` block) and return its variables."""
    argv = sys.argv
    sys.argv = [str(path)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = argv


def pairs(d: dict) -> list:
    """dict -> [[key, value], ...]: keeps Python's insertion order when it travels as JSON."""
    return [[k, v] for k, v in d.items()]


def labs() -> dict:
    sd, dec = load(P3 / "sampling_demo.py"), load(P3 / "llm_decoding.py")
    qkv, e2e = load(P3 / "llm_qkv.py"), load(P3 / "llm_e2e_walkthrough.py")
    ch, bm, vm = load(P1 / "rag_chunkers.py"), load(P1 / "rag_bm25.py"), load(P1 / "rag_vector_math.py")
    att_main = main_globals(P3 / "llm_attention.py")
    vec_main = main_globals(P1 / "rag_vector_math.py")
    e2e_main = main_globals(P3 / "llm_e2e_walkthrough.py")
    return {
        "sampling": {"TOKENS": sd.TOKENS, "LOGITS": sd.LOGITS, "src": "03-llm-concepts/code/sampling_demo.py"},
        "decoding": {"NEXT": {k: pairs(v) for k, v in dec.NEXT.items()}, "END": dec.END,
                     "src": "03-llm-concepts/code/llm_decoding.py"},
        "qkv": {"TOKENS": qkv.TOKENS, "X": qkv.X, "W_Q": qkv.W_Q, "W_K": qkv.W_K, "W_V": qkv.W_V,
                "src": "03-llm-concepts/code/llm_qkv.py"},
        "attention": {"x": att_main["x"], "src": "03-llm-concepts/code/llm_attention.py"},
        "toyllm": {"VOCAB": e2e.VOCAB, "EMBEDDING": e2e.EMBEDDING, "W_Q": e2e.W_Q, "W_K": e2e.W_K, "W_V": e2e.W_V,
                   "W_FF": e2e.W_FF, "W_OUT": e2e.W_OUT, "N_BLOCKS": e2e.N_BLOCKS, "D_MODEL": e2e.D_MODEL,
                   "POS_SCALE": e2e.POS_SCALE, "example": e2e_main["text"],
                   "src": "03-llm-concepts/code/llm_e2e_walkthrough.py"},
        "chunking": {"SAMPLE": ch.SAMPLE, "src": "01-rag-foundations/code/rag_chunkers.py"},
        "vectors": {"docs": vec_main["docs"], "q": vec_main["q"], "worked": pairs(vm.worked_examples()),
                    "src": "01-rag-foundations/code/rag_vector_math.py"},
        "search": {"DOCS": bm.DOCS, "STOP": sorted(bm.STOP), "src": "01-rag-foundations/code/rag_bm25.py"},
    }


# ----------------------------------------------------------------- demo output
def demo_scripts() -> list:
    return sorted(p for p in ROOT.glob("0[1-7]-*/code/*.py") if not p.name.startswith("test_"))


def _run(path: pathlib.Path):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}   # always the offline path
    env["PYTHONHASHSEED"] = "0"
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60, env=env, cwd=str(ROOT))
    return r.returncode, r.stdout


def demo_outputs() -> dict:
    """{repo-relative path: stdout} for every demo whose output is the same on two runs.

    A demo that prints timings or random ids is left out: showing one frozen run of it would be misleading.
    """
    out = {}
    for p in demo_scripts():
        (c1, o1), (c2, o2) = _run(p), _run(p)
        if c1 == 0 and c2 == 0 and o1 == o2 and o1.strip():
            out[p.relative_to(ROOT).as_posix()] = o1.rstrip("\n")
    return out


def topic_demos(topic_text: str, phase_dir: str) -> list:
    """Demo scripts a topic page tells the reader to run, in the order they are mentioned."""
    seen = []
    for name in re.findall(r"code/([a-z_0-9]+\.py)", topic_text):
        rel = f"{phase_dir}/code/{name}"
        if not name.startswith("test_") and (ROOT / rel).exists() and rel not in seen:
            seen.append(rel)
    return seen


if __name__ == "__main__":
    d = labs()
    print("labs:", ", ".join(f"{k} ({v['src'].rsplit('/', 1)[1]})" for k, v in d.items()))
    outs = demo_outputs()
    print(f"demo output captured for {len(outs)} of {len(demo_scripts())} scripts")
