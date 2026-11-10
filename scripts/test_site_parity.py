"""The labs on the site run JavaScript ports of the Python demos. These tests run that JavaScript
under node and compare it with the Python it was ported from.

How strict:
  * strings, integers, rankings, sampled tokens, chunk lists: must be identical.
  * every number shown on a page (formatted to 2-4 decimals): must be the identical string.
  * raw floats: identical when the code path only uses + - * / sqrt (and Python is >= 3.12, whose
    sum() is compensated: older Pythons add up in a plain loop and can differ in the last bit);
    within 1e-12 when exp/log/pow/sin/cos are involved, because V8 and the C library may round the
    last bit of those differently. That gap is fixed by the platform, not random, so it cannot flake.
"""
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import site_data as sd  # noqa: E402

SITE = pathlib.Path(__file__).resolve().parent / "site"
P1, P3 = sd.P1, sd.P3
dec, sdemo = sd.load(P3 / "llm_decoding.py"), sd.load(P3 / "sampling_demo.py")
att, qkv, e2e = sd.load(P3 / "llm_attention.py"), sd.load(P3 / "llm_qkv.py"), sd.load(P3 / "llm_e2e_walkthrough.py")
vm, ch, bm = sd.load(P1 / "rag_vector_math.py"), sd.load(P1 / "rag_chunkers.py"), sd.load(P1 / "rag_bm25.py")
DATA = sd.labs()
EXACT_SUM = sys.version_info >= (3, 12)
NODE = shutil.which("node")


def js(cases):
    """Run [[op, *args], ...] through scripts/site/parity_node.js and return the results."""
    if NODE is None:
        if os.environ.get("CI"):
            pytest.fail("node is required in CI for the JS/Python parity tests")
        pytest.skip("node is not installed")
    r = subprocess.run([NODE, str(SITE / "parity_node.js")], input=json.dumps({"data": DATA, "cases": cases}),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert len(out) == len(cases)
    return out


def norm(v):
    """Python value -> the shape the JS side sends back."""
    if isinstance(v, float) and math.isinf(v):
        return "-inf" if v < 0 else "inf"
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    return v


def same(got, want, exact, path="$"):
    want = norm(want)
    if isinstance(want, bool) or isinstance(want, str) or want is None:
        assert got == want, f"{path}: {got!r} != {want!r}"
    elif isinstance(want, (int, float)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{path}: {got!r} is not a number"
        if exact or isinstance(want, int):
            assert got == want, f"{path}: {got!r} != {want!r}"
        else:
            assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12), f"{path}: {got!r} !~ {want!r}"
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length {len(got) if isinstance(got, list) else got!r} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            same(g, w, exact, f"{path}[{i}]")
    elif isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            same(got[k], want[k], exact, f"{path}.{k}")
    else:
        raise TypeError(type(want))


def check(cases, expected, exact):
    for case, res, want in zip(cases, js(cases), expected):
        assert "ok" in res, f"{case[0]}{case[1:]!r}: JS raised {res.get('err')}"
        same(res["ok"], want, exact, f"{case[0]}{json.dumps(case[1:])[:120]}")


def pairs(d):
    return [[k, v] for k, v in d.items()]


def fmt_pairs(d, digits):
    return [[k, f"{v:.{digits}f}"] for k, v in d.items()]


# --------------------------------------------------------------- formatting, sum, RNG
def test_python_number_formatting_matches():
    rng = random.Random(7)
    xs = [0.0625, 0.125, 0.375, 2.5, 3.5, 0.5, 1.5, -0.0625, 1e-7, 123456.789, 0.9995, 0.0005, 1.0, -1.0, 0.1 + 0.2,
          1 / 3, 2 / 3, 1e16, 1.5e16, 5e-5, 1e-4, 0.00012345, 4.35, 2.675, 1.005, 0.045, 0.285, 1e15, 9007199254740993.0, 0.0, -0.0]
    xs += [rng.uniform(-5, 5) for _ in range(400)] + [rng.random() * 10 ** rng.randint(-8, 8) for _ in range(400)]
    xs += [k / 2 ** 10 for k in range(0, 1024, 7)] + [k / 2 ** 4 for k in range(-40, 41)]      # many exact ties
    cases, want = [], []
    for x in xs:
        for d in (0, 1, 2, 3, 4, 6):
            cases.append(["fixed", x, d]); want.append(f"{x:.{d}f}")
            cases.append(["round", x, d]); want.append(float(round(x, d)))
        cases.append(["repr", x]); want.append(repr(float(x)))
    check(cases, want, exact=True)


def test_sum_matches_builtin_sum():
    if not EXACT_SUM:
        pytest.skip("sum() is compensated only from Python 3.12")
    rng = random.Random(11)
    lists = [[0.1] * 10, [1e16, 1.0, -1e16], [1.0, 1e100, 1.0, -1e100], [0.1, 0.2, 0.3], []]
    lists += [[rng.uniform(-1, 1) * 10 ** rng.randint(-6, 6) for _ in range(rng.randint(1, 12))] for _ in range(500)]
    check([["sum", xs] for xs in lists], [sum(xs) for xs in lists], exact=True)


def test_whitespace_helpers_match_str_methods():
    texts = ["  a  b\tc\n\nd  ", "", "   ", "a b c　d", "x\x1cy\x1dz\x85w", "﻿a b", "one", " a b c "]
    cases = [["split", t] for t in texts] + [["strip", t] for t in texts]
    check(cases, [t.split() for t in texts] + [t.strip() for t in texts], exact=True)


def test_mersenne_twister_matches_random_module():
    seeds = [0, 1, 2, 3, 42, 123456789, 2 ** 31, 2 ** 32 - 1]
    def draws(seed, n):
        r = random.Random(seed)
        return [r.random() for _ in range(n)]

    check([["random", s, 60] for s in seeds], [draws(s, 60) for s in seeds], exact=True)
    pop, w = ["a", "b", "c", "d"], [0.5, 0.25, 0.2, 0.05]
    cases = [["choices", s, pop, w, 200] for s in seeds]
    want = [(lambda r: [r.choices(pop, weights=w)[0] for _ in range(200)])(random.Random(s)) for s in seeds]
    check(cases, want, exact=True)


# --------------------------------------------------------------- sampling lab
TEMPS = [round(0.1 * i, 1) for i in range(1, 31)]          # the slider: 0.1 .. 3.0
PS = [round(0.05 * i, 2) for i in range(1, 21)]           # the slider: 0.05 .. 1.00


def test_softmax_and_demo_top_p_match_sampling_demo():
    cases, want = [], []
    for t in TEMPS:
        cases.append(["softmax", sdemo.LOGITS, t]); want.append(sdemo.softmax(sdemo.LOGITS, t))
        for p in PS:
            probs = sdemo.softmax(sdemo.LOGITS, t)
            cases.append(["demo_top_p", sdemo.TOKENS, probs, p]); want.append(pairs(sdemo.top_p_filter(probs, p)))
    check(cases, want, exact=False)


def _chain_logits(t, k, p):
    probs = dict(zip(sdemo.TOKENS, sdemo.softmax(sdemo.LOGITS, t)))
    if k is not None:
        probs = dec.top_k_filter(probs, k)
    if p is not None:
        probs = dec.top_p_filter(probs, p)
    return probs


def _chain_table(prev, t, k, p):
    probs = dec.apply_temperature(dec.next_probs(prev), t)       # the body of llm_decoding.sample()
    if k is not None:
        probs = dec.top_k_filter(probs, k)
    if p is not None:
        probs = dec.top_p_filter(probs, p)
    return probs


def test_every_slider_position_of_the_sampling_lab_matches():
    """All 30 temperatures x every k x all 20 p values, for both sources: raw values and shown strings."""
    cases, want, shown = [], [], []
    ks = [None] + list(range(1, len(sdemo.TOKENS) + 1))
    for t in TEMPS:
        for k in ks:
            for p in [None] + PS:
                cases.append(["chain_logits", sdemo.TOKENS, sdemo.LOGITS, t, k, p])
                d = _chain_logits(t, k, p); want.append(pairs(d)); shown.append(fmt_pairs(d, 3))
    for prev in list(dec.NEXT) + ["unknown-word"]:
        n = len(dec.next_probs(prev))
        for t in TEMPS:
            for k in [None] + list(range(1, n + 1)):
                for p in [None] + PS:
                    cases.append(["chain_table", prev, t, k, p])
                    d = _chain_table(prev, t, k, p); want.append(pairs(d)); shown.append(fmt_pairs(d, 3))
    results = js(cases)
    assert len(cases) > 20000
    for case, res, w, s in zip(cases, results, want, shown):
        assert "ok" in res, (case, res)
        same(res["ok"], w, False, f"{case}")
        assert [[tok, f"{v:.3f}"] for tok, v in res["ok"]] == s, case     # what the page prints


def test_filters_match_llm_decoding_on_random_distributions():
    rng = random.Random(5)
    cases, want = [], []
    for _ in range(300):
        n = rng.randint(1, 9)
        raw = [rng.random() + 1e-3 for _ in range(n)]
        if rng.random() < 0.3:                         # ties are where a sort can go wrong
            raw[rng.randrange(n)] = raw[0]
        d = {f"t{i}": v / sum(raw) for i, v in enumerate(raw)}
        k, p, t = rng.randint(1, n), rng.choice(PS), rng.choice(TEMPS)
        cases += [["top_k", pairs(d), k], ["top_p", pairs(d), p], ["apply_temperature", pairs(d), t]]
        want += [pairs(dec.top_k_filter(d, k)), pairs(dec.top_p_filter(d, p)), pairs(dec.apply_temperature(d, t))]
    check(cases, want, exact=False)


def test_decoders_match_llm_decoding():
    cases, want = [], []
    for start in ["<start>", "the", "a", "my", "big", "cat", "nope"]:
        cases.append(["greedy", start, 10]); want.append(dec.greedy(start))
        for width in (1, 2, 3, 5):
            cases.append(["beam", width, start, 10]); want.append(list(dec.beam_search(width, start)))
        for seed in range(12):
            for k, p, t in [(None, None, 1.0), (2, None, 1.0), (None, 0.8, 1.0), (2, 0.9, 0.5), (3, 0.5, 2.0), (1, None, 3.0)]:
                cases.append(["sample", start, 10, k, p, t, seed])
                want.append(dec.sample(start, k=k, p=p, temperature=t, rng=random.Random(seed)))
    seqs = [dec.greedy(), dec.beam_search(2)[0], ["a", "cat", "ran", dec.END]]
    cases += [["logprob", s, "<start>"] for s in seqs]; want += [dec.sequence_logprob(s) for s in seqs]
    cases += [["count_kv", n, c] for n in (1, 10, 100, 1000) for c in (True, False)]
    want += [dec.count_kv_computations(n, c) for n in (1, 10, 100, 1000) for c in (True, False)]
    check(cases, want, exact=False)


def test_the_demo_printout_is_reproduced():
    """The exact lines `python llm_decoding.py` and `python sampling_demo.py` print."""
    got = js([["greedy", "<start>", 10], ["beam", 2, "<start>", 10], ["sample", "<start>", 10, 2, None, 1.0, 1],
              ["sample", "<start>", 10, None, 0.8, 1.0, 2]] + [["softmax", sdemo.LOGITS, t] for t in (0.2, 1.0, 2.0)])
    assert " ".join(got[0]["ok"]) == "the cat sat <end>" == " ".join(dec.greedy())
    assert " ".join(got[1]["ok"][0]) == " ".join(dec.beam_search(2)[0])
    assert got[2]["ok"] == dec.sample(k=2, rng=random.Random(1)) and got[3]["ok"] == dec.sample(p=0.8, rng=random.Random(2))
    for res, t in zip(got[4:], (0.2, 1.0, 2.0)):
        assert [f"{p:.2f}" for p in res["ok"]] == [f"{p:.2f}" for p in sdemo.softmax(sdemo.LOGITS, t)]


# --------------------------------------------------------------- attention explorer
def _rand_matrix(rng, n, m, step=0.5):
    return [[rng.randint(-4, 4) * step for _ in range(m)] for _ in range(n)]


def test_qkv_trace_matches_llm_qkv():
    rng = random.Random(3)
    xs = [qkv.X] + [_rand_matrix(rng, 3, 4) for _ in range(150)]
    cases, want = [], []
    for x in xs:
        for causal in (True, False):
            cases.append(["trace", x, qkv.W_Q, qkv.W_K, qkv.W_V, causal])
            want.append(qkv.attention_with_trace(x, qkv.W_Q, qkv.W_K, qkv.W_V, causal))
    results = js(cases)
    for case, res, w in zip(cases, results, want):
        assert "ok" in res, res
        for key in ("x", "q", "k", "v", "scores", "scaled", "mask", "masked", "d_k"):       # + - * / sqrt only
            same(res["ok"][key], w[key], EXACT_SUM, key)
        for key in ("weights", "output"):                                                       # softmax uses exp
            same(res["ok"][key], w[key], False, key)
            shown = [[f"{v:.3f}" for v in row] for row in w[key]]
            assert [[f"{v:.3f}" for v in row] for row in res["ok"][key]] == shown, (case, key)
    check([["param_count", 4, 2, False], ["param_count", 4096, None, False], ["param_count", 8, 4, True]],
          [qkv.qkv_param_count(4, 2), qkv.qkv_param_count(4096), qkv.qkv_param_count(8, 4, True)], exact=True)


def test_attention_functions_match_llm_attention():
    rng = random.Random(9)
    x = DATA["attention"]["x"]
    cases = [["sdpa", x, x, x, False], ["sdpa", x, x, x, True], ["mha", x, 2, False], ["mha", x, 2, True], ["mha", x, 4, True]]
    want = [list(att.scaled_dot_product_attention(x, x, x)), list(att.scaled_dot_product_attention(x, x, x, att.causal_mask(3))),
            list(att.multi_head_attention(x, 2)), list(att.multi_head_attention(x, 2, True)), list(att.multi_head_attention(x, 4, True))]
    for _ in range(60):
        n = rng.randint(1, 5)
        q, k, v = _rand_matrix(rng, n, 4, 0.25), _rand_matrix(rng, n, 4, 0.25), _rand_matrix(rng, n, 3, 0.25)
        c = rng.random() < 0.5
        cases.append(["sdpa", q, k, v, c])
        want.append(list(att.scaled_dot_product_attention(q, k, v, att.causal_mask(n) if c else None)))
    for seq, d in [(3, 4), (1, 2), (8, 8), (16, 6), (50, 4)]:
        cases.append(["pos_enc", seq, d]); want.append(att.sinusoidal_positional_encoding(seq, d))
    check(cases, want, exact=False)
    mats = [(_rand_matrix(rng, 3, 4, 0.1), _rand_matrix(rng, 4, 2, 0.1)) for _ in range(100)]
    check([["matmul", a, b] for a, b in mats], [att.matmul(a, b) for a, b in mats], exact=EXACT_SUM)


def test_toy_llm_matches_llm_e2e_walkthrough():
    texts = ["the cat sat on the", "the", "the dog ran", "cat", "the cat sat on the mat.", "The Dog sat on", "mat mat mat", "on the"]
    cases = [["toy_tokenize", t] for t in texts + ["Hello world.", " a.  b ", "."]]
    want = [e2e.tokenize(t) for t in texts + ["Hello world.", " a.  b ", "."]]
    check(cases, want, exact=True)
    fwd = js([["toy_forward", t] for t in texts])
    for t, res in zip(texts, fwd):
        assert "ok" in res, res
        w = e2e.forward(e2e.to_ids(e2e.tokenize(t)))
        same(res["ok"], w, False, t)
        assert [f"{p:.3f}" for p in res["ok"]["probs"]] == [f"{p:.3f}" for p in w["probs"]]
        assert [[f"{v:6.3f}" for v in row] for row in res["ok"]["block_outputs"][-1]] == [[f"{v:6.3f}" for v in row] for row in w["block_outputs"][-1]]
    gen = js([["toy_generate", t, 4] for t in texts] + [["toy_params"]])
    for t, res in zip(texts, gen):
        final, steps = e2e.generate(t, 4)
        assert res["ok"][0] == final, t
        same(res["ok"][1], [[s["next_id"], s["probs"]] for s in steps], False, t)
    assert gen[-1]["ok"] == e2e.parameter_count()
    assert "RangeError" in js([["toy_forward", "the banana"]])[0]["err"]


# --------------------------------------------------------------- similarity lab
def test_vector_math_matches_rag_vector_math():
    rng = random.Random(21)
    vecs = [([1, 2, 3], [2, 4, 6]), ([1, 2, 3], [3, 0, -1]), ([0.0, 0.0], [1.0, 2.0]), ([1.0, 0.2], [5.0, 5.0])]
    vecs += [([rng.randint(-10, 10) * 0.5 for _ in range(2)], [rng.randint(-10, 10) * 0.5 for _ in range(2)]) for _ in range(400)]
    vecs += [([rng.uniform(-3, 3) for _ in range(d)], [rng.uniform(-3, 3) for _ in range(d)]) for d in (2, 3, 8, 64) for _ in range(50)]
    cases, want, cases_pow, want_pow = [], [], [], []
    for a, b in vecs:
        cases += [["dot", a, b], ["norm", a], ["cosine", a, b]]
        want += [vm.dot(a, b), vm.norm(a), vm.cosine_similarity(a, b)]
        cases_pow.append(["euclid", a, b]); want_pow.append(vm.euclidean_distance(a, b))        # uses ** 2 (pow)
        if vm.norm(a):
            cases.append(["normalize", a]); want.append(vm.normalize(a))
    check(cases, want, exact=EXACT_SUM)
    check(cases_pow, want_pow, exact=False)
    shown = js(cases_pow)
    assert [f"{r['ok']:.3f}" for r in shown] == [f"{w:.3f}" for w in want_pow]
    docs, q = DATA["vectors"]["docs"], DATA["vectors"]["q"]
    rcases = [["rank", q, docs, m] for m in ("cosine", "dot", "euclidean")]
    rwant = [vm.rank(q, docs, m) for m in ("cosine", "dot", "euclidean")]
    for _ in range(200):
        qq = [rng.randint(-8, 8) * 0.5, rng.randint(-8, 8) * 0.5]
        for m in ("cosine", "dot", "euclidean"):
            rcases.append(["rank", qq, docs, m]); rwant.append(vm.rank(qq, docs, m))
    check(rcases, rwant, exact=True)


# --------------------------------------------------------------- chunking lab
TEXTS = [
    ch.SAMPLE,
    "One sentence only.",
    "",
    "   \n\n  ",
    "No full stop here just a long run of words that keeps going and going without any punctuation at all",
    "Short. Tiny! Is it? Yes.  Two  spaces.\nNew line.\n\nNew paragraph. End",
    "Unicode: café naïve 日本語の文章です。 emoji \U0001f600\U0001f680 done. Tabs\tand nbsp em space.",
    "word " * 120,
    "A.B.C. D.E.F.\n\n\n\nG. H.",
    "x" * 500,
]


def test_chunkers_match_rag_chunkers():
    cases, want = [], []
    for t in TEXTS:
        for size in (1, 5, 20, 60, 120, 200, 400):
            for overlap in (0, 1, 10, 50):
                if overlap < size:
                    cases.append(["fixed_chunks", t, size, overlap]); want.append(ch.fixed_size_chunks(t, size, overlap))
            cases.append(["recursive_chunks", t, size]); want.append(ch.recursive_chunks(t, size))
        for window in (1, 3, 12, 50):
            for stride in (1, 2, 6, 25, 60):
                cases.append(["sliding_chunks", t, window, stride]); want.append(ch.sliding_window_chunks(t, window, stride))
        for th in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            cases.append(["semantic_chunks", t, th]); want.append(ch.semantic_chunks(t, th))
        cases.append(["sentences", t]); want.append(ch.split_sentences(t))
        for ps, cs in ((400, 100), (200, 80), (60, 60)):
            cases.append(["parent_child", t, ps, cs])
            parents, children = ch.parent_child_chunks(t, ps, cs)
            want.append([pairs(parents), children])
    sents = ch.split_sentences(" ".join(TEXTS))
    for a in sents[:12]:
        for b in sents[:12]:
            cases.append(["jaccard", a, b]); want.append(ch.jaccard(a, b))
    assert len(cases) > 700
    check(cases, want, exact=True)


def test_the_lab_sliders_reproduce_the_demo_printout():
    got = js([["fixed_chunks", ch.SAMPLE, 60, 10], ["recursive_chunks", ch.SAMPLE, 120], ["sliding_chunks", ch.SAMPLE, 12, 6],
              ["semantic_chunks", ch.SAMPLE, 0.1]])
    assert [g["ok"] for g in got] == [ch.fixed_size_chunks(ch.SAMPLE, 60, 10), ch.recursive_chunks(ch.SAMPLE, 120),
                                      ch.sliding_window_chunks(ch.SAMPLE, 12, 6), ch.semantic_chunks(ch.SAMPLE, 0.1)]


# --------------------------------------------------------------- search lab
QUERIES = ["rotate refresh token", "refresh tokens", "caching", "ETag 304", "docker image size", "how long is an access token valid",
           "max-age", "the of and", "", "zzz", "Cache-Control", "small image", "tokens", "token", "cookie rotated", "fresh response seconds"]


def test_search_matches_rag_bm25():
    texts = [d["text"] for d in bm.DOCS]
    cases, want, exact_cases, exact_want = [], [], [], []
    for q in QUERIES:
        cases.append(["bm25_search", q, texts, len(texts), None]); want.append([list(x) for x in bm.BM25(texts).search(q, k=len(texts))])
        cases.append(["bm25_search", q, texts, 2, [0, 3, 4]]); want.append([list(x) for x in bm.BM25(texts).search(q, k=2, allowed={0, 3, 4})])
        exact_cases.append(["bm25_tokenize", q]); exact_want.append(bm.tokenize(q))
        exact_cases.append(["dense", q, texts, None]); exact_want.append([list(x) for x in bm.toy_dense_scores(q, texts)])
        exact_cases.append(["dense", q, texts, [1, 2]]); exact_want.append([list(x) for x in bm.toy_dense_scores(q, texts, {1, 2})])
        for filters, jsf in [(None, None), ({"year": ("gte", 2025)}, {"year": {"op": "gte", "target": 2025}}),
                             ({"team": "platform"}, {"team": "platform"}), ({"team": ["security", "x"], "year": ("lt", 2025)},
                                                                            {"team": ["security", "x"], "year": {"op": "lt", "target": 2025}}),
                             ({"missing": 1}, {"missing": 1})]:
            for k in (3, 5):
                exact_cases.append(["hybrid", q, bm.DOCS, k, jsf, 60]); exact_want.append(bm.hybrid_search(q, bm.DOCS, k=k, filters=filters))
    for term in ["tokens", "refresh", "docker", "nope", "etag"]:
        cases.append(["bm25_idf", term, texts]); want.append(bm.BM25(texts).idf(term))
    check(cases, want, exact=False)                      # BM25 uses log
    check(exact_cases, exact_want, exact=EXACT_SUM)      # trigram cosine and RRF: + * / sqrt only
    shown = js([["bm25_search", q, texts, len(texts), None] for q in QUERIES])
    for q, res in zip(QUERIES, shown):
        assert [[f"{s:.3f}", i] for s, i in res["ok"]] == [[f"{s:.3f}", i] for s, i in bm.BM25(texts).search(q, k=len(texts))], q
    rng = random.Random(4)
    rr = [[rng.sample(range(8), rng.randint(0, 8)) for _ in range(rng.randint(1, 4))] for _ in range(200)]
    check([["rrf", r, k] for r in rr for k in (60, 1)], [bm.reciprocal_rank_fusion(r, k) for r in rr for k in (60, 1)], exact=True)


def test_the_search_demo_printout_is_reproduced():
    texts, q = [d["text"] for d in bm.DOCS], "rotate refresh token"
    got = js([["bm25_search", q, texts, 5, None], ["dense", q, texts, None], ["hybrid", q, bm.DOCS, 3, None, 60],
              ["hybrid", q, bm.DOCS, 3, {"year": {"op": "gte", "target": 2025}}, 60]])
    assert [i for _, i in got[0]["ok"]] == [i for _, i in bm.BM25(texts).search(q)]
    assert [(round(s, 3), i) for s, i in got[1]["ok"]][:3] == [(round(s, 3), i) for s, i in bm.toy_dense_scores(q, texts)][:3]
    assert got[2]["ok"] == bm.hybrid_search(q, bm.DOCS) and got[3]["ok"] == bm.hybrid_search(q, bm.DOCS, filters={"year": ("gte", 2025)})


# --------------------------------------------------------------- the site ships what was tested
def test_site_data_comes_from_the_demo_modules():
    assert DATA["sampling"]["LOGITS"] == sdemo.LOGITS and DATA["sampling"]["TOKENS"] == sdemo.TOKENS
    assert {k: dict(v) for k, v in DATA["decoding"]["NEXT"].items()} == dec.NEXT
    assert DATA["qkv"]["X"] == qkv.X and DATA["qkv"]["W_V"] == qkv.W_V and DATA["toyllm"]["W_OUT"] == e2e.W_OUT
    assert DATA["chunking"]["SAMPLE"] == ch.SAMPLE and DATA["search"]["DOCS"] == bm.DOCS
    json.dumps(DATA)      # must be plain JSON
