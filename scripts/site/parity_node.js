#!/usr/bin/env node
/* Runs the browser maths (scripts/site/core/*.js) under node for scripts/test_site_parity.py.
   stdin:  {"data": <site data>, "cases": [[op, args...], ...]}     stdout: [result, ...]
   Floats travel as JSON numbers (shortest round-trip form), so they arrive bit-for-bit. */
'use strict';
const path = require('path');
const fs = require('fs');
for (const f of ['py', 'sampling', 'attention', 'toyllm', 'rag', 'search']) {
  const file = path.join(__dirname, 'core', f + '.js');
  if (fs.existsSync(file)) require(file);
}
const { py, sampling: S, attention: A, toyllm: T, rag: R, search: Q } = globalThis.AIER;

const enc = v => (v === -Infinity ? '-inf' : v === Infinity ? 'inf' : Array.isArray(v) ? v.map(enc)
  : v instanceof Map ? Array.from(v.entries()).map(enc)
  : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, enc(x)])) : v);

function run(D, op, a) {
  const table = D && D.decoding ? D.decoding.NEXT : null;
  const stop = D && D.search ? D.search.STOP : null;
  switch (op) {
    // formatting and RNG
    case 'fixed': return py.fixed(a[0], a[1]);
    case 'repr': return py.repr(a[0]);
    case 'round': return py.round(a[0], a[1]);
    case 'sum': return py.sum(a[0]);
    case 'split': return py.split(a[0]);
    case 'strip': return py.strip(a[0]);
    case 'random': { const r = new py.Random(a[0]); return Array.from({ length: a[1] }, () => r.random()); }
    case 'choices': { const r = new py.Random(a[0]); return Array.from({ length: a[3] }, () => r.choice(a[1], a[2])); }
    // sampling_demo.py
    case 'softmax': return S.softmax(a[0], a[1]);
    case 'demo_top_p': return S.demoTopP(a[0], a[1], a[2]);
    // llm_decoding.py
    case 'top_k': return S.topK(a[0], a[1]);
    case 'top_p': return S.topP(a[0], a[1]);
    case 'apply_temperature': return S.applyTemperature(a[0], a[1]);
    case 'chain_logits': return S.filterChain(a[0].map((t, i) => [t, S.softmax(a[1], a[2])[i]]), a[3], a[4]);
    case 'chain_table': return S.filterChain(S.applyTemperature(S.nextProbs(table, a[0]), a[1]), a[2], a[3]);
    case 'greedy': return S.greedy(table, a[0], a[1]);
    case 'sample': return S.sample(table, { start: a[0], maxLen: a[1], k: a[2], p: a[3], temperature: a[4], rng: new py.Random(a[5]) });
    case 'beam': return S.beamSearch(table, a[0], a[1], a[2]);
    case 'logprob': return S.sequenceLogprob(table, a[0], a[1]);
    case 'count_kv': return S.countKv(a[0], a[1]);
    // llm_attention.py / llm_qkv.py
    case 'matmul': return A.matmul(a[0], a[1]);
    case 'sdpa': { const r = A.scaledDotProductAttention(a[0], a[1], a[2], a[3] ? A.causalMask(a[0].length) : null); return [r.output, r.weights]; }
    case 'mha': { const r = A.multiHeadAttention(a[0], a[1], a[2]); return [r.output, r.weights]; }
    case 'pos_enc': return A.sinusoidalPositionalEncoding(a[0], a[1]);
    case 'trace': return A.attentionWithTrace(a[0], a[1], a[2], a[3], a[4]);
    case 'param_count': return A.qkvParamCount(a[0], a[1], a[2]);
    // llm_e2e_walkthrough.py
    case 'toy_tokenize': return new T.ToyLLM(D.toyllm).tokenize(a[0]);
    case 'toy_forward': { const m = new T.ToyLLM(D.toyllm); return m.forward(m.toIds(m.tokenize(a[0]))); }
    case 'toy_generate': { const m = new T.ToyLLM(D.toyllm); const g = m.generate(a[0], a[1]); return [g.tokens, g.steps.map(s => [s.next_id, s.probs])]; }
    case 'toy_params': return new T.ToyLLM(D.toyllm).parameterCount();
    // rag_vector_math.py
    case 'dot': return R.dot(a[0], a[1]);
    case 'norm': return R.norm(a[0]);
    case 'normalize': return R.normalize(a[0]);
    case 'cosine': return R.cosineSimilarity(a[0], a[1]);
    case 'euclid': return R.euclideanDistance(a[0], a[1]);
    case 'rank': return R.rank(a[0], a[1], a[2]);
    // rag_chunkers.py
    case 'fixed_chunks': return R.fixedSizeChunks(a[0], a[1], a[2]);
    case 'recursive_chunks': return R.recursiveChunks(a[0], a[1]);
    case 'sliding_chunks': return R.slidingWindowChunks(a[0], a[1], a[2]);
    case 'sentences': return R.splitSentences(a[0]);
    case 'jaccard': return R.jaccard(a[0], a[1]);
    case 'semantic_chunks': return R.semanticChunks(a[0], a[1]);
    case 'parent_child': { const r = R.parentChildChunks(a[0], a[1], a[2]); return [r.parents, r.children]; }
    // rag_bm25.py
    case 'bm25_tokenize': return Q.makeTokenizer(stop)(a[0]);
    case 'bm25_search': return new Q.BM25(a[1], stop).search(a[0], a[2], a[3]);
    case 'bm25_idf': return new Q.BM25(a[1], stop).idf(a[0]);
    case 'dense': return Q.toyDenseScores(a[0], a[1], a[2]);
    case 'rrf': return Q.reciprocalRankFusion(a[0], a[1]);
    case 'hybrid': return Q.hybridSearch(a[0], a[1], stop, a[2], a[3], a[4]).ids;
    default: throw new Error('unknown op ' + op);
  }
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', c => { input += c; });
process.stdin.on('end', () => {
  const { data, cases } = JSON.parse(input);
  const out = cases.map(([op, ...args]) => {
    try { return { ok: enc(run(data, op, args)) }; } catch (e) { return { err: e.constructor.name + ': ' + e.message }; }
  });
  process.stdout.write(JSON.stringify(out));
});
