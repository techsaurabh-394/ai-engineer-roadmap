/* JavaScript port of 01-rag-foundations/code/rag_pipeline.py: chunk, TF-IDF embed, cosine search, rerank,
   context, extractive answer. No DOM. Tested against the Python by scripts/test_site_parity.py. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var py = AIER.py, sum = py.sum;

  function counter(items) { var m = new Map(); items.forEach(function (x) { m.set(x, (m.get(x) || 0) + 1); }); return m; }
  function stem(w) { return w.length > 3 && /s$/.test(w) && !/ss$/.test(w) ? w.slice(0, -1) : w; }
  function makeTokens(stop) {
    var S = new Set(stop);
    return function (text) { return (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter(function (w) { return !S.has(w); }).map(stem); };
  }
  function sentences(text) {            // re.split(r"(?<=[.!?])\s+", text.strip())
    var t = py.strip(text), out = [], cur = '', i, ch;
    if (t === '') return [''];
    for (i = 0; i < t.length; i++) {
      ch = t.charAt(i);
      if (py.isSpace(ch) && /[.!?]/.test(t.charAt(i - 1) || '')) {
        while (i + 1 < t.length && py.isSpace(t.charAt(i + 1))) i++;
        out.push(cur); cur = '';
      } else cur += ch;
    }
    out.push(cur);
    return out;
  }
  function chunk(text, size) {
    var chunks = [], cur = '';
    sentences(text).forEach(function (s) {
      if (cur && cur.length + s.length + 1 > size) { chunks.push(cur); cur = s; } else cur = py.strip(cur + ' ' + s);
    });
    if (cur) chunks.push(cur);
    return chunks;
  }
  function embed(toks, index) {
    var counts = counter(toks);
    var v = index.vocab.map(function (t) { return counts.get(t) ? (1 + Math.log(counts.get(t))) * index.idf[t] : 0.0; });
    var norm = Math.sqrt(sum(v.map(function (x) { return x * x; }))) || 1.0;
    return v.map(function (x) { return x / norm; });
  }
  /** build_index(docs, stop, size). docs: [{name, text}] */
  function buildIndex(docs, stop, size) {
    var tokens = makeTokens(stop), items = [];
    docs.forEach(function (d) { chunk(d.text, size).forEach(function (c) { items.push({ doc: d.name, text: c }); }); });
    var df = new Map();
    items.forEach(function (it, i) {
      it.id = i; it.tokens = tokens(it.text);
      new Set(it.tokens).forEach(function (t) { df.set(t, (df.get(t) || 0) + 1); });
    });
    var vocab = Array.from(df.keys()).sort(), n = items.length, idf = {};
    vocab.forEach(function (t) { idf[t] = Math.log((n + 1) / (df.get(t) + 1)) + 1; });
    var index = { items: items, vocab: vocab, idf: idf, tokens: tokens, stop: stop };
    items.forEach(function (it) { it.vector = embed(it.tokens, index); });
    return index;
  }
  function cosine(a, b) { return sum(a.map(function (x, i) { return x * b[i]; })); }
  function search(qv, index, k) {
    var scored = index.items.map(function (it) { return [cosine(qv, it.vector), it.id]; }).filter(function (x) { return x[0] > 0; });
    scored.sort(function (a, b) { return (b[0] - a[0]) || (a[1] - b[1]); });
    return scored.slice(0, k);
  }
  function coverage(terms, chunkTokens) {
    var uniq = Array.from(new Set(terms)).sort(), have = new Set(chunkTokens);
    var hit = 0; uniq.forEach(function (t) { if (have.has(t)) hit++; });
    return uniq.length ? hit / uniq.length : 0.0;
  }
  function rerank(terms, index, hits, weight) {
    if (weight === undefined) weight = 0.5;
    var out = hits.map(function (h) {
      var cov = coverage(terms, index.items[h[1]].tokens);
      return [weight * cov + (1 - weight) * h[0], h[1], cov, h[0]];
    });
    out.sort(function (a, b) { return (b[0] - a[0]) || (a[1] - b[1]); });
    return out;
  }
  function buildContext(index, ids) {
    return ids.map(function (i, n) { return '[' + (n + 1) + '] (' + index.items[i].doc + ') ' + index.items[i].text; }).join('\n');
  }
  function buildPrompt(question, context) {
    return 'Answer ONLY from the numbered sources and cite them like [1]. If they do not contain the answer, say so.\n\nSources:\n' +
      context + '\n\nQuestion: ' + question;
  }
  function extractiveAnswer(terms, index, ids) {
    var best = null, bestKey = null;
    ids.forEach(function (id, n0) {
      var n = n0 + 1;
      sentences(index.items[id].text).forEach(function (s) {
        var st = new Set(index.tokens(s)), overlap = 0;
        new Set(terms).forEach(function (t) { if (st.has(t)) overlap++; });
        var key = [overlap, -n];
        if (overlap && (bestKey === null || key[0] > bestKey[0] || (key[0] === bestKey[0] && key[1] > bestKey[1]))) { best = s + ' [' + n + ']'; bestKey = key; }
      });
    });
    return best || 'I could not find that in the documents.';
  }
  /** run(query, docs, stop, size, k, top): every stage's data, like rag_pipeline.run() */
  function run(query, docs, stop, size, k, top) {
    var index = buildIndex(docs, stop, size), words = query.toLowerCase().match(/[a-z0-9]+/g) || [], terms = index.tokens(query);
    var qv = embed(terms, index), hits = search(qv, index, k), ranked = rerank(terms, index, hits);
    var ids = ranked.slice(0, top).map(function (r) { return r[1]; });
    var context = buildContext(index, ids);
    return { index: index, words: words, terms: terms, queryVector: qv, hits: hits, ranked: ranked, contextIds: ids, context: context,
      prompt: buildPrompt(query, context), answer: ids.length ? extractiveAnswer(terms, index, ids) : 'I could not find that in the documents.' };
  }
  AIER.ragpipe = { makeTokens: makeTokens, chunk: chunk, buildIndex: buildIndex, embed: embed, cosine: cosine, search: search,
    coverage: coverage, rerank: rerank, buildContext: buildContext, buildPrompt: buildPrompt, extractiveAnswer: extractiveAnswer, run: run };
  if (typeof module === 'object' && module.exports) module.exports = AIER.ragpipe;
})(typeof self !== 'undefined' ? self : globalThis);
