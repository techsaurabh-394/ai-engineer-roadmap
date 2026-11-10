/* JavaScript port of 01-rag-foundations/code/rag_bm25.py: BM25, the toy "dense" search
   (character trigrams, NOT a real embedding model), reciprocal rank fusion and metadata filters. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var py = AIER.py, sum = py.sum;

  function counter(items) { var m = new Map(); items.forEach(function (x) { m.set(x, (m.get(x) || 0) + 1); }); return m; }
  function makeTokenizer(stopWords) {
    var stop = new Set(stopWords);
    return function (text) { return (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter(function (w) { return !stop.has(w); }); };
  }
  /** class BM25(texts, k1=1.5, b=0.75). `stop` is the STOP list exported from the Python module. */
  function BM25(texts, stop, k1, b) {
    this.k1 = k1 === undefined ? 1.5 : k1; this.b = b === undefined ? 0.75 : b;
    this.tokenize = makeTokenizer(stop);
    this.docs = texts.map(this.tokenize);
    this.tfs = this.docs.map(counter);
    this.lengths = this.docs.map(function (d) { return d.length; });
    this.n = this.docs.length;
    this.avgdl = this.n ? sum(this.lengths) / this.n : 0.0;
    var df = new Map();
    this.tfs.forEach(function (tf) { tf.forEach(function (_, t) { df.set(t, (df.get(t) || 0) + 1); }); });
    this.df = df;
  }
  BM25.prototype.idf = function (term) { var d = this.df.get(term) || 0; return Math.log((this.n - d + 0.5) / (d + 0.5) + 1); };
  /** score(query, i); also returns the per-term parts when `detail` is passed (an array to fill). */
  BM25.prototype.score = function (query, i, detail) {
    var tf = this.tfs[i], length = this.lengths[i], s = 0.0, self = this;
    this.tokenize(query).forEach(function (t) {
      var f = tf.get(t) || 0;
      if (f === 0) return;
      var denom = f + self.k1 * (1 - self.b + self.b * length / (self.avgdl || 1));
      var part = self.idf(t) * f * (self.k1 + 1) / denom;
      s += part;
      if (detail) detail.push({ term: t, tf: f, idf: self.idf(t), part: part });
    });
    return s;
  };
  /** search(query, k, allowed) -> [[score, index]] best first */
  BM25.prototype.search = function (query, k, allowed) {
    if (k === undefined) k = 5;
    var ids = [], self = this, i;
    if (allowed === undefined || allowed === null) for (i = 0; i < this.n; i++) ids.push(i);
    else ids = Array.from(allowed).sort(function (a, b) { return a - b; });
    var scored = ids.map(function (j) { return [self.score(query, j), j]; }).filter(function (x) { return x[0] > 0; });
    scored.sort(function (a, b) { return (b[0] - a[0]) || (a[1] - b[1]); });
    return scored.slice(0, k);
  };
  function trigrams(text) {
    var out = [];
    (text.toLowerCase().match(/[a-z0-9]+/g) || []).forEach(function (w) {
      var cps = py.cp('#' + w + '#');
      for (var i = 0; i < cps.length - 2; i++) out.push(cps.slice(i, i + 3).join(''));
    });
    return counter(out);
  }
  /** toy_dense_scores(query, texts, allowed) -> [[score, index]] best first (character trigrams, NOT embeddings) */
  function toyDenseScores(query, texts, allowed) {
    var q = trigrams(query), qv = [];
    q.forEach(function (v) { qv.push(v * v); });
    var qn = Math.sqrt(sum(qv)) || 1.0, ids = [], i;
    if (allowed === undefined || allowed === null) for (i = 0; i < texts.length; i++) ids.push(i);
    else ids = Array.from(allowed).sort(function (a, b) { return a - b; });
    var out = ids.map(function (j) {
      var d = trigrams(texts[j]), dv = [], terms = [];
      d.forEach(function (v) { dv.push(v * v); });
      var dn = Math.sqrt(sum(dv)) || 1.0;
      q.forEach(function (v, g) { terms.push(v * (d.get(g) || 0)); });
      return [sum(terms) / (qn * dn), j];
    });
    out.sort(function (a, b) { return (b[0] - a[0]) || (a[1] - b[1]); });
    return out;
  }
  /** reciprocal_rank_fusion(rankings, k=60) -> ids best first. Pass `detail` (a Map) to get the scores. */
  function reciprocalRankFusion(rankings, k, detail) {
    if (k === undefined) k = 60;
    var scores = detail || new Map();
    rankings.forEach(function (ranking) {
      ranking.forEach(function (id, r) { scores.set(id, (scores.has(id) ? scores.get(id) : 0.0) + 1.0 / (k + (r + 1))); });
    });
    return Array.from(scores.keys()).sort(function (a, b) { return (scores.get(b) - scores.get(a)) || (a - b); });
  }
  /** matches(meta, filters): value, [list], or ["op", target] written as {op, target}. */
  function matches(meta, filters) {
    return Object.keys(filters).every(function (field) {
      if (!Object.prototype.hasOwnProperty.call(meta, field)) return false;
      var value = meta[field], cond = filters[field];
      if (cond && typeof cond === 'object' && !Array.isArray(cond)) {
        var ok = { gte: value >= cond.target, lte: value <= cond.target, gt: value > cond.target, lt: value < cond.target }[cond.op];
        if (ok === undefined) throw new RangeError('unknown operator ' + cond.op);
        return ok;
      }
      if (Array.isArray(cond)) return cond.indexOf(value) !== -1;
      return value === cond;
    });
  }
  /** hybrid_search(query, docs, k, filters, rrf_k) -> {ids, sparse, dense, allowed} */
  function hybridSearch(query, docs, stop, k, filters, rrfK) {
    if (k === undefined) k = 3;
    if (rrfK === undefined) rrfK = 60;
    var texts = docs.map(function (d) { return d.text; }), allowed = null;
    if (filters && Object.keys(filters).length) {
      allowed = [];
      docs.forEach(function (d, i) { if (matches(d.meta || {}, filters)) allowed.push(i); });
      if (!allowed.length) return { ids: [], sparse: [], dense: [], allowed: allowed };
    }
    var sparse = new BM25(texts, stop).search(query, texts.length, allowed).map(function (x) { return x[1]; });
    var dense = toyDenseScores(query, texts, allowed).filter(function (x) { return x[0] > 0; }).map(function (x) { return x[1]; });
    return { ids: reciprocalRankFusion([sparse, dense], rrfK).slice(0, k), sparse: sparse, dense: dense, allowed: allowed };
  }

  AIER.search = { BM25: BM25, makeTokenizer: makeTokenizer, trigrams: trigrams, toyDenseScores: toyDenseScores,
    reciprocalRankFusion: reciprocalRankFusion, matches: matches, hybridSearch: hybridSearch };
  if (typeof module === 'object' && module.exports) module.exports = AIER.search;
})(typeof self !== 'undefined' ? self : globalThis);
