/* JavaScript port of 03-llm-concepts/code/llm_attention.py and llm_qkv.py.
   Matrices are arrays of rows. -Infinity stands for Python's float("-inf"). */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var sum = AIER.py.sum, NEG_INF = -Infinity;

  // ---------------- llm_attention.py ----------------
  function matmul(a, b) {
    if (!a.length || !b.length || a[0].length !== b.length) throw new RangeError('shape mismatch in matmul');
    return a.map(function (row) {
      return b[0].map(function (_, j) {
        var terms = [];
        for (var t = 0; t < b.length; t++) terms.push(row[t] * b[t][j]);
        return sum(terms);
      });
    });
  }
  function transpose(m) { return m[0].map(function (_, j) { return m.map(function (row) { return row[j]; }); }); }
  function softmaxRow(row) {
    var m = Math.max.apply(null, row);
    if (m === NEG_INF) throw new RangeError('a row cannot be fully masked');
    var exps = row.map(function (x) { return x !== NEG_INF ? Math.exp(x - m) : 0.0; });
    var total = sum(exps);
    return exps.map(function (e) { return e / total; });
  }
  function causalMask(n) {
    var out = [];
    for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < n; j++) r.push(j <= i ? 0.0 : NEG_INF); out.push(r); }
    return out;
  }
  /** scaled_dot_product_attention(q, k, v, mask) -> {output, weights} */
  function scaledDotProductAttention(q, k, v, mask) {
    var dK = k[0].length, scale = Math.sqrt(dK);
    var scores = matmul(q, transpose(k)).map(function (row) { return row.map(function (s) { return s / scale; }); });
    if (mask) scores = scores.map(function (row, i) { return row.map(function (s, j) { return s + mask[i][j]; }); });
    var weights = scores.map(softmaxRow);
    return { output: matmul(weights, v), weights: weights };
  }
  function splitHeads(x, numHeads) {
    var dModel = x[0].length;
    if (dModel % numHeads) throw new RangeError('d_model must be divisible by num_heads');
    var dHead = dModel / numHeads, heads = [];
    for (var h = 0; h < numHeads; h++) heads.push(x.map(function (row) { return row.slice(h * dHead, (h + 1) * dHead); }));
    return heads;
  }
  function mergeHeads(heads) {
    return heads[0].map(function (_, i) { var r = []; heads.forEach(function (h) { r = r.concat(h[i]); }); return r; });
  }
  /** multi_head_attention(x, num_heads, causal) with Q = K = V = x */
  function multiHeadAttention(x, numHeads, causal) {
    var mask = causal ? causalMask(x.length) : null, outs = [], all = [];
    splitHeads(x, numHeads).forEach(function (h) {
      var r = scaledDotProductAttention(h, h, h, mask); outs.push(r.output); all.push(r.weights);
    });
    return { output: mergeHeads(outs), weights: all };
  }
  function sinusoidalPositionalEncoding(seqLen, dModel) {
    var pe = [];
    for (var pos = 0; pos < seqLen; pos++) {
      var row = [];
      for (var j = 0; j < dModel; j++) {
        var i = Math.floor(j / 2), angle = pos / Math.pow(10000, 2 * i / dModel);
        row.push(j % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
      }
      pe.push(row);
    }
    return pe;
  }
  function add(a, b) { return a.map(function (ra, i) { return ra.map(function (x, j) { return x + b[i][j]; }); }); }

  // ---------------- llm_qkv.py ----------------
  /** attention_with_trace(x, w_q, w_k, w_v, causal): every intermediate result. */
  function attentionWithTrace(x, wQ, wK, wV, causal) {
    if (causal === undefined) causal = true;
    var q = matmul(x, wQ), k = matmul(x, wK), v = matmul(x, wV);
    var dK = k[0].length, n = x.length;
    var scores = matmul(q, transpose(k));
    var scaled = scores.map(function (row) { return row.map(function (s) { return s / Math.sqrt(dK); }); });
    var mask = causal ? causalMask(n) : x.map(function () { return x.map(function () { return 0.0; }); });
    var masked = scaled.map(function (sr, i) { return sr.map(function (s, j) { return s + mask[i][j]; }); });
    var weights = masked.map(softmaxRow);
    return { x: x, q: q, k: k, v: v, scores: scores, scaled: scaled, mask: mask, masked: masked,
      weights: weights, output: matmul(weights, v), d_k: dK };
  }
  /** qkv_param_count(d_model, d_k=None, bias=False) */
  function qkvParamCount(dModel, dK, bias) {
    if (dK === undefined || dK === null) dK = dModel;
    return 3 * (dModel * dK + (bias ? dK : 0));
  }

  AIER.attention = { NEG_INF: NEG_INF, matmul: matmul, transpose: transpose, softmaxRow: softmaxRow, causalMask: causalMask,
    scaledDotProductAttention: scaledDotProductAttention, splitHeads: splitHeads, mergeHeads: mergeHeads,
    multiHeadAttention: multiHeadAttention, sinusoidalPositionalEncoding: sinusoidalPositionalEncoding, add: add,
    attentionWithTrace: attentionWithTrace, qkvParamCount: qkvParamCount };
  if (typeof module === 'object' && module.exports) module.exports = AIER.attention;
})(typeof self !== 'undefined' ? self : globalThis);
