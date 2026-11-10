/* JavaScript port of 03-llm-concepts/code/llm_e2e_walkthrough.py: a toy LLM small enough to print.
   Every weight is hand-picked in the Python file and exported by scripts/site_data.py. Nothing is trained. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.attention) { require('./py.js'); require('./attention.js'); }
  var A = AIER.attention, matmul = A.matmul, add = A.add, causalMask = A.causalMask, softmaxRow = A.softmaxRow,
    scaledDotProductAttention = A.scaledDotProductAttention, sinusoidalPositionalEncoding = A.sinusoidalPositionalEncoding;

  /** Build the toy model from the weights exported by scripts/site_data.py (never typed by hand). */
  function ToyLLM(w) {
    this.w = w;
    this.tokenToId = {};
    for (var i = 0; i < w.VOCAB.length; i++) this.tokenToId[w.VOCAB[i]] = i;
  }
  /** tokenize(text): split on spaces, split a trailing "." off a word. */
  ToyLLM.prototype.tokenize = function (text) {
    var pieces = [];
    AIER.py.split(text.toLowerCase()).forEach(function (word) {
      if (word.charAt(word.length - 1) === '.' && AIER.py.cp(word).length > 1) { pieces.push(word.slice(0, -1)); pieces.push('.'); }
      else pieces.push(word);
    });
    return pieces;
  };
  ToyLLM.prototype.unknown = function (tokens) {
    var map = this.tokenToId;
    return tokens.filter(function (t) { return !Object.prototype.hasOwnProperty.call(map, t); });
  };
  ToyLLM.prototype.toIds = function (tokens) {
    var unknown = this.unknown(tokens), map = this.tokenToId;
    if (unknown.length) throw new RangeError('tokens not in the toy vocabulary: ' + unknown.join(', '));
    return tokens.map(function (t) { return map[t]; });
  };
  ToyLLM.prototype.embed = function (ids) { var E = this.w.EMBEDDING; return ids.map(function (i) { return E[i].slice(); }); };
  ToyLLM.prototype.addPosition = function (x) {
    var s = this.w.POS_SCALE, pe = sinusoidalPositionalEncoding(x.length, this.w.D_MODEL);
    return add(x, pe.map(function (row) { return row.map(function (v) { return s * v; }); }));
  };
  ToyLLM.prototype.block = function (x) {
    var w = this.w, q = matmul(x, w.W_Q), k = matmul(x, w.W_K), v = matmul(x, w.W_V);
    var r = scaledDotProductAttention(q, k, v, causalMask(x.length));
    x = add(x, r.output);
    x = add(x, matmul(x, w.W_FF).map(function (row) { return row.map(function (t) { return Math.max(0.0, t); }); }));
    return { x: x, weights: r.weights };
  };
  /** forward(ids): every stage's value. */
  ToyLLM.prototype.forward = function (ids) {
    var st = { ids: ids };
    st.embeddings = this.embed(ids);
    var x = this.addPosition(st.embeddings);
    st.with_position = x; st.block_outputs = []; st.attention_weights = [];
    for (var b = 0; b < this.w.N_BLOCKS; b++) { var r = this.block(x); x = r.x; st.block_outputs.push(x); st.attention_weights.push(r.weights); }
    st.logits = matmul([x[x.length - 1]], this.w.W_OUT)[0];
    st.probs = softmaxRow(st.logits);
    var best = 0;
    for (var i = 1; i < st.probs.length; i++) if (st.probs[i] > st.probs[best]) best = i;
    st.next_id = best;
    return st;
  };
  /** generate(text, max_new_tokens) -> {tokens, steps} */
  ToyLLM.prototype.generate = function (text, maxNew) {
    if (maxNew === undefined) maxNew = 3;
    var ids = this.toIds(this.tokenize(text)), steps = [], V = this.w.VOCAB;
    for (var n = 0; n < maxNew; n++) {
      var s = this.forward(ids); steps.push(s); ids = ids.concat([s.next_id]);
      if (V[s.next_id] === '.') break;
    }
    return { tokens: ids.map(function (i) { return V[i]; }), steps: steps };
  };
  ToyLLM.prototype.parameterCount = function () {
    var w = this.w, n = 0;
    [w.EMBEDDING, w.W_Q, w.W_K, w.W_V, w.W_FF, w.W_OUT].forEach(function (m) { n += m.length * m[0].length; });
    return n;
  };

  AIER.toyllm = { ToyLLM: ToyLLM };
  if (typeof module === 'object' && module.exports) module.exports = AIER.toyllm;
})(typeof self !== 'undefined' ? self : globalThis);
