/* JavaScript port of 03-llm-concepts/code/sampling_demo.py and llm_decoding.py.
   Same order of operations as the Python, so the numbers match (see scripts/test_site_parity.py).
   Distributions are arrays of [token, probability] pairs: like a Python dict they keep insertion order. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var END = '<end>';

  // ---------------- sampling_demo.py ----------------
  /** softmax(logits, temperature) -> list of probabilities */
  function softmax(logits, temperature) {
    if (temperature === undefined) temperature = 1.0;
    if (!(temperature > 0)) throw new RangeError('temperature must be > 0 (use a tiny value for near-greedy)');
    var scaled = logits.map(function (x) { return x / temperature; });
    var m = Math.max.apply(null, scaled);
    var exps = scaled.map(function (x) { return Math.exp(x - m); });
    var total = AIER.py.sum(exps);
    return exps.map(function (e) { return e / total; });
  }
  /** sampling_demo.top_p_filter(probs, p): indexes kept + renormalised pairs. Note: no epsilon here. */
  function demoTopP(tokens, probs, p) {
    var order = probs.map(function (_, i) { return i; }).sort(function (a, b) { return probs[b] - probs[a]; });
    var kept = [], total = 0.0;
    for (var n = 0; n < order.length; n++) { kept.push(order[n]); total += probs[order[n]]; if (total >= p) break; }
    var norm = AIER.py.sum(kept.map(function (i) { return probs[i]; }));
    return kept.map(function (i) { return [tokens[i], probs[i] / norm]; });
  }

  // ---------------- llm_decoding.py ----------------
  function nextProbs(table, prev) {
    return Object.prototype.hasOwnProperty.call(table, prev) ? table[prev] : [[END, 1.0]];
  }
  function byProbDesc(pairs) { return pairs.slice().sort(function (a, b) { return b[1] - a[1]; }); }
  function sumP(pairs) { return AIER.py.sum(pairs.map(function (kv) { return kv[1]; })); }

  /** top_k_filter(probs, k) */
  function topK(pairs, k) {
    if (k < 1) throw new RangeError('k must be >= 1');
    var kept = byProbDesc(pairs).slice(0, k), total = sumP(kept);
    return kept.map(function (kv) { return [kv[0], kv[1] / total]; });
  }
  /** top_p_filter(probs, p) (nucleus) */
  function topP(pairs, p) {
    if (!(p > 0 && p <= 1)) throw new RangeError('p must be in (0, 1]');
    var kept = [], mass = 0.0, sorted = byProbDesc(pairs);
    for (var i = 0; i < sorted.length; i++) { kept.push(sorted[i]); mass += sorted[i][1]; if (mass >= p - 1e-12) break; }
    var total = sumP(kept);
    return kept.map(function (kv) { return [kv[0], kv[1] / total]; });
  }
  /** apply_temperature(probs, temperature) */
  function applyTemperature(pairs, temperature) {
    if (!(temperature > 0)) throw new RangeError('temperature must be > 0');
    var logits = pairs.filter(function (kv) { return kv[1] > 0; }).map(function (kv) { return [kv[0], Math.log(kv[1]) / temperature]; });
    var m = Math.max.apply(null, logits.map(function (kv) { return kv[1]; }));
    var exps = logits.map(function (kv) { return [kv[0], Math.exp(kv[1] - m)]; });
    var z = sumP(exps);
    return exps.map(function (kv) { return [kv[0], kv[1] / z]; });
  }
  /** The filter chain inside sample(): temperature (done by the caller), then top-k, then top-p. */
  function filterChain(pairs, k, p) {
    if (k !== null && k !== undefined) pairs = topK(pairs, k);
    if (p !== null && p !== undefined) pairs = topP(pairs, p);
    return pairs;
  }
  /** greedy(start, max_len) */
  function greedy(table, start, maxLen) {
    if (start === undefined) start = '<start>';
    if (maxLen === undefined) maxLen = 10;
    var seq = [start];
    while (seq.length < maxLen) {
      var probs = nextProbs(table, seq[seq.length - 1]), tok = probs[0][0], best = probs[0][1];
      for (var i = 1; i < probs.length; i++) if (probs[i][1] > best) { best = probs[i][1]; tok = probs[i][0]; }
      seq.push(tok);
      if (tok === END) break;
    }
    return seq.slice(1);
  }
  /** sample(start, max_len, k, p, temperature, rng) where rng is an AIER.py.Random */
  function sample(table, opts) {
    opts = opts || {};
    var rng = opts.rng || new AIER.py.Random(0);
    var maxLen = opts.maxLen === undefined ? 10 : opts.maxLen;
    var seq = [opts.start === undefined ? '<start>' : opts.start];
    while (seq.length < maxLen) {
      var probs = applyTemperature(nextProbs(table, seq[seq.length - 1]), opts.temperature === undefined ? 1.0 : opts.temperature);
      probs = filterChain(probs, opts.k, opts.p);
      var tok = rng.choice(probs.map(function (kv) { return kv[0]; }), probs.map(function (kv) { return kv[1]; }));
      seq.push(tok);
      if (tok === END) break;
    }
    return seq.slice(1);
  }
  /** sequence_logprob(tokens, start) */
  function sequenceLogprob(table, tokens, start) {
    var prev = start === undefined ? '<start>' : start, lp = 0.0;
    tokens.forEach(function (t) {
      var row = nextProbs(table, prev), hit = null;
      row.forEach(function (kv) { if (kv[0] === t) hit = kv[1]; });
      if (hit === null) throw new RangeError('token not reachable: ' + t);
      lp += Math.log(hit); prev = t;
    });
    return lp;
  }
  /** beam_search(beam_width, start, max_len) -> [tokens, logprob] */
  function beamSearch(table, beamWidth, start, maxLen) {
    if (beamWidth === undefined) beamWidth = 2;
    if (start === undefined) start = '<start>';
    if (maxLen === undefined) maxLen = 10;
    var beams = [[[start], 0.0]], finished = [];
    for (var step = 0; step < maxLen; step++) {
      var candidates = [];
      beams.forEach(function (b) {
        nextProbs(table, b[0][b[0].length - 1]).forEach(function (kv) {
          var cand = [b[0].concat([kv[0]]), b[1] + Math.log(kv[1])];
          (kv[0] === END ? finished : candidates).push(cand);
        });
      });
      if (!candidates.length) break;
      beams = candidates.slice().sort(function (a, b) { return b[1] - a[1]; }).slice(0, beamWidth);
    }
    finished = finished.concat(beams);
    var best = finished[0];
    for (var i = 1; i < finished.length; i++) if (finished[i][1] > best[1]) best = finished[i];
    return [best[0].slice(1), best[1]];
  }
  /** count_kv_computations(num_tokens, use_cache) */
  function countKv(numTokens, useCache) {
    var computed = 0;
    for (var t = 1; t <= numTokens; t++) computed += useCache ? 1 : t;
    return computed;
  }

  AIER.sampling = { END: END, softmax: softmax, demoTopP: demoTopP, nextProbs: nextProbs, topK: topK, topP: topP,
    applyTemperature: applyTemperature, filterChain: filterChain, greedy: greedy, sample: sample,
    sequenceLogprob: sequenceLogprob, beamSearch: beamSearch, countKv: countKv };
  if (typeof module === 'object' && module.exports) module.exports = AIER.sampling;
})(typeof self !== 'undefined' ? self : globalThis);
