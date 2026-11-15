/* JavaScript port of 01-rag-foundations/code/rag_vector_math.py and rag_chunkers.py.
   Strings are handled by code point (like Python's len and slicing) and whitespace follows str.isspace(). */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var py = AIER.py, sum = py.sum;

  // ================= rag_vector_math.py =================
  function check(a, b) { if (a.length !== b.length) throw new RangeError('vectors must have the same length (' + a.length + ' != ' + b.length + ')'); }
  function dot(a, b) { check(a, b); return sum(a.map(function (x, i) { return x * b[i]; })); }
  function norm(a) { return Math.sqrt(sum(a.map(function (x) { return x * x; }))); }
  function normalize(a) {
    var n = norm(a);
    if (n === 0) throw new RangeError('cannot normalize a zero vector');
    return a.map(function (x) { return x / n; });
  }
  function cosineSimilarity(a, b) {
    var na = norm(a), nb = norm(b);
    if (na === 0 || nb === 0) return 0.0;
    return dot(a, b) / (na * nb);
  }
  function euclideanDistance(a, b) { check(a, b); return Math.sqrt(sum(a.map(function (x, i) { return Math.pow(x - b[i], 2); }))); }
  /** rank(query, vectors, metric) -> indexes, best first */
  function rank(query, vectors, metric) {
    if (metric === undefined) metric = 'cosine';
    var idx = vectors.map(function (_, i) { return i; }), scores;
    if (metric === 'cosine') { scores = vectors.map(function (v) { return cosineSimilarity(query, v); }); return idx.sort(function (i, j) { return scores[j] - scores[i]; }); }
    if (metric === 'dot') { scores = vectors.map(function (v) { return dot(query, v); }); return idx.sort(function (i, j) { return scores[j] - scores[i]; }); }
    if (metric === 'euclidean') { scores = vectors.map(function (v) { return euclideanDistance(query, v); }); return idx.sort(function (i, j) { return scores[i] - scores[j]; }); }
    throw new RangeError('unknown metric: ' + metric);
  }

  // ================= rag_chunkers.py =================
  var DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', ' ', ''];
  function cpLen(s) { return py.cp(s).length; }

  /** fixed_size_chunks(text, size, overlap) */
  function fixedSizeChunks(text, size, overlap) {
    if (size === undefined) size = 200;
    if (overlap === undefined) overlap = 0;
    if (size <= 0) throw new RangeError('size must be positive');
    if (!(overlap >= 0 && overlap < size)) throw new RangeError('overlap must be >= 0 and smaller than size');
    var cps = py.cp(text), step = size - overlap, chunks = [];
    for (var start = 0; start < cps.length; start += step) {
      chunks.push(cps.slice(start, start + size).join(''));
      if (start + size >= cps.length) break;
    }
    return chunks;
  }
  /** recursive_chunks(text, size, separators) */
  function recursiveChunks(text, size, separators) {
    if (size === undefined) size = 200;
    if (separators === undefined) separators = DEFAULT_SEPARATORS;
    if (cpLen(text) <= size) return py.strip(text) ? [text] : [];
    var sep = separators[0], rest = separators.slice(1), out = [];
    if (sep === '') {
      var cps = py.cp(text);
      for (var i = 0; i < cps.length; i += size) out.push(cps.slice(i, i + size).join(''));
      return out;
    }
    if (text.indexOf(sep) === -1) return recursiveChunks(text, size, rest);
    var parts = text.split(sep);
    var pieces = parts.slice(0, -1).map(function (p) { return p + sep; }).concat([parts[parts.length - 1]]);
    var chunks = [], current = '';
    pieces.forEach(function (piece) {
      if (cpLen(piece) > size) {
        if (current) { chunks.push(current); current = ''; }
        chunks = chunks.concat(recursiveChunks(piece, size, rest));
      } else if (cpLen(current) + cpLen(piece) <= size) {
        current += piece;
      } else { chunks.push(current); current = piece; }
    });
    if (current) chunks.push(current);
    return chunks.filter(function (c) { return py.strip(c); });
  }
  /** sliding_window_chunks(text, window, stride) */
  function slidingWindowChunks(text, win, stride) {
    if (win === undefined) win = 50;
    if (stride === undefined) stride = 25;
    if (win <= 0 || stride <= 0) throw new RangeError('window and stride must be positive');
    var words = py.split(text), chunks = [];
    if (!words.length) return [];
    for (var start = 0; start < words.length; start += stride) {
      chunks.push(words.slice(start, start + win).join(' '));
      if (start + win >= words.length) break;
    }
    return chunks;
  }
  /** split_sentences(text): re.split(r"(?<=[.!?])\s+", text.strip()), empties dropped. No lookbehind needed. */
  function splitSentences(text) {
    var t = py.strip(text), out = [], cur = '', i = 0;
    while (i < t.length) {
      var ch = t.charAt(i);
      if (py.isSpace(ch) && i > 0 && '.!?'.indexOf(t.charAt(i - 1)) !== -1) {
        while (i < t.length && py.isSpace(t.charAt(i))) i++;
        out.push(cur); cur = '';
      } else { cur += ch; i++; }
    }
    out.push(cur);
    return out.filter(function (s) { return s; });
  }
  function wordSet(text) { return new Set(text.toLowerCase().match(/[a-z0-9]+/g) || []); }
  /** jaccard(a, b) */
  function jaccard(a, b) {
    var wa = wordSet(a), wb = wordSet(b);
    if (!wa.size || !wb.size) return 0.0;
    var inter = 0; wa.forEach(function (w) { if (wb.has(w)) inter++; });
    return inter / (wa.size + wb.size - inter);
  }
  /** semantic_chunks(text, threshold) with the default word-overlap similarity */
  function semanticChunks(text, threshold) {
    if (threshold === undefined) threshold = 0.1;
    var sentences = splitSentences(text);
    if (!sentences.length) return [];
    var chunks = [], current = [sentences[0]];
    for (var i = 1; i < sentences.length; i++) {
      if (jaccard(sentences[i - 1], sentences[i]) < threshold) { chunks.push(current.join(' ')); current = [sentences[i]]; }
      else current.push(sentences[i]);
    }
    chunks.push(current.join(' '));
    return chunks;
  }
  /** parent_child_chunks(text, parent_size, child_size) -> {parents: [[id, text]], children: [{id, parent_id, text}]} */
  function parentChildChunks(text, parentSize, childSize) {
    if (parentSize === undefined) parentSize = 400;
    if (childSize === undefined) childSize = 100;
    if (childSize > parentSize) throw new RangeError('child_size must be <= parent_size');
    var parents = [], children = [];
    recursiveChunks(text, parentSize).forEach(function (parent, p) {
      var pid = 'p' + p; parents.push([pid, parent]);
      recursiveChunks(parent, childSize).forEach(function (child, c) { children.push({ id: pid + '-c' + c, parent_id: pid, text: child }); });
    });
    return { parents: parents, children: children };
  }

  AIER.rag = { dot: dot, norm: norm, normalize: normalize, cosineSimilarity: cosineSimilarity, euclideanDistance: euclideanDistance, rank: rank,
    fixedSizeChunks: fixedSizeChunks, recursiveChunks: recursiveChunks, slidingWindowChunks: slidingWindowChunks,
    splitSentences: splitSentences, jaccard: jaccard, semanticChunks: semanticChunks, parentChildChunks: parentChildChunks };
  if (typeof module === 'object' && module.exports) module.exports = AIER.rag;
})(typeof self !== 'undefined' ? self : globalThis);
