/* Python-compatible helpers so the browser shows exactly what the repo's Python demos print.
   No DOM. Loaded in the browser (window.AIER.py) and by node for the parity tests. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});

  // ---- str.isspace() / re \s for Python str patterns (not the same set as JS \s) ----
  var WS = '\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
  var WS_RUN = new RegExp('[' + WS + ']+', 'g');
  var WS_LEAD = new RegExp('^[' + WS + ']+');
  var WS_TRAIL = new RegExp('[' + WS + ']+$');
  var WS_ONE = new RegExp('[' + WS + ']');

  function isSpace(ch) { return WS_ONE.test(ch); }
  /** text.strip() */
  function strip(s) { return s.replace(WS_LEAD, '').replace(WS_TRAIL, ''); }
  /** text.split() with no argument: split on runs of whitespace, drop empties. */
  function split(s) { var t = strip(s); return t === '' ? [] : t.split(WS_RUN); }
  /** len(text): Python counts code points, JS .length counts UTF-16 units. */
  function cp(s) { return Array.from(s); }

  // ---- number formatting ----
  /** f"{x:.{d}f}". JS toFixed rounds exact ties up; Python rounds them to even (0.0625 -> '0.062'). */
  function fixed(x, d) {
    if (!isFinite(x)) return x !== x ? 'nan' : (x > 0 ? 'inf' : '-inf');
    if (d > 12) throw new RangeError('fixed() supports at most 12 decimals');
    var neg = x < 0 || (x === 0 && 1 / x < 0);
    var a = Math.abs(x);
    if (a >= 1e21) throw new RangeError('fixed() supports |x| < 1e21');
    var out = a.toFixed(d);
    var exact = a.toFixed(100);                       // exact decimal expansion of the double
    var tail = exact.slice(exact.indexOf('.') + 1 + d);
    if (/^50*$/.test(tail)) {                         // an exact tie: Python picks the even neighbour
      var down = exact.slice(0, exact.indexOf('.') + 1 + d);
      if (d === 0) down = down.slice(0, -1);
      var last = down.charCodeAt(down.length - 1) - 48;
      if (last % 2 === 0) out = down;                 // toFixed went up to the odd..even? keep the even one
    }
    return (neg ? '-' : '') + out;
  }
  /** f"{x:{width}.{d}f}" (right aligned). */
  function fixedw(x, width, d) { return fixed(x, d).padStart(width, ' '); }
  /** f"{x:+.{d}f}" */
  function signed(x, d) { var s = fixed(x, d); return s.charAt(0) === '-' ? s : '+' + s; }
  /** round(x, d): a float, rounded half to even on the exact binary value. */
  function round(x, d) { return Number(fixed(x, d)); }
  /** repr(float): shortest digits that round-trip, in Python's layout ('1.0', '1e-05', '0.1'). */
  function repr(x) {
    if (x !== x) return 'nan';
    if (x === Infinity) return 'inf';
    if (x === -Infinity) return '-inf';
    if (x === 0) return 1 / x < 0 ? '-0.0' : '0.0';
    var m = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(x.toExponential());
    var sign = m[1], digits = m[2] + (m[3] || ''), e = parseInt(m[4], 10);
    if (e < -4 || e >= 16) {
      var mant = digits.charAt(0) + (digits.length > 1 ? '.' + digits.slice(1) : '');
      var ae = Math.abs(e);
      return sign + mant + 'e' + (e < 0 ? '-' : '+') + (ae < 10 ? '0' + ae : String(ae));
    }
    if (e < 0) return sign + '0.' + '0'.repeat(-e - 1) + digits;
    if (digits.length <= e + 1) return sign + digits + '0'.repeat(e + 1 - digits.length) + '.0';
    return sign + digits.slice(0, e + 1) + '.' + digits.slice(e + 1);
  }

  // ---- sum() ----
  /** Python's built-in sum() over floats. Since Python 3.12 it uses Neumaier compensated summation,
      so a plain `total += x` loop can differ in the last bit. This mirrors CPython's builtin_sum. */
  function sum(xs) {
    if (!xs.length) return 0;
    var f = 0 + xs[0], c = 0.0;
    for (var i = 1; i < xs.length; i++) {
      var x = xs[i], t = f + x;
      if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x; else c += (x - t) + f;
      f = t;
    }
    if (c && isFinite(c)) f += c;
    return f;
  }

  // ---- random.Random(seed): MT19937 exactly as CPython seeds and reads it ----
  function Random(seed) {
    if (!(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff)) throw new RangeError('seed must be an integer in [0, 2**32)');
    var N = 624, mt = new Uint32Array(N), i, j, k;
    mt[0] = 19650218;
    for (i = 1; i < N; i++) mt[i] = (Math.imul(1812433253, mt[i - 1] ^ (mt[i - 1] >>> 30)) + i) >>> 0;
    var key = [seed >>> 0];                            // init_by_array([seed])
    i = 1; j = 0;
    for (k = N; k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (k = N - 1; k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
    this.mt = mt; this.idx = N;
  }
  Random.prototype.u32 = function () {
    var mt = this.mt, N = 624, M = 397, y, kk;
    if (this.idx >= N) {
      for (kk = 0; kk < N - M; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + M] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0); }
      for (; kk < N - 1; kk++) { y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff); mt[kk] = mt[kk + M - N] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0); }
      y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff); mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      this.idx = 0;
    }
    y = mt[this.idx++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  };
  /** random.random(): 53 random bits in [0, 1). */
  Random.prototype.random = function () {
    var a = this.u32() >>> 5, b = this.u32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  };
  /** rng.choices(population, weights=weights)[0] */
  Random.prototype.choice = function (population, weights) {
    var cum = [], acc = 0, i;
    for (i = 0; i < weights.length; i++) { acc = i === 0 ? weights[0] : acc + weights[i]; cum.push(acc); }
    var total = cum[cum.length - 1] + 0.0;
    if (!(total > 0)) throw new RangeError('Total of weights must be greater than zero');
    var x = this.random() * total, lo = 0, hi = population.length - 1;   // bisect_right(cum, x, 0, hi)
    while (lo < hi) { var mid = (lo + hi) >> 1; if (x < cum[mid]) hi = mid; else lo = mid + 1; }
    return population[lo];
  };

  AIER.py = { isSpace: isSpace, strip: strip, split: split, cp: cp, sum: sum, fixed: fixed, fixedw: fixedw, signed: signed, round: round, repr: repr, Random: Random, WS: WS };
  if (typeof module === 'object' && module.exports) module.exports = AIER.py;
})(typeof self !== 'undefined' ? self : globalThis);
