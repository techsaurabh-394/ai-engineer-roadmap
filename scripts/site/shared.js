/* Shared UI helpers for the lessons viewer, the labs and the quiz. Vanilla JS, no dependencies. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  var doc = root.document;

  // ---------------------------------------------------------------- storage (never throws)
  var store = {
    get: function (key, fallback) {
      try { var v = root.localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
    },
    set: function (key, value) { try { root.localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode */ } }
  };

  // ---------------------------------------------------------------- DOM builder
  /** h('div.card', {onclick: fn, 'aria-label': 'x'}, child, 'text', [more]) */
  function h(spec, attrs) {
    var parts = spec.split('.'), el = doc.createElement(parts[0] || 'div'), i;
    if (parts.length > 1) el.className = parts.slice(1).join(' ');
    var start = 1;
    if (attrs && typeof attrs === 'object' && !attrs.nodeType && !Array.isArray(attrs)) {
      start = 2;
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else if (k === 'html') el.innerHTML = v;                 // only for build-time, already-escaped HTML
        else if (k === 'style' && typeof v === 'object') Object.keys(v).forEach(function (p) { el.style.setProperty(p, v[p]); });
        else if (k === 'dataset') Object.keys(v).forEach(function (d) { el.dataset[d] = v[d]; });
        else el.setAttribute(k, v === true ? '' : String(v));
      });
    }
    for (i = start; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function append(el, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(el, c); }); return; }
    el.appendChild(child.nodeType ? child : doc.createTextNode(String(child)));
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  // ---------------------------------------------------------------- icons (own artwork, 24x24 strokes)
  var ICONS = {
    play: '<path d="M7 5l12 7-12 7z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M8 5v14M16 5v14"/>',
    prev: '<path d="M15 6l-6 6 6 6"/>',
    next: '<path d="M9 6l6 6-6 6"/>',
    chevron: '<path d="M9 6l6 6-6 6"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.2 5.2l1.8 1.8M17 17l1.8 1.8M5.2 18.8L7 17M17 7l1.8-1.8"/>',
    moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    auto: '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17" /><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor" stroke="none"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    redo: '<path d="M4 12a8 8 0 1 0 2.6-5.9"/><path d="M4 4v5h5"/>',
    flask: '<path d="M9.5 3h5M10.5 3v6.2L5.2 18a2 2 0 0 0 1.7 3h10.2a2 2 0 0 0 1.7-3l-5.3-8.8V3"/><path d="M7.8 15h8.4"/>',
    book: '<path d="M5 4.5h10a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h10"/>',
    code: '<path d="M8.5 7.5L4 12l4.5 4.5M15.5 7.5L20 12l-4.5 4.5M13.5 5l-3 14"/>',
    quiz: '<circle cx="12" cy="12" r="8.5"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.6 2.2c-.7.4-1.1.9-1.1 1.8"/><path d="M12 16.6v.1"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.6v.1"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    ext: '<path d="M13 5h6v6M19 5l-8 8M10 6H6a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-4"/>',
    spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
    shuffle: '<path d="M4 7h3c4 0 5 10 9 10h4M4 17h3c1.5 0 2.5-1.5 3.5-3M14 8.500c.700-.900 1.3-1.5 2-1.500h4M17.5 4.500L20 7l-2.5 2.500M17.5 14.500L20 17l-2.5 2.5"/>',
    star: '<path d="M12 3.500l2.6 5.4 5.9.800-4.3 4.1 1 5.900L12 16.900l-5.2 2.8 1-5.9-4.3-4.1 5.9-.800z"/>',
    home: '<path d="M4 11l8-7 8 7v8a1 1 0 0 1-1 1h-4.500v-6h-5v6H5a1 1 0 0 1-1-1z"/>'
  };
  function icon(name, cls) {
    var span = doc.createElement('span');
    span.innerHTML = '<svg class="ic' + (cls ? ' ' + cls : '') + '" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' + (ICONS[name] || '') + '</svg>';
    return span.firstChild;
  }

  // ---------------------------------------------------------------- theme
  var THEME_KEY = 'aier-theme', ORDER = ['auto', 'light', 'dark'];
  var THEME_LABEL = { auto: 'Theme: auto (follows your device)', light: 'Theme: light', dark: 'Theme: dark' };
  var THEME_ICON = { auto: 'auto', light: 'sun', dark: 'moon' };
  function themePref() { var t = store.get(THEME_KEY, 'auto'); return ORDER.indexOf(t) === -1 ? 'auto' : t; }
  function isDark() {
    var t = themePref();
    if (t === 'auto') return !!(root.matchMedia && root.matchMedia('(prefers-color-scheme: dark)').matches);
    return t === 'dark';
  }
  function applyTheme() {
    var t = themePref(), el = doc.documentElement;
    if (t === 'auto') el.removeAttribute('data-theme'); else el.setAttribute('data-theme', t);
    var btn = doc.getElementById('theme');
    if (btn) { clear(btn).appendChild(icon(THEME_ICON[t])); btn.setAttribute('aria-label', THEME_LABEL[t] + '. Click to change.'); btn.title = THEME_LABEL[t]; }
    doc.dispatchEvent(new CustomEvent('aier:theme', { detail: { dark: isDark() } }));
  }
  function initTheme() {
    applyTheme();
    var btn = doc.getElementById('theme');
    if (btn) btn.addEventListener('click', function () { store.set(THEME_KEY, ORDER[(ORDER.indexOf(themePref()) + 1) % ORDER.length]); applyTheme(); });
    if (root.matchMedia) {
      var mq = root.matchMedia('(prefers-color-scheme: dark)'), fn = function () { if (themePref() === 'auto') applyTheme(); };
      if (mq.addEventListener) mq.addEventListener('change', fn); else if (mq.addListener) mq.addListener(fn);
    }
  }
  function reducedMotion() { return !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches); }

  // ---------------------------------------------------------------- controls
  var uid = 0;
  function id(prefix) { uid += 1; return (prefix || 'c') + '-' + uid; }

  /** A labelled range input. opts: {label, min, max, step, value, format(v), help, ticks, oninput(v)} */
  function slider(opts) {
    var inputId = id('rng'), fmt = opts.format || String;
    var val = h('span.val', { 'aria-hidden': 'true' }, fmt(opts.value));
    var input = h('input', { type: 'range', id: inputId, min: opts.min, max: opts.max, step: opts.step, value: opts.value,
      'aria-valuetext': fmt(opts.value) });
    function paint() {
      var v = Number(input.value), pct = (v - opts.min) / (opts.max - opts.min) * 100;
      input.style.setProperty('--fill', pct + '%');
      val.textContent = fmt(v); input.setAttribute('aria-valuetext', fmt(v));
    }
    input.addEventListener('input', function () { paint(); opts.oninput(Number(input.value)); });
    paint();
    var el = h('div.control', h('label', { 'for': inputId }, h('span', opts.label), val), input,
      opts.ticks ? h('div.ticks', { 'aria-hidden': 'true' }, opts.ticks.map(function (t) { return h('span', t); })) : null,
      opts.help ? h('p.help', opts.help) : null);
    el.input = input;
    el.set = function (v) { input.value = v; paint(); };
    el.setMax = function (max, v) { opts.max = max; input.max = max; if (v !== undefined) input.value = v; paint(); };
    return el;
  }
  /** Segmented buttons. opts: {label, options: [[value, text]], value, onchange(v)} */
  function seg(opts) {
    var el = h('div.seg', { role: 'group', 'aria-label': opts.label });
    opts.options.forEach(function (o) {
      el.appendChild(h('button', { type: 'button', 'aria-pressed': String(o[0] === opts.value), dataset: { v: o[0] },
        onclick: function () { el.set(o[0]); opts.onchange(o[0]); } }, o[1]));
    });
    el.set = function (v) { Array.prototype.forEach.call(el.children, function (b) { b.setAttribute('aria-pressed', String(b.dataset.v === String(v))); }); };
    return el;
  }
  /** <details class="math"> with a chevron. */
  function disclosure(summary, body, open) {
    var d = h('details.math', open ? { open: true } : {}, h('summary', icon('chevron', 'sm'), h('span', summary)), h('div.body', body));
    return d;
  }
  /** The honesty banner every lab carries. */
  function honest(text, srcPath, repo) {
    return h('p.honest', icon('info', 'sm'), h('span', text + ' ',
      srcPath ? h('a', { href: repo + '/blob/main/' + srcPath, rel: 'noopener' }, 'Read the Python it comes from') : null,
      srcPath ? '.' : null));
  }
  /** A numeric matrix as a table. opts: {rows: labels, cols: labels, hl: row index, d: decimals} */
  function matrix(m, opts) {
    opts = opts || {};
    var d = opts.d === undefined ? 3 : opts.d, fixed = AIER.py.fixed;
    var t = h('table.mx');
    if (opts.caption) t.appendChild(h('caption.sr-only', opts.caption));
    if (opts.cols) t.appendChild(h('thead', h('tr', opts.rows ? h('th') : null, opts.cols.map(function (c) { return h('th', { scope: 'col' }, c); }))));
    var body = h('tbody');
    m.forEach(function (row, i) {
      body.appendChild(h('tr' + (opts.hl === i ? '.hl' : ''), opts.rows ? h('th', { scope: 'row' }, opts.rows[i]) : null,
        row.map(function (v) { return h('td' + (v === -Infinity ? '.neg' : ''), v === -Infinity ? '-inf' : fixed(v, d)); })));
    });
    t.appendChild(body);
    return t;
  }

  /** Horizontal probability bars. rows: [{label, value (0..1), ghost (0..1 or null), cut: 'reason' | null, dim, picked}] */
  function bars(el, rows, opts) {
    opts = opts || {};
    var fixed = AIER.py.fixed, d = opts.d === undefined ? 3 : opts.d, max = opts.max || 1;
    var existing = el.querySelectorAll('.bar-row');
    if (existing.length !== rows.length || el.dataset.sig !== rows.map(function (r) { return r.label; }).join('\u0001')) {
      clear(el); el.dataset.sig = rows.map(function (r) { return r.label; }).join('\u0001');
      rows.forEach(function () {
        el.appendChild(h('div.bar-row', { role: 'listitem' }, h('span.tok'), h('span.track', h('span.ghost'), h('span.fill'), h('span.why')), h('span.pct')));
      });
      existing = el.querySelectorAll('.bar-row');
    }
    el.setAttribute('role', 'list');
    rows.forEach(function (r, i) {
      var row = existing[i];
      row.className = 'bar-row' + (r.cut ? ' cut' : '') + (r.dim ? ' dim' : '') + (r.picked ? ' picked' : '');
      row.children[0].textContent = r.label;
      var track = row.children[1];
      track.children[0].style.width = r.ghost === null || r.ghost === undefined ? '0' : (r.ghost / max * 100) + '%';
      track.children[0].style.display = r.ghost === null || r.ghost === undefined ? 'none' : '';
      track.children[1].style.width = (r.cut ? 0 : r.value / max * 100) + '%';
      track.children[2].textContent = r.cut || '';
      row.children[2].textContent = r.cut ? '0' : fixed(r.value, d);
      row.setAttribute('aria-label', r.label + ': ' + (r.cut ? r.cut : fixed(r.value, d)));
    });
  }

  AIER.ui = { store: store, h: h, clear: clear, esc: esc, icon: icon, initTheme: initTheme, applyTheme: applyTheme, isDark: isDark,
    reducedMotion: reducedMotion, id: id, slider: slider, seg: seg, disclosure: disclosure, honest: honest, matrix: matrix, bars: bars };
})(typeof self !== 'undefined' ? self : globalThis);
