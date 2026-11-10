/* Attention explorer: pick a token, see its attention weights, open up Q, K and V.
   Maths: scripts/site/core/attention.js (a tested port of llm_attention.py and llm_qkv.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, A = AIER.attention, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var SVGNS = 'http://www.w3.org/2000/svg';
  function s(tag, attrs) { var e = root.document.createElementNS(SVGNS, tag); Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); }); return e; }
  function identity(n) { var m = []; for (var i = 0; i < n; i++) { var r = []; for (var j = 0; j < n; j++) r.push(i === j ? 1.0 : 0.0); m.push(r); } return m; }
  function copy(m) { return m.map(function (r) { return r.slice(); }); }

  AIER.labs.attention = {
    mount: function (el, ctx) {
      var q = ctx.data.qkv, ad = ctx.data.attention;
      var presets = {
        qkv: { label: '"the animal it"', tokens: q.TOKENS, X: q.X, WQ: q.W_Q, WK: q.W_K, WV: q.W_V,
          note: 'Three tokens, 4 numbers each. W_q, W_k and W_v are hand-picked 4×2 matrices from llm_qkv.py. In a real model they are learned.' },
        plain: { label: '"The cat sat" (no projections)', tokens: ['The', 'cat', 'sat'], X: ad.x, WQ: identity(4), WK: identity(4), WV: identity(4),
          note: 'The llm_attention.py example: Q = K = V = X, so there are no learned matrices at all. Each token just compares its own numbers with the others.' }
      };
      var st = { preset: 'qkv', X: copy(presets.qkv.X), sel: 2, causal: true, stage: 'weights' };
      var STAGES = [
        ['x', 'X', 'The input: one row of numbers per token.'],
        ['q', 'Q = X·W_q', 'What each token is LOOKING FOR.'],
        ['k', 'K = X·W_k', 'How each token ADVERTISES itself.'],
        ['v', 'V = X·W_v', 'What each token HANDS OVER when it is attended to.'],
        ['scores', 'Q·Kᵀ', 'Row i: token i’s query against every key (dot products).'],
        ['scaled', '÷ √d_k', 'Divide by the square root of d_k so the softmax does not saturate.'],
        ['masked', '+ mask', 'Causal mask: -inf where a token would look at the future.'],
        ['weights', 'softmax', 'Each row becomes weights that add up to 1. These are the attention weights.'],
        ['output', 'weights·V', 'Each token’s new vector: a weighted mix of the value vectors.']
      ];

      function trace() { var p = presets[st.preset]; return A.attentionWithTrace(st.X, p.WQ, p.WK, p.WV, st.causal); }

      var tokensRow = h('div.att-tokens', { role: 'group', 'aria-label': 'Pick the token that is asking' });
      var arcs = s('svg', { 'class': 'att-arcs', 'aria-hidden': 'true', preserveAspectRatio: 'none' });
      var heat = h('div.att-heat');
      var summary = h('p.summary-line', { role: 'status' });
      var stageBody = h('div.att-stage');
      var xEditor = h('div');
      var presetNote = h('p.help');

      function select(i) { st.sel = i; render(); }

      function drawArcs(w) {
        while (arcs.firstChild) arcs.removeChild(arcs.firstChild);
        var box = tokensRow.getBoundingClientRect(), btns = tokensRow.querySelectorAll('button');
        if (!box.width || !btns.length) return;
        var H = 84;
        arcs.setAttribute('viewBox', '0 0 ' + box.width + ' ' + H);
        arcs.style.height = H + 'px';
        var cx = function (b) { var r = b.getBoundingClientRect(); return r.left - box.left + r.width / 2; };
        var from = cx(btns[st.sel]);
        w[st.sel].forEach(function (wt, j) {
          if (wt <= 0) return;
          var to = cx(btns[j]), width = Math.max(1.5, wt * 14), d;
          if (j === st.sel) d = 'M' + (from - 16) + ',' + H + ' C' + (from - 44) + ',' + (H - 62) + ' ' + (from + 44) + ',' + (H - 62) + ' ' + (from + 16) + ',' + H;
          else d = 'M' + from + ',' + H + ' Q' + (from + to) / 2 + ',' + (H - 40 - Math.abs(to - from) * 0.16) + ' ' + to + ',' + H;
          arcs.appendChild(s('path', { d: d, fill: 'none', stroke: 'var(--accent)', 'stroke-width': width, 'stroke-linecap': 'round', opacity: 0.35 + 0.65 * wt }));
        });
      }

      var tokenBtns = [], cells = [], rowEls = [];
      function build() {
        var p = presets[st.preset];
        ui.clear(tokensRow); tokenBtns = [];
        p.tokens.forEach(function (tok, j) {
          var b = h('button.att-token', { type: 'button', onclick: function () { select(j); },
            onmouseenter: function () { if (st.sel !== j) select(j); }, onfocus: function () { if (st.sel !== j) select(j); } },
            h('span.w', { 'aria-hidden': 'true' }), h('span.t', tok), h('span.r', { 'aria-hidden': 'true' }));
          tokenBtns.push(b); tokensRow.appendChild(b);
        });
        ui.clear(heat); cells = []; rowEls = [];
        var tb = h('tbody');
        p.tokens.forEach(function (tok, i) {
          var tds = p.tokens.map(function () { return h('td'); });
          var tr = h('tr', h('th', { scope: 'row' }, h('button', { type: 'button', onclick: function () { select(i); }, 'aria-label': 'Select ' + tok }, tok)), tds);
          cells.push(tds); rowEls.push(tr); tb.appendChild(tr);
        });
        heat.appendChild(h('table.att-map', h('caption', 'All attention weights. Row = the token asking, column = the token it looks at.'),
          h('thead', h('tr', h('td'), p.tokens.map(function (tok) { return h('th', { scope: 'col' }, tok); }))), tb));
      }

      function render() {
        var p = presets[st.preset], t = trace(), w = t.weights, n = p.tokens.length;
        if (st.sel >= n) st.sel = n - 1;
        tokenBtns.forEach(function (b, j) {
          var wt = w[st.sel][j], masked = t.mask[st.sel][j] === -Infinity;
          b.className = 'att-token' + (masked ? ' masked' : '');
          b.setAttribute('aria-pressed', String(j === st.sel));
          b.setAttribute('aria-label', p.tokens[j] + (j === st.sel ? ', the token that is asking' : '') + '. Weight from ' + p.tokens[st.sel] + ': ' + py.fixed(wt, 3) + (masked ? ' (hidden by the causal mask)' : ''));
          b.children[0].textContent = masked ? 'hidden' : py.fixed(wt * 100, 1) + '%';
          b.children[2].textContent = j === st.sel ? 'asking' : masked ? 'future' : '';
        });
        drawArcs(w);
        var best = 0; w[st.sel].forEach(function (x, j) { if (x > w[st.sel][best]) best = j; });
        ui.clear(summary);
        summary.appendChild(h('span', h('b', p.tokens[st.sel]), ' puts ' + py.fixed(w[st.sel][best] * 100, 1) + '% of its attention on ', h('b', p.tokens[best]),
          best === st.sel ? ' (itself).' : '.', st.causal ? ' With the causal mask it cannot see later tokens.' : ' Without the mask it sees every token.'));
        w.forEach(function (row, i) {
          rowEls[i].className = i === st.sel ? 'hl' : '';
          row.forEach(function (x, j) {
            var masked = t.mask[i][j] === -Infinity, td = cells[i][j];
            td.className = masked ? 'm' : ''; td.style.setProperty('--a', String(x)); td.textContent = masked ? '\u2013' : py.fixed(x, 3);
          });
        });
        // stage matrix
        ui.clear(stageBody);
        var meta = STAGES.filter(function (x) { return x[0] === st.stage; })[0];
        var m = t[st.stage], cols = null;
        if (['scores', 'scaled', 'masked', 'weights'].indexOf(st.stage) !== -1) cols = p.tokens.map(function (tok) { return '→ ' + tok; });
        stageBody.appendChild(h('p', h('b', meta[1] + ': '), meta[2], st.stage === 'scaled' ? ' Here √d_k = ' + py.fixed(Math.sqrt(t.d_k), 3) + '.' : '',
          st.stage === 'masked' && !st.causal ? ' The mask is off, so nothing changes.' : ''));
        stageBody.appendChild(h('div', { style: { 'overflow-x': 'auto' } }, ui.matrix(m, { rows: p.tokens, cols: cols, hl: st.sel, caption: meta[1] })));
        if (st.stage === 'weights') {
          var parts = t.masked[st.sel].map(function (x) { return x === -Infinity ? 'exp(-inf) = 0' : 'exp(' + py.fixed(x, 3) + ')'; });
          stageBody.appendChild(h('p.src', 'Row "' + p.tokens[st.sel] + '": softmax = each of [' + parts.join(', ') + '] divided by their sum.'));
        }
        if (st.stage === 'output') {
          var mix = w[st.sel].map(function (x, j) { return py.fixed(x, 3) + ' × V[' + p.tokens[j] + ']'; });
          stageBody.appendChild(h('p.src', 'Row "' + p.tokens[st.sel] + '" = ' + mix.join(' + ') + '.'));
        }
        presetNote.textContent = p.note;
      }

      function renderEditor() {
        ui.clear(xEditor);
        var p = presets[st.preset];
        var tbl = h('table.mx.att-edit', h('caption.sr-only', 'Edit the input numbers'));
        var tb = h('tbody');
        st.X.forEach(function (row, i) {
          tb.appendChild(h('tr', h('th', { scope: 'row' }, p.tokens[i]), row.map(function (v, j) {
            var out = h('output', py.fixed(v, 1));
            var bump = function (dv) { return function () { st.X[i][j] = py.round(Math.max(-3, Math.min(3, st.X[i][j] + dv)), 1); out.textContent = py.fixed(st.X[i][j], 1); render(); }; };
            return h('td', h('span.stepper', h('button', { type: 'button', 'aria-label': 'Decrease ' + p.tokens[i] + ' number ' + (j + 1), onclick: bump(-0.5) }, '−'), out,
              h('button', { type: 'button', 'aria-label': 'Increase ' + p.tokens[i] + ' number ' + (j + 1), onclick: bump(0.5) }, '+')));
          })));
        });
        tbl.appendChild(tb); xEditor.appendChild(tbl);
        xEditor.appendChild(h('button.btn.small.ghost', { type: 'button', style: { 'margin-top': '8px' }, onclick: function () { st.X = copy(presets[st.preset].X); renderEditor(); render(); } }, icon('redo', 'sm'), 'Reset the numbers'));
      }

      var presetSeg = ui.seg({ label: 'Example', options: [['qkv', presets.qkv.label], ['plain', presets.plain.label]], value: st.preset,
        onchange: function (v) { st.preset = v; st.X = copy(presets[v].X); st.sel = presets[v].tokens.length - 1; build(); renderEditor(); render(); } });
      var maskSeg = ui.seg({ label: 'Causal mask', options: [['on', 'Mask on (GPT-style)'], ['off', 'Mask off']], value: 'on', onchange: function (v) { st.causal = v === 'on'; render(); } });
      var stageSeg = ui.seg({ label: 'Step of the calculation', options: STAGES.map(function (x) { return [x[0], x[1]]; }), value: st.stage, onchange: function (v) { st.stage = v; render(); } });

      el.appendChild(ui.honest('Tiny hand-picked matrices, not a trained model. Real attention does the same arithmetic with hundreds of numbers per token and many heads.', q.src, ctx.repo));
      el.appendChild(h('div.controls', { style: { 'flex-direction': 'row', 'flex-wrap': 'wrap', gap: '16px 24px', 'margin-bottom': '16px' } },
        h('div.control', h('span.lbl', 'Example'), presetSeg), h('div.control', h('span.lbl', 'Causal mask'), maskSeg)));
      el.appendChild(presetNote);
      el.appendChild(h('p.help', { style: { margin: '12px 0 4px' } }, 'Hover, tap or tab to a word. It becomes the one asking "which words help me?"'));
      el.appendChild(h('div.att-wrap', arcs, tokensRow));
      el.appendChild(summary);
      el.appendChild(h('div.att-grid', heat,
        h('div', ui.disclosure('Show the Q, K, V numbers, step by step', [h('div', { style: { margin: '4px 0 12px' } }, stageSeg), stageBody], true),
          ui.disclosure('Change the input numbers yourself', [h('p', 'Each row is one token’s vector X. Nudge a number and every matrix above recomputes.'), xEditor]))));
      build(); renderEditor(); render();
      var onResize = function () { drawArcs(trace().weights); };
      root.addEventListener('resize', onResize);
      root.requestAnimationFrame(onResize);
      return function () { root.removeEventListener('resize', onResize); };
    }
  };
})(typeof self !== 'undefined' ? self : globalThis);
