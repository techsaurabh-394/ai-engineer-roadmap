/* Similarity lab: drag two vectors and compare cosine similarity, dot product and Euclidean distance.
   Maths: scripts/site/core/rag.js (a tested port of rag_vector_math.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, R = AIER.rag, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var NS = 'http://www.w3.org/2000/svg', SIZE = 360, LIM = 6, SCALE = SIZE / (2 * LIM);
  function s(tag, attrs, text) {
    var e = root.document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function px(v) { return [SIZE / 2 + v[0] * SCALE, SIZE / 2 - v[1] * SCALE]; }
  function snap(x) { return py.round(Math.max(-LIM, Math.min(LIM, Math.round(x * 2) / 2)), 1); }

  AIER.labs.similarity = {
    mount: function (el, ctx) {
      var vd = ctx.data.vectors;
      var st = { a: [4.0, 1.0], b: [1.5, 3.0], unit: false };
      var svg = s('svg', { viewBox: '0 0 ' + SIZE + ' ' + SIZE, 'class': 'vec-plane', role: 'group', 'aria-label': 'A plane with two vectors, a and b, that you can move' });
      var defs = s('defs');
      ['a', 'b'].forEach(function (k) {
        var m = s('marker', { id: 'vec-head-' + k, viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse' });
        m.appendChild(s('path', { d: 'M0,0 L10,5 L0,10 z', 'class': 'fill-' + k }));
        defs.appendChild(m);
      });
      svg.appendChild(defs);
      for (var g = -LIM; g <= LIM; g++) {
        var p = SIZE / 2 + g * SCALE, cls = g === 0 ? 'axis' : 'grid';
        svg.appendChild(s('line', { x1: p, y1: 0, x2: p, y2: SIZE, 'class': cls }));
        svg.appendChild(s('line', { x1: 0, y1: p, x2: SIZE, y2: p, 'class': cls }));
      }
      var docDots = vd.docs.map(function (d, i) {
        var c = px(d), grp = s('g', { 'class': 'doc' });
        grp.appendChild(s('circle', { cx: c[0], cy: c[1], r: 5 }));
        grp.appendChild(s('text', { x: c[0] + 8, y: c[1] + 4 }, 'D' + i));
        svg.appendChild(grp);
        return grp;
      });
      var arc = s('path', { 'class': 'angle' }), dist = s('line', { 'class': 'dist' });
      svg.appendChild(arc); svg.appendChild(dist);
      var parts = {};
      ['a', 'b'].forEach(function (k) {
        var line = s('line', { x1: SIZE / 2, y1: SIZE / 2, 'class': 'vec stroke-' + k, 'marker-end': 'url(#vec-head-' + k + ')' });
        var unit = s('line', { x1: SIZE / 2, y1: SIZE / 2, 'class': 'unit stroke-' + k });
        var handle = s('g', { 'class': 'handle', tabindex: '0', role: 'slider', 'aria-valuemin': -LIM, 'aria-valuemax': LIM });
        handle.appendChild(s('circle', { r: 22, 'class': 'hit' }));
        handle.appendChild(s('circle', { r: 11, 'class': 'knob fill-' + k }));
        handle.appendChild(s('text', { 'class': 'knob-t', y: 4 }, k));
        svg.appendChild(unit); svg.appendChild(line); svg.appendChild(handle);
        parts[k] = { line: line, unit: unit, handle: handle };
        var dragging = false;
        function move(e) {
          var r = svg.getBoundingClientRect();
          st[k] = [snap(((e.clientX - r.left) / r.width * SIZE - SIZE / 2) / SCALE), snap((SIZE / 2 - (e.clientY - r.top) / r.height * SIZE) / SCALE)];
          render();
        }
        handle.addEventListener('pointerdown', function (e) { dragging = true; handle.setPointerCapture(e.pointerId); handle.focus(); e.preventDefault(); });
        handle.addEventListener('pointermove', function (e) { if (dragging) move(e); });
        handle.addEventListener('pointerup', function (e) { dragging = false; if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId); });
        handle.addEventListener('pointercancel', function () { dragging = false; });
        handle.addEventListener('keydown', function (e) {
          var d = { ArrowLeft: [-0.5, 0], ArrowRight: [0.5, 0], ArrowUp: [0, 0.5], ArrowDown: [0, -0.5] }[e.key];
          if (!d) return;
          st[k] = [snap(st[k][0] + d[0]), snap(st[k][1] + d[1])];
          e.preventDefault(); e.stopPropagation(); render();
        });
      });

      var cards = h('div.metric-grid'), verdict = h('p.summary-line', { role: 'status' }), ranks = h('div'), math = h('div');
      function metric(name, value, note, cls) { return h('div.metric' + (cls ? '.' + cls : ''), h('span.k', name), h('span.v', value), h('span.n', note)); }
      function vec(v) { return '(' + py.fixed(v[0], 1) + ', ' + py.fixed(v[1], 1) + ')'; }

      function render() {
        var a = st.a, b = st.b, na = R.norm(a), nb = R.norm(b), dotv = R.dot(a, b), cos = R.cosineSimilarity(a, b), euc = R.euclideanDistance(a, b);
        var zero = na === 0 || nb === 0;
        ['a', 'b'].forEach(function (k) {
          var c = px(st[k]), pt = parts[k];
          pt.line.setAttribute('x2', c[0]); pt.line.setAttribute('y2', c[1]);
          pt.handle.setAttribute('transform', 'translate(' + c[0] + ',' + c[1] + ')');
          pt.handle.setAttribute('aria-label', 'Vector ' + k + ' at ' + vec(st[k]) + '. Arrow keys move it by 0.5.');
          pt.handle.setAttribute('aria-valuenow', st[k][0]); pt.handle.setAttribute('aria-valuetext', vec(st[k]));
          var n = R.norm(st[k]);
          if (st.unit && n) { var u = px(R.normalize(st[k])); pt.unit.setAttribute('x2', u[0]); pt.unit.setAttribute('y2', u[1]); pt.unit.style.display = ''; }
          else pt.unit.style.display = 'none';
        });
        var ca = px(a), cb = px(b);
        dist.setAttribute('x1', ca[0]); dist.setAttribute('y1', ca[1]); dist.setAttribute('x2', cb[0]); dist.setAttribute('y2', cb[1]);
        var angle = zero ? 0 : Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;      // derived from the cosine, for the picture only
        if (!zero) {
          var r = 30, aa = Math.atan2(a[1], a[0]), ab = Math.atan2(b[1], b[0]);
          var sweep = ((ab - aa + 3 * Math.PI) % (2 * Math.PI)) - Math.PI > 0 ? 0 : 1;
          arc.setAttribute('d', 'M' + (SIZE / 2 + r * Math.cos(aa)) + ',' + (SIZE / 2 - r * Math.sin(aa)) + ' A' + r + ',' + r + ' 0 0 ' + sweep + ' ' + (SIZE / 2 + r * Math.cos(ab)) + ',' + (SIZE / 2 - r * Math.sin(ab)));
          arc.style.display = '';
        } else arc.style.display = 'none';

        ui.clear(cards);
        cards.appendChild(metric('Cosine similarity', py.fixed(cos, 3), zero ? 'a zero vector has no direction, so the demo returns 0' : 'direction only: 1 same, 0 unrelated, -1 opposite', 'main'));
        cards.appendChild(metric('Dot product', py.fixed(dotv, 3), 'direction AND length'));
        cards.appendChild(metric('Euclidean distance', py.fixed(euc, 3), 'straight-line gap between the tips (dashed)'));
        cards.appendChild(metric('Angle', py.fixed(angle, 1) + '°', 'worked out from the cosine'));
        cards.appendChild(metric('Length of a', py.fixed(na, 3), 'a = ' + vec(a)));
        cards.appendChild(metric('Length of b', py.fixed(nb, 3), 'b = ' + vec(b)));

        ui.clear(verdict);
        verdict.appendChild(h('span', zero ? 'One vector is zero, so there is no angle to measure.'
          : cos > 0.95 ? 'Almost the same direction: these would count as "about the same meaning".'
          : cos > 0.3 ? 'Related: they lean the same way.' : cos > -0.3 ? 'Close to a right angle: unrelated.' : 'Pointing apart: opposite meaning.',
          ' Now make b longer without turning it: cosine stays put, dot product and distance change.'));

        // rankings with the demo's three documents
        var names = ['cosine', 'dot', 'euclidean'], order = names.map(function (m) { return R.rank(a, vd.docs, m); });
        ui.clear(ranks);
        ranks.appendChild(h('p.k', 'Which document is closest to a? Three metrics, three opinions'));
        ranks.appendChild(h('table.data', h('thead', h('tr', h('th', 'Metric'), h('th', '1st'), h('th', '2nd'), h('th', '3rd'))),
          h('tbody', names.map(function (m, i) { return h('tr', h('th', { scope: 'row' }, m), order[i].map(function (d) { return h('td', 'D' + d + ' ' + vec(vd.docs[d])); })); }))));
        var agree = order.every(function (o) { return o.join() === order[0].join(); });
        ranks.appendChild(h('p.help', agree ? 'All three agree for this a. Move a towards (1.0, 0.2), the demo’s query, to make them disagree.'
          : 'They disagree. Dot product likes long vectors (D1 is far from the origin); cosine ignores length; Euclidean distance wants nearby tips.'));
        docDots.forEach(function (gEl, i) { gEl.setAttribute('class', 'doc' + (order[0][0] === i ? ' top' : '')); });

        // the arithmetic
        ui.clear(math);
        var f = function (x) { return py.fixed(x, 3); }, f1 = function (x) { return py.fixed(x, 1); };
        var lines = [
          'dot      = ' + f1(a[0]) + '*' + f1(b[0]) + ' + ' + f1(a[1]) + '*' + f1(b[1]) + ' = ' + f(dotv),
          '|a|      = sqrt(' + f1(a[0]) + '^2 + ' + f1(a[1]) + '^2) = ' + f(na),
          '|b|      = sqrt(' + f1(b[0]) + '^2 + ' + f1(b[1]) + '^2) = ' + f(nb),
          'cosine   = dot / (|a| * |b|) = ' + f(dotv) + ' / (' + f(na) + ' * ' + f(nb) + ') = ' + f(cos),
          'distance = sqrt((' + f1(a[0]) + ' - ' + f1(b[0]) + ')^2 + (' + f1(a[1]) + ' - ' + f1(b[1]) + ')^2) = ' + f(euc)
        ];
        if (!zero) {
          var ua = R.normalize(a), ub = R.normalize(b), d2 = Math.pow(R.euclideanDistance(ua, ub), 2);
          lines.push('', 'After scaling both to length 1 (normalizing):',
            'dot      = ' + f(R.dot(ua, ub)) + '   (now equal to the cosine)',
            'dist^2   = ' + f(d2) + '   and 2 - 2*cosine = ' + f(2 - 2 * cos) + '   (the same number)');
        }
        math.appendChild(h('pre.out', lines.join('\n')));
        math.appendChild(h('p.src', 'The topic pages work the 3-number example by hand: ' + vd.worked.slice(0, 3).map(function (kv) { return kv[0] + ' = ' + (typeof kv[1] === 'number' ? py.fixed(kv[1], 3) : kv[1]); }).join(', ') + '.'));
      }

      var unitSeg = ui.seg({ label: 'Unit vectors', options: [['off', 'As drawn'], ['on', 'Also show length-1 versions']], value: 'off', onchange: function (v) { st.unit = v === 'on'; render(); } });
      el.appendChild(ui.honest('Real embeddings have hundreds or thousands of numbers. Two numbers are enough to see what each metric measures.', vd.src, ctx.repo));
      el.appendChild(h('div.lab-grid.vec-grid',
        h('div', svg, h('p.help', 'Drag the round handles, or focus one and use the arrow keys. D0, D1, D2 are the three documents from the demo.'),
          h('div.row-btns', unitSeg,
            h('button.btn.small', { type: 'button', onclick: function () { st.a = vd.q.slice(); render(); } }, 'Set a to the demo query ' + vec(vd.q)),
            h('button.btn.small.ghost', { type: 'button', onclick: function () { st.a = [4.0, 1.0]; st.b = [1.5, 3.0]; render(); } }, icon('redo', 'sm'), 'Reset'))),
        h('div', verdict, cards, h('div.result', ranks), ui.disclosure('Show the math', math))));
      render();
    }
  };
})(typeof self !== 'undefined' ? self : globalThis);
