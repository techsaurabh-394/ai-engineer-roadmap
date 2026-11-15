/* Chunking lab: cut a pasted text four ways and see where the cuts land.
   Maths: scripts/site/core/rag.js (a tested port of rag_chunkers.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, R = AIER.rag, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var MAX_CARDS = 40;

  AIER.labs.chunking = {
    mount: function (el, ctx) {
      var cd = ctx.data.chunking;
      var start = { chunking: 'fixed', '05-recursive-chunking': 'recursive', '06-semantic-chunking': 'semantic', '08-sliding-window-chunking': 'sliding' };
      var st = { mode: (ctx.lesson && start[ctx.lesson.name]) || 'fixed', text: cd.SAMPLE, size: 60, overlap: 10, win: 12, stride: 6, rsize: 120, th: 0.1 };
      var MODES = {
        fixed: ['Fixed size', 'Cut every N characters. Neighbours share the overlap, so a sentence cut in half still appears whole in one of them.'],
        sliding: ['Sliding window', 'Windows of N words. Each window starts "stride" words after the last one.'],
        recursive: ['Recursive', 'Split on paragraphs first, then lines, then sentences, then words: only as small as needed to fit the size.'],
        semantic: ['Semantic (toy)', 'Start a new chunk when a sentence shares too few words with the one before it. Real systems compare embeddings instead of words.']
      };

      var areaId = ui.id('txt');
      var area = h('textarea.textin', { id: areaId, rows: 6, spellcheck: 'false' });
      area.value = st.text;
      area.addEventListener('input', function () { st.text = area.value; render(); });
      var sliders = h('div.controls');
      var modeNote = h('p.help');
      var stats = h('p.summary-line', { role: 'status' });
      var map = h('div.chunk-map', { 'aria-hidden': 'true' });
      var list = h('ol.chunk-list');
      var seg = ui.seg({ label: 'How to cut', options: Object.keys(MODES).map(function (k) { return [k, MODES[k][0]]; }), value: st.mode, onchange: function (v) { st.mode = v; layout(); render(); } });

      function layout() {
        ui.clear(sliders);
        modeNote.textContent = MODES[st.mode][1];
        if (st.mode === 'fixed') {
          var ov = ui.slider({ label: 'Overlap (characters)', min: 0, max: Math.max(1, st.size - 1), step: 1, value: Math.min(st.overlap, st.size - 1), oninput: function (v) { st.overlap = v; render(); } });
          sliders.appendChild(ui.slider({ label: 'Chunk size (characters)', min: 10, max: 300, step: 5, value: st.size, oninput: function (v) {
            st.size = v; if (st.overlap > v - 1) st.overlap = v - 1; ov.setMax(v - 1, st.overlap); render(); } }));
          sliders.appendChild(ov);
        } else if (st.mode === 'sliding') {
          sliders.appendChild(ui.slider({ label: 'Window (words)', min: 2, max: 40, step: 1, value: st.win, oninput: function (v) { st.win = v; render(); } }));
          sliders.appendChild(ui.slider({ label: 'Stride (words to move forward)', min: 1, max: 40, step: 1, value: st.stride, oninput: function (v) { st.stride = v; render(); } }));
        } else if (st.mode === 'recursive') {
          sliders.appendChild(ui.slider({ label: 'Largest chunk (characters)', min: 20, max: 400, step: 10, value: st.rsize, oninput: function (v) { st.rsize = v; render(); } }));
        } else {
          sliders.appendChild(ui.slider({ label: 'New chunk when word overlap is below', min: 0, max: 0.5, step: 0.05, value: st.th, format: function (v) { return py.fixed(v, 2); },
            oninput: function (v) { st.th = py.round(v, 2); render(); } }));
        }
      }
      function visible(text) { return text.replace(/\n/g, '↵\n'); }

      function render() {
        var text = st.text, chunks, shared = [], spans = null, total, unit;
        if (st.mode === 'fixed') {
          chunks = R.fixedSizeChunks(text, st.size, st.overlap);
          total = py.cp(text).length; unit = 'characters';
          var step = st.size - st.overlap;
          spans = chunks.map(function (c, i) { return [i * step, py.cp(c).length]; });
          shared = chunks.map(function (c, i) { return i ? py.cp(c).slice(0, st.overlap).join('') : ''; });
        } else if (st.mode === 'sliding') {
          chunks = R.slidingWindowChunks(text, st.win, st.stride);
          total = py.split(text).length; unit = 'words';
          spans = chunks.map(function (c, i) { return [i * st.stride, py.split(c).length]; });
          var keep = Math.max(0, st.win - st.stride);
          shared = chunks.map(function (c, i) { return i && keep ? c.split(' ').slice(0, keep).join(' ') : ''; });
        } else if (st.mode === 'recursive') {
          chunks = R.recursiveChunks(text, st.rsize);
        } else {
          chunks = R.semanticChunks(text, st.th);
        }
        // numbers
        var lens = chunks.map(function (c) { return py.cp(c).length; }), sumLen = 0;
        lens.forEach(function (n) { sumLen += n; });
        ui.clear(stats);
        if (!chunks.length) { stats.textContent = 'No chunks: the text is empty.'; ui.clear(list); ui.clear(map); return; }
        var msg = [h('b', String(chunks.length)), chunks.length === 1 ? ' chunk' : ' chunks', ', ' + Math.min.apply(null, lens) + ' to ' + Math.max.apply(null, lens) + ' characters each.'];
        if (spans) {
          var covered = 0, stored = 0;
          spans.forEach(function (sp) { stored += sp[1]; });
          var seen = {};
          spans.forEach(function (sp) { for (var x = sp[0]; x < sp[0] + sp[1]; x++) if (!seen[x]) { seen[x] = 1; covered += 1; } });
          var extra = stored - covered, lost = total - covered;
          if (extra > 0) msg.push(' ' + extra + ' ' + unit + ' are stored twice (' + py.fixed(extra / total * 100, 0) + '% extra to embed and search).');
          else msg.push(' Nothing is stored twice.');
          if (lost > 0) msg.push(h('span', { style: { color: 'var(--bad)', 'font-weight': '700' } }, ' ' + lost + ' ' + unit + ' fall between windows and are lost: the stride is larger than the window.'));
        } else if (st.mode === 'semantic') {
          msg.push(' ' + R.splitSentences(text).length + ' sentences in total.');
        }
        msg.forEach(function (m) { stats.appendChild(m.nodeType ? m : root.document.createTextNode(m)); });
        // map
        ui.clear(map);
        if (spans && total) {
          spans.slice(0, 24).forEach(function (sp, i) {
            map.appendChild(h('div.row', h('i.c' + (i % 2), { style: { 'margin-left': (sp[0] / total * 100) + '%', width: Math.max(0.8, sp[1] / total * 100) + '%' } })));
          });
          map.hidden = false;
        } else map.hidden = true;
        // cards
        ui.clear(list);
        chunks.slice(0, MAX_CARDS).forEach(function (c, i) {
          var body = h('p.txt');
          if (shared[i]) { body.appendChild(h('mark', { title: 'Also in chunk ' + i }, visible(shared[i]))); body.appendChild(root.document.createTextNode(visible(c.slice(shared[i].length)))); }
          else body.textContent = visible(c);
          list.appendChild(h('li.chunk.c' + (i % 2), h('p.hd', h('span', 'Chunk ' + (i + 1)), h('span', lens[i] + ' chars' + (shared[i] ? ' · highlighted part is also in chunk ' + i : ''))), body));
        });
        if (chunks.length > MAX_CARDS) list.appendChild(h('li.muted', '… and ' + (chunks.length - MAX_CARDS) + ' more chunks. Make the chunks larger to see them all.'));
      }

      el.appendChild(ui.honest('These chunkers count characters and words. Production systems usually count tokens, and the "semantic" one here compares shared words, not embeddings.', cd.src, ctx.repo));
      el.appendChild(h('div.lab-grid',
        h('div.controls', h('div.control', h('span.lbl', 'How to cut'), seg, modeNote), sliders,
          h('div.control', h('label', { 'for': areaId }, h('span', 'Your text (paste anything)')), area,
            h('button.btn.small.ghost', { type: 'button', style: { 'margin-top': '6px' }, onclick: function () { area.value = cd.SAMPLE; st.text = cd.SAMPLE; render(); } }, icon('redo', 'sm'), 'Use the sample text'))),
        h('div', stats, map, list)));
      layout(); render();
    }
  };

  AIER.labs.chunking.tour = function (el) {
    var q = function (s) { return function () { return el.querySelector(s); }; };
    var mode = function (i) { return function () { var b = el.querySelectorAll('.seg button')[i]; if (b) b.click(); }; };
    return [
      { el: q('.seg'), before: mode(0), title: 'Four ways to cut', text: 'Fixed size, sliding window, recursive and a toy semantic chunker, all ported from rag_chunkers.py.' },
      { el: function () { return el.querySelectorAll('input[type=range]')[1].closest('.control'); }, title: 'Overlap', text: 'Neighbouring chunks share these characters, highlighted in each chunk, so an idea cut in half still appears whole somewhere.' },
      { el: q('.chunk-map'), title: 'Where the cuts land', text: 'Each bar is one chunk across the text. Overlapping bars are text stored twice: more context, more to embed.' },
      { el: q('.seg'), before: mode(2), title: 'Recursive respects structure', text: 'It splits on paragraphs first and only goes smaller when a piece is too big.' },
      { el: q('textarea'), title: 'Bring your own text', text: 'Paste anything here and every chunker runs on it straight away.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
