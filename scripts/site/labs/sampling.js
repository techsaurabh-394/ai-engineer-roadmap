/* Sampling lab: temperature, top-k and top-p on a toy next-token distribution.
   Maths: scripts/site/core/sampling.js (a tested port of sampling_demo.py and llm_decoding.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, S = AIER.sampling, py = AIER.py;
  AIER.labs = AIER.labs || {};

  AIER.labs.sampling = {
    mount: function (el, ctx) {
      var sd = ctx.data.sampling, dd = ctx.data.decoding, td = ctx.data.toyllm;
      var toy = td && AIER.toyllm ? new AIER.toyllm.ToyLLM(td) : null;
      var tableWords = Object.keys(dd.NEXT).filter(function (w) { return w !== '<start>'; });
      var st = { source: 'logits', prompt: 'the', toyPrompt: td ? td.example : '', t: 1.0, k: null, p: null, seed: 0, draws: [], rng: null };
      if (ctx.lesson && ctx.lesson.name === '12-beam-search') st.source = 'table';

      // ---- distribution for the current settings ----
      function lastWord(text) { var w = py.split(text.toLowerCase()); return w.length ? w[w.length - 1] : '<start>'; }
      function compute() {
        var vocab, base, temp, note = null, context;
        if (st.source === 'logits') {
          vocab = sd.TOKENS; context = 'The capital of France is';
          base = vocab.map(function (t, i) { return [t, S.softmax(sd.LOGITS, 1.0)[i]]; });
          temp = vocab.map(function (t, i) { return [t, S.softmax(sd.LOGITS, st.t)[i]]; });
        } else if (st.source === 'toy') {
          var tokens = toy.tokenize(st.toyPrompt), unknown = toy.unknown(tokens);
          vocab = td.VOCAB; context = st.toyPrompt;
          if (!tokens.length || unknown.length) {
            return { error: !tokens.length ? 'Add at least one word.' : 'Not in this toy model’s 8-word vocabulary: ' + unknown.join(', ') + '. Use the word buttons.' };
          }
          var logits = toy.forward(toy.toIds(tokens)).logits;
          base = vocab.map(function (t, i) { return [t, S.softmax(logits, 1.0)[i]]; });
          temp = vocab.map(function (t, i) { return [t, S.softmax(logits, st.t)[i]]; });
        } else {
          var prev = lastWord(st.prompt); context = st.prompt;
          base = S.nextProbs(dd.NEXT, prev);
          vocab = base.map(function (kv) { return kv[0]; });
          temp = S.applyTemperature(base, st.t);
          if (!Object.prototype.hasOwnProperty.call(dd.NEXT, prev)) note = '"' + prev + '" is not in the toy table, so the table says the sentence ends. Pick one of the words below.';
        }
        var k = st.k === null || st.k >= vocab.length ? null : st.k;
        var afterK = k === null ? temp : S.topK(temp, k);
        var afterP = st.p === null ? afterK : S.topP(afterK, st.p);
        return { vocab: vocab, base: base, temp: temp, afterK: afterK, final: afterP, k: k, note: note, context: context };
      }
      function lookup(pairs) { var m = {}; pairs.forEach(function (kv) { m[kv[0]] = kv[1]; }); return m; }

      // ---- controls ----
      var sources = [['logits', 'Capital of France'], ['table', 'Word table']];
      if (toy) sources.push(['toy', 'Toy transformer']);
      var sourceSeg = ui.seg({ label: 'Where the scores come from', options: sources, value: st.source, onchange: function (v) { st.source = v; st.k = null; resetDraws(); layout(); render(); } });
      var promptBox = h('div');
      var tSlider = ui.slider({ label: 'Temperature', min: 0.1, max: 3, step: 0.1, value: st.t, format: function (v) { return py.fixed(v, 1); },
        ticks: ['0.1 sharp', '1.0 as trained', '3.0 flat'], help: 'Divides every score before softmax. Below 1 sharpens the odds, above 1 flattens them.',
        oninput: function (v) { st.t = py.round(v, 1); resetDraws(); render(); } });
      var kSlider = ui.slider({ label: 'Top-k', min: 1, max: 5, step: 1, value: 5, format: function (v) { return v >= Number(kSlider ? kSlider.input.max : 5) ? 'off' : String(v); },
        help: 'Keep only the k most likely tokens. At the far right every token stays.',
        oninput: function (v) { st.k = v >= Number(kSlider.input.max) ? null : v; resetDraws(); render(); } });
      var pSlider = ui.slider({ label: 'Top-p (nucleus)', min: 0.05, max: 1, step: 0.05, value: 1, format: function (v) { return v >= 1 ? 'off' : py.fixed(v, 2); },
        help: 'Keep the smallest group of top tokens whose odds add up to p.',
        oninput: function (v) { st.p = v >= 1 ? null : py.round(v, 2); resetDraws(); render(); } });
      var controls = h('div.controls', h('div.control', h('span.lbl', 'Where the scores come from'), sourceSeg), promptBox, tSlider, kSlider, pSlider,
        h('button.btn.small.ghost', { type: 'button', onclick: function () { st.t = 1; st.k = null; st.p = null; tSlider.set(1); pSlider.set(1); resetDraws(); layout(); render(); } }, icon('redo', 'sm'), 'Reset sliders'));

      function layout() {
        ui.clear(promptBox);
        if (st.source === 'logits') {
          promptBox.appendChild(h('div.control', h('span.lbl', 'Prompt'), h('p.textin', { style: { background: 'var(--surface-2)' } }, 'The capital of France is …'),
            h('p.help', 'A fixed example: five raw scores (logits) a model might give for the next word.')));
        } else {
          var isToy = st.source === 'toy', inputId = ui.id('prompt');
          var input = h('input.textin', { id: inputId, type: 'text', value: isToy ? st.toyPrompt : st.prompt, autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
          input.addEventListener('input', function () { if (isToy) st.toyPrompt = input.value; else st.prompt = input.value; st.k = null; resetDraws(); syncK(); render(); });
          var words = isToy ? td.VOCAB : tableWords;
          promptBox.appendChild(h('div.control', h('label', { 'for': inputId }, h('span', 'Type a prompt')), input,
            h('div.wordchips', { role: 'group', 'aria-label': 'Add a word' }, words.map(function (w) {
              return h('button', { type: 'button', onclick: function () { input.value = (py.strip(input.value) + ' ' + w).replace(/^ /, ''); input.dispatchEvent(new root.Event('input')); } }, w);
            }), h('button', { type: 'button', 'aria-label': 'Clear the prompt', onclick: function () { input.value = ''; input.dispatchEvent(new root.Event('input')); } }, 'clear')),
            h('p.help', isToy ? 'A toy transformer with an 8-word vocabulary and hand-picked weights reads the whole sentence.' : 'This toy table only looks at your last word. It knows ' + tableWords.length + ' words.')));
        }
        syncK();
        genBox.hidden = st.source === 'logits';
      }
      function syncK() {
        var d = compute(), n = d.error ? 5 : d.vocab.length;
        kSlider.setMax(Math.max(n, 2), st.k === null ? Math.max(n, 2) : Math.min(st.k, n));
      }

      // ---- output ----
      var flow = h('p.flow', { 'aria-hidden': 'true' });
      var summary = h('p.summary-line', { role: 'status' });
      var chart = h('div.bars');
      var noteEl = h('p.note-inline', { hidden: true });
      var mathBody = h('div');
      var drawOut = h('div.result', { hidden: true });
      var genOut = h('div');
      var seedId = ui.id('seed');
      var seedInput = h('input.textin', { id: seedId, type: 'number', min: 0, max: 9999, step: 1, value: 0, style: { width: '96px', 'min-height': '36px' } });
      seedInput.addEventListener('input', function () { var v = Math.floor(Number(seedInput.value)); st.seed = v >= 0 && v <= 0xffffffff ? v : 0; resetDraws(); render(); });
      var genBox = h('div.result', h('p.k', 'Write a whole sentence, three ways'),
        h('div.row-btns',
          h('button.btn.small', { type: 'button', onclick: function () { generate('greedy'); } }, 'Greedy'),
          h('button.btn.small', { type: 'button', onclick: function () { generate('sample'); } }, 'Sample (uses the sliders)'),
          h('button.btn.small', { type: 'button', onclick: function () { generate('beam'); } }, 'Beam search, width 2')), genOut);

      function resetDraws() { st.draws = []; st.rng = null; drawOut.hidden = true; ui.clear(genOut); }
      function draw(n) {
        var d = compute(); if (d.error) return;
        if (!st.rng) st.rng = new py.Random(st.seed);
        var toks = d.final.map(function (kv) { return kv[0]; }), w = d.final.map(function (kv) { return kv[1]; });
        for (var i = 0; i < n; i++) st.draws.push(st.rng.choice(toks, w));
        render();
      }
      function seqEl(prefix, tokens) {
        return h('p.seq', prefix.map(function (t) { return h('span.tk', t); }), tokens.map(function (t) { return h('span.tk.new', t); }));
      }
      function generate(kind) {
        ui.clear(genOut);
        if (st.source === 'table') {
          var words = py.split(st.prompt.toLowerCase()), start = words.length ? words[words.length - 1] : '<start>';
          if (kind === 'greedy') { genOut.appendChild(seqEl(words, S.greedy(dd.NEXT, start))); genOut.appendChild(h('p.help', 'Greedy always takes the single most likely next word.')); }
          else if (kind === 'sample') {
            genOut.appendChild(seqEl(words, S.sample(dd.NEXT, { start: start, k: st.k, p: st.p, temperature: st.t, rng: new py.Random(st.seed) })));
            genOut.appendChild(h('p.help', 'Sampled with temperature ' + py.fixed(st.t, 1) + ', top-k ' + (st.k === null ? 'off' : st.k) + ', top-p ' + (st.p === null ? 'off' : py.fixed(st.p, 2)) + ', seed ' + st.seed + '. Change the seed for another roll.'));
          } else {
            var b = S.beamSearch(dd.NEXT, 2, start), g = S.greedy(dd.NEXT, start);
            genOut.appendChild(seqEl(words, b[0]));
            genOut.appendChild(h('p.help', 'Beam total log-probability ' + py.fixed(b[1], 3) + ' (probability ' + py.fixed(Math.exp(b[1]), 3) + '). Greedy scores ' +
              py.fixed(S.sequenceLogprob(dd.NEXT, g, start), 3) + ' (' + py.fixed(Math.exp(S.sequenceLogprob(dd.NEXT, g, start)), 3) + '). ' +
              (b[0].join(' ') === g.join(' ') ? 'Here both find the same sentence.' : 'Beam found a more likely whole sentence than greedy.')));
          }
        } else if (st.source === 'toy') {
          var toks = toy.tokenize(st.toyPrompt);
          if (!toks.length || toy.unknown(toks).length) return;
          if (kind !== 'greedy') { genOut.appendChild(h('p.help', 'The toy transformer demo only does greedy decoding. Use the Word table for sampling and beam search.')); return; }
          var res = toy.generate(st.toyPrompt, 4);
          genOut.appendChild(seqEl(toks, res.tokens.slice(toks.length)));
          genOut.appendChild(h('p.help', 'Each new word is fed back in and the whole model runs again (autoregressive generation). It stops at "." or after 4 words.'));
        }
      }

      function render() {
        var d = compute();
        if (d.error) {
          noteEl.hidden = false; noteEl.textContent = d.error; ui.clear(chart); delete chart.dataset.sig; summary.textContent = ''; ui.clear(mathBody); return;
        }
        noteEl.hidden = !d.note; noteEl.textContent = d.note || '';
        var base = lookup(d.base), temp = lookup(d.temp), ak = lookup(d.afterK), fin = lookup(d.final);
        var last = st.draws.length ? st.draws[st.draws.length - 1] : null;
        ui.bars(chart, d.vocab.map(function (tok) {
          var cut = fin[tok] !== undefined ? null : (temp[tok] === undefined ? 'probability 0' : ak[tok] === undefined ? 'cut by top-k' : 'cut by top-p');
          return { label: tok, value: fin[tok] || 0, ghost: base[tok], cut: cut, picked: last === tok };
        }));
        var best = d.final[0];
        d.final.forEach(function (kv) { if (kv[1] > best[1]) best = kv; });
        ui.clear(summary);
        summary.appendChild(h('span', 'Most likely next token: ', h('b', best[0]), ' at ' + py.fixed(best[1] * 100, 1) + '%. ' +
          d.final.length + ' of ' + d.vocab.length + ' tokens can be picked.'));
        ui.clear(flow);
        [[st.source === 'table' ? 'table row' : 'logits', 'on'], ['÷ T = ' + py.fixed(st.t, 1), st.t === 1 ? '' : 'on'], ['softmax', 'on'],
          ['top-k' + (d.k === null ? '' : ' = ' + d.k), d.k === null ? 'off' : 'on'], ['top-p' + (st.p === null ? '' : ' = ' + py.fixed(st.p, 2)), st.p === null ? 'off' : 'on'],
          ['rescale to 1', d.k === null && st.p === null ? 'off' : 'on'], ['pick', 'on']].forEach(function (b, i) {
          if (i) flow.appendChild(h('span.arr', '→'));
          flow.appendChild(h('span.box' + (b[1] ? '.' + b[1] : ''), b[0]));
        });
        // the math table
        ui.clear(mathBody);
        var head = ['token'];
        if (st.source !== 'table') head.push('logit', 'logit ÷ T');
        head.push(st.source === 'table' ? 'table (T = 1)' : 'softmax at T = 1', 'at T = ' + py.fixed(st.t, 1), 'after top-k', 'after top-p (final)');
        var logits = st.source === 'logits' ? sd.LOGITS : st.source === 'toy' ? toy.forward(toy.toIds(toy.tokenize(st.toyPrompt))).logits : null;
        var cell = function (v) { return h('td.num', v === undefined ? '–' : py.fixed(v, 3)); };
        mathBody.appendChild(h('table.data', h('thead', h('tr', head.map(function (t, i) { return h('th' + (i ? '.num' : ''), { scope: 'col' }, t); }))),
          h('tbody', d.vocab.map(function (tok, i) {
            return h('tr', h('th', { scope: 'row' }, h('code', tok)), logits ? [cell(logits[i]), cell(logits[i] / st.t)] : null, cell(base[tok]), cell(temp[tok]), cell(ak[tok]), cell(fin[tok]));
          }))));
        mathBody.appendChild(h('p.src', st.source === 'table'
          ? 'Temperature on a probability table: take log(p), divide by T, softmax again (apply_temperature in llm_decoding.py). A dash means the token was removed.'
          : 'softmax(x) = exp(x) / sum of exp(x). A dash means the token was removed, then the rest are divided by what is left so they add up to 1.'));
        // draws
        if (st.draws.length) {
          var counts = {}; st.draws.forEach(function (t) { counts[t] = (counts[t] || 0) + 1; });
          drawOut.hidden = false; ui.clear(drawOut);
          drawOut.appendChild(h('p.k', st.draws.length + (st.draws.length === 1 ? ' draw' : ' draws') + ' with seed ' + st.seed));
          drawOut.appendChild(h('p.seq', d.final.map(function (kv) { return h('span.tk' + (last === kv[0] ? '.new' : ''), kv[0] + ' × ' + (counts[kv[0]] || 0)); })));
          drawOut.appendChild(h('p.help', 'Last pick: ' + last + '. The same seed and settings always give the same picks, here and in Python’s random.Random(' + st.seed + ').'));
        }
      }

      el.appendChild(ui.honest('A teaching model with hand-picked numbers, not a real LLM. A real model scores tens of thousands of tokens with a neural network; here the scores are a short list.', sd.src, ctx.repo));
      el.appendChild(h('div.lab-grid', controls,
        h('div', flow, summary, noteEl, chart,
          h('p.legend', h('span', h('i.sw'), 'chance of being picked now'), h('span', h('i.gh'), 'dashed line: before the sliders (T = 1, no cuts)')),
          h('div.result', h('p.k', 'Roll the dice'),
            h('div.row-btns', { style: { 'align-items': 'center' } },
              h('button.btn.small.primary', { type: 'button', onclick: function () { draw(1); } }, 'Sample 1 token'),
              h('button.btn.small', { type: 'button', onclick: function () { draw(100); } }, 'Sample 100'),
              h('label', { 'for': seedId, style: { 'font-size': 'var(--fs-sm)', 'font-weight': '650' } }, 'Seed'), seedInput)),
          drawOut, genBox,
          ui.disclosure('Show the math', mathBody))));
      layout(); render();
    }
  };
})(typeof self !== 'undefined' ? self : globalThis);
