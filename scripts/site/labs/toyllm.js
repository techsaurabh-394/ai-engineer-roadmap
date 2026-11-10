/* Toy LLM lab: one sentence through every stage of a tiny transformer, then the next word.
   Maths: scripts/site/core/toyllm.js (a tested port of llm_e2e_walkthrough.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var FEATURES = ['thing', 'action', 'place', 'glue'];      // the meaning of the 4 numbers, from the comments in the Python file

  AIER.labs.toyllm = {
    mount: function (el, ctx) {
      var td = ctx.data.toyllm, model = new AIER.toyllm.ToyLLM(td);
      var st = { text: td.example, added: 0 };
      var inputId = ui.id('toy');
      var input = h('input.textin', { id: inputId, type: 'text', value: st.text, autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
      input.addEventListener('input', function () { st.text = input.value; st.added = 0; render(); });
      var out = h('div.toy-stages'), err = h('p.note-inline', { hidden: true, role: 'status' });
      function setText(t, added) { st.text = t; st.added = added; input.value = t; render(); }

      function stage(n, title, line, visual, numbers) {
        return h('section.toy-stage', h('span.badge', { 'aria-hidden': 'true' }, String(n)),
          h('div.toy-body', h('h3', title), h('p.line', line), visual, numbers ? ui.disclosure('Show the numbers', numbers) : null));
      }
      function tokenChips(tokens, ids, newFrom) {
        return h('p.seq', tokens.map(function (t, i) { return h('span.tk' + (newFrom !== undefined && i >= newFrom ? '.new' : ''), t, ids ? h('sub', ' ' + ids[i]) : null); }));
      }

      function render() {
        ui.clear(out);
        var tokens = model.tokenize(st.text), unknown = model.unknown(tokens);
        if (!tokens.length || unknown.length) {
          err.hidden = false;
          err.textContent = !tokens.length ? 'Add a word to start.' : 'This toy model only knows 8 words. Not in its vocabulary: ' + unknown.join(', ') + '.';
          return;
        }
        err.hidden = true;
        if (tokens.length > 12) { err.hidden = false; err.textContent = 'Keep it to 12 words so the tables stay readable.'; return; }
        var ids = model.toIds(tokens), s = model.forward(ids), V = td.VOCAB, last = tokens.length - 1;
        var mx = function (m, cols) { return h('div', { style: { 'overflow-x': 'auto' } }, ui.matrix(m, { rows: tokens, cols: cols || FEATURES, hl: last })); };

        out.appendChild(stage(1, 'Text becomes tokens, tokens become IDs', 'The tokenizer splits on spaces and looks each word up in a fixed list of ' + V.length + ' words. The model never sees letters, only these numbers.',
          tokenChips(tokens, ids, tokens.length - st.added), [h('p', 'Vocabulary: ' + V.map(function (w, i) { return w + ' = ' + i; }).join(', ') + '.')]));
        out.appendChild(stage(2, 'Each ID becomes a vector (embedding)', 'A lookup table gives every token 4 numbers. In this toy they mean: thing, action, place, grammar glue. Real models learn hundreds of numbers nobody named.',
          null, [mx(s.embeddings)]));
        out.appendChild(stage(3, 'Add position', 'Attention alone cannot tell "cat sat" from "sat cat". A small position pattern (sines and cosines, scaled by ' + td.POS_SCALE + ') is added to each vector.',
          null, [mx(s.with_position)]));
        s.block_outputs.forEach(function (blockOut, b) {
          var w = s.attention_weights[b][last], bars = h('div.bars');
          ui.bars(bars, tokens.map(function (t, j) { return { label: t + (tokens.indexOf(t) !== tokens.lastIndexOf(t) ? ' #' + (j + 1) : ''), value: w[j], ghost: null }; }));
          out.appendChild(stage(4 + b, 'Transformer block ' + (b + 1) + ': attention, then a small network',
            'Every token looks back at the tokens before it and mixes in their information. The bars show where the last token, "' + tokens[last] + '", is looking.',
            bars, [h('p', 'All vectors after block ' + (b + 1) + ' (the last row is the one used to predict):'), mx(blockOut)]));
        });
        var n = 4 + s.block_outputs.length;
        var probBars = h('div.bars');
        ui.bars(probBars, V.map(function (wd, i) { return { label: wd, value: s.probs[i], ghost: null, picked: i === s.next_id }; }));
        out.appendChild(stage(n, 'Score every word, then softmax', 'The last token’s vector is multiplied by an output matrix: one score (logit) per vocabulary word. Softmax turns the scores into probabilities that add up to 1.',
          probBars, [h('table.data', h('thead', h('tr', h('th', 'word'), h('th.num', 'logit'), h('th.num', 'probability'))),
            h('tbody', V.map(function (wd, i) { return h('tr', h('th', { scope: 'row' }, h('code', wd)), h('td.num', py.fixed(s.logits[i], 3)), h('td.num', py.fixed(s.probs[i], 3))); })))]));
        var next = V[s.next_id], done = next === '.';
        out.appendChild(stage(n + 1, 'Pick the next token', 'Greedy decoding takes the most likely word. Then the whole sentence, with the new word on the end, goes through the model again.',
          h('div', h('p.seq', tokens.map(function (t) { return h('span.tk', t); }), h('span.tk.new', next), h('span', { style: { 'margin-left': '8px', 'font-family': 'var(--font)' } }, 'p = ' + py.fixed(s.probs[s.next_id], 3))),
            h('div.row-btns',
              h('button.btn.small.primary', { type: 'button', disabled: tokens.length >= 12, onclick: function () { setText(tokens.concat([next]).join(' '), st.added + 1); } }, 'Add "' + next + '" and run again'),
              AIER.labs.sampling ? h('a.btn.small', { href: '#lab/sampling' }, icon('flask', 'sm'), 'Sample instead of greedy') : null,
              h('button.btn.small.ghost', { type: 'button', onclick: function () { setText(td.example, 0); } }, icon('redo', 'sm'), 'Start over')),
            done ? h('p.help', 'The model predicted ".", which is where the Python demo stops generating.') : null), null));
      }

      el.appendChild(ui.honest('A toy with ' + model.parameterCount() + ' hand-picked numbers, an 8-word vocabulary and nothing trained. A real LLM has the same stages with billions of learned numbers.', td.src, ctx.repo));
      el.appendChild(h('div.control', { style: { 'margin-bottom': '16px' } }, h('label', { 'for': inputId }, h('span', 'Build a sentence from the 8 words it knows')), input,
        h('div.wordchips', { role: 'group', 'aria-label': 'Add a word' }, td.VOCAB.map(function (w) {
          return h('button', { type: 'button', onclick: function () { setText((py.strip(st.text) + ' ' + w).replace(/^ /, ''), 0); } }, w);
        }), h('button', { type: 'button', onclick: function () { setText('', 0); } }, 'clear'))));
      el.appendChild(err); el.appendChild(out);
      render();
    }
  };

  AIER.labs.toyllm.tour = function (el) {
    var st = function (n) { return function () { return el.querySelectorAll('.toy-stage')[n]; }; };
    return [
      { el: function () { return el.querySelector('.control'); }, title: 'Write a sentence', text: 'This toy model knows 8 words. Tap them to build a sentence, or type.' },
      { el: st(0), title: 'Words become numbers', text: 'Each word is replaced by its ID in the vocabulary. The model never sees letters.' },
      { el: st(3), title: 'Attention inside a block', text: 'The bars show where the last word looks. Two blocks repeat the same idea, each refining the vectors.' },
      { el: st(5), title: 'Odds for every word', text: 'The last vector is turned into one score per word, then softmax makes them probabilities.' },
      { el: st(6), title: 'Pick, append, repeat', text: 'The most likely word is added and the whole sentence runs through again. That loop is how every LLM writes.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
