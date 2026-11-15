/* RAG pipeline lab: type a question and walk it through every stage of a small, real RAG pipeline.
   Maths: scripts/site/core/ragpipe.js (a tested port of rag_pipeline.py). Play, pause, step back and forward. */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, P = AIER.ragpipe, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var PRESETS = ['How should refresh tokens be stored?', 'What is an ETag?', 'Why cap the steps of an agent loop?', 'quantum entanglement'];
  var STAGES = [
    { id: 'ask', name: 'Question', real: true, why: 'Everything starts with the words the user typed.',
      what: 'The raw question arrives. Nothing has been changed yet.' },
    { id: 'query', name: 'Query processing', real: true, why: 'Search compares words, so the question is cleaned first: lowercase, split into words, drop filler words such as "how" and "the", strip a plural "s".',
      what: 'The words kept here are the only ones that can match anything below.' },
    { id: 'embed', name: 'Embedding', real: true, why: 'A computer can compare numbers, not sentences. The question becomes a vector with one number per vocabulary word.',
      what: 'Here the vector is TF-IDF: a rare word counts for more than a common one. A real embedding model produces a dense vector that also captures meaning.' },
    { id: 'search', name: 'Vector search', real: true, why: 'Score every chunk by cosine similarity with the question vector, then keep the top k.',
      what: 'cosine = how closely two vectors point the same way (1 = same direction, 0 = nothing in common).' },
    { id: 'rerank', name: 'Reranking', real: true, why: 'Vector search is fast but rough. A second pass re-scores only the few candidates with a closer look.',
      what: 'This teaching reranker mixes the cosine score with how many question terms the chunk contains. A real one is a cross-encoder model that reads question and chunk together.' },
    { id: 'context', name: 'Context', real: true, why: 'The best chunks are pasted into the prompt as numbered sources. This is the only knowledge the model gets about your documents.',
      what: 'The prompt below is exactly what would be sent to a language model.' },
    { id: 'answer', name: 'Answer', real: false, why: 'A language model writes the final answer from the sources. This lab has no model, so it shows the sentence from the sources that shares the most words with the question.',
      what: 'SIMULATED: a real LLM would rephrase and combine sources. The citation [n] points to the numbered source.' }
  ];

  function vecBars(vec, vocab, max) {
    var rows = vec.map(function (v, i) { return [v, vocab[i]]; }).filter(function (x) { return x[0] > 0; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, max || 8);
    if (!rows.length) return h('p.none', 'No word of the question is in the vocabulary, so the vector is all zeros. Nothing can match.');
    var top = rows[0][0];
    return h('div.vbars', { role: 'list' }, rows.map(function (r) {
      return h('div.vb', { role: 'listitem' }, h('span.w', r[1]), h('span.t', h('i', { style: { width: (r[0] / top * 100) + '%' } })), h('span.n', py.fixed(r[0], 3)));
    }));
  }

  AIER.labs.ragpipe = {
    mount: function (el, ctx) {
      var cd = ctx.data.ragpipe, st = { q: PRESETS[0], size: 160, k: 3, top: 2, stage: 0, timer: null };
      var res = null;
      var inputId = ui.id('q');
      var input = h('input.textin', { id: inputId, type: 'text', value: st.q, autocomplete: 'off', spellcheck: 'false' });
      var strip = h('ol.pipe-strip', { 'aria-label': 'Pipeline stages' });
      var detail = h('div.pipe-detail', { 'aria-live': 'polite' });
      var playBtn = h('button.btn.small.primary', { type: 'button' });
      var backBtn = h('button.btn.small', { type: 'button' }, icon('prev', 'sm'), 'Back');
      var nextBtn = h('button.btn.small', { type: 'button' }, 'Next', icon('next', 'sm'));
      var resetBtn = h('button.btn.small.ghost', { type: 'button' }, icon('redo', 'sm'), 'Reset');

      function compute() { res = P.run(st.q, cd.DOCS, cd.STOP, st.size, st.k, st.top); }
      function setPlay(on) {
        ui.clear(playBtn); playBtn.appendChild(icon(on ? 'pause' : 'play', 'sm')); playBtn.appendChild(h('span', on ? 'Pause' : 'Play'));
      }
      function stop() { if (st.timer) { root.clearInterval(st.timer); st.timer = null; } setPlay(false); }
      function go(n) { st.stage = Math.max(0, Math.min(STAGES.length - 1, n)); paint(); }
      playBtn.addEventListener('click', function () {
        if (st.timer) { stop(); return; }
        if (st.stage >= STAGES.length - 1) st.stage = 0;
        setPlay(true); paint();
        st.timer = root.setInterval(function () { if (st.stage >= STAGES.length - 1) { stop(); return; } go(st.stage + 1); if (st.stage >= STAGES.length - 1) stop(); }, ui.reducedMotion() ? 2600 : 2000);
      });
      backBtn.addEventListener('click', function () { stop(); go(st.stage - 1); });
      nextBtn.addEventListener('click', function () { stop(); go(st.stage + 1); });
      resetBtn.addEventListener('click', function () { stop(); go(0); });
      input.addEventListener('input', function () { st.q = input.value; compute(); paint(); });

      function chunkCard(it, extra, cls) {
        return h('li.chunk' + (cls ? '.' + cls : ''), h('p.hd', h('span', 'chunk ' + it.id + ' · ' + it.doc), h('span', extra || '')), h('p.txt', it.text));
      }
      function body() {
        var s = STAGES[st.stage].id, items = res.index.items, out = [];
        if (s === 'ask') {
          out.push(h('p.big', '“' + st.q + '”'));
          out.push(h('p.help', 'The knowledge base has ' + cd.DOCS.length + ' short documents. At indexing time (done once, before any question) they were cut into ' + items.length + ' chunks and every chunk was embedded.'));
          out.push(h('details.math', h('summary', icon('chevron', 'sm'), h('span', 'See the ' + items.length + ' chunks that were indexed')), h('div.body', h('ol.chunk-list', items.map(function (it) { return chunkCard(it, it.text.length + ' chars'); })))));
        } else if (s === 'query') {
          var kept = res.terms;
          out.push(h('p.help', 'Words in the question:'));
          out.push(h('div.chips', res.words.map(function (w) { return h('span.chip' + (res.index.stop.indexOf(w) === -1 ? '.accent' : ''), { title: res.index.stop.indexOf(w) === -1 ? 'kept' : 'dropped: stop word' }, w); })));
          out.push(h('p.summary-line', h('b', String(kept.length)), ' terms kept: ', kept.length ? kept.join(', ') : 'none. Every word was a stop word.'));
          if (kept.length) out.push(h('p.help', 'Highlighted words are kept. Grey words are stop words and are dropped. A kept word that ends in “s” loses it (tokens → token).'));
        } else if (s === 'embed') {
          out.push(h('p.help', 'The question vector has ' + res.queryVector.length + ' numbers, one per word in the vocabulary. Only the question’s own words are non-zero. Biggest weights:'));
          out.push(vecBars(res.queryVector, res.index.vocab, 8));
          out.push(h('p.help', 'A word that appears in fewer chunks gets a bigger weight (IDF). The vector is scaled to length 1, so its squared weights add up to 1.'));
        } else if (s === 'search') {
          if (!res.hits.length) out.push(h('p.none', 'No chunk shares a word with the question, so every cosine is 0 and nothing is retrieved. A real system would then answer “I don’t know” rather than guess.'));
          else {
            out.push(h('p.help', 'All ' + items.length + ' chunks were scored. These are the top ' + res.hits.length + ' (cosine similarity):'));
            out.push(h('ol.chunk-list', res.hits.map(function (x, i) { return chunkCard(items[x[1]], 'cosine ' + py.fixed(x[0], 4), i === 0 ? 'top' : ''); })));
          }
        } else if (s === 'rerank') {
          if (!res.ranked.length) out.push(h('p.none', 'Nothing to rerank.'));
          else {
            out.push(h('p.help', 'score = 0.5 × term coverage + 0.5 × cosine. Coverage is the share of the question’s terms found in the chunk.'));
            var before = res.hits.map(function (x) { return x[1]; }), after = res.ranked.map(function (x) { return x[1]; });
            out.push(h('p.summary-line', before.join() === after.join() ? 'The order did not change this time.' : 'The order changed: reranking moved a chunk that vector search had placed lower.'));
            out.push(h('ol.chunk-list', res.ranked.map(function (x, i) {
              return chunkCard(items[x[1]], 'score ' + py.fixed(x[0], 4) + ' = coverage ' + py.fixed(x[2], 2) + ', cosine ' + py.fixed(x[3], 4) + (i < st.top ? ' · kept' : ' · dropped'), i < st.top ? 'top' : 'dim');
            })));
          }
        } else if (s === 'context') {
          out.push(h('p.help', 'The best ' + res.contextIds.length + ' chunk(s) go into the prompt, in order. Bigger chunks or a larger top-n means more tokens, more cost and more distraction.'));
          out.push(h('pre.out', { tabindex: '0', 'aria-label': 'The prompt sent to the model' }, res.prompt));
          out.push(h('p.help', '≈ ' + res.prompt.split(/\s+/).length + ' words in this prompt.'));
        } else {
          out.push(h('p.big', res.answer));
          out.push(h('p.honest', icon('info', 'sm'), h('span', 'Simulated: this is the best-matching sentence, not text written by a model. In production the prompt from the previous stage is sent to an LLM; see ', h('a', { href: ctx.repo + '/blob/main/01-rag-foundations/code/mini_rag.py', rel: 'noopener' }, 'mini_rag.py'), ' for the call.')));
        }
        return out;
      }
      function paint() {
        var cur = STAGES[st.stage];
        strip.querySelectorAll('button').forEach(function (b, i) {
          b.classList.toggle('on', i === st.stage); b.classList.toggle('done', i < st.stage);
          if (i === st.stage) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
        });
        ui.clear(detail);
        detail.appendChild(h('header', h('span.k', 'Stage ' + (st.stage + 1) + ' of ' + STAGES.length), h('h3', cur.name),
          h('span.chip' + (cur.real ? '.good' : '.warn'), cur.real ? 'real computation' : 'simulated')));
        detail.appendChild(h('p.why', h('b', 'Why: '), cur.why));
        detail.appendChild(h('p.what', cur.what));
        body().forEach(function (n) { detail.appendChild(n); });
        backBtn.disabled = st.stage === 0; nextBtn.disabled = st.stage === STAGES.length - 1;
      }
      STAGES.forEach(function (s, i) {
        var li = h('li', h('button', { type: 'button', onclick: function () { stop(); go(i); }, 'aria-label': 'Stage ' + (i + 1) + ': ' + s.name },
          h('span.n', { 'aria-hidden': 'true' }, String(i + 1)), h('span.l', s.name)));
        strip.appendChild(li);
      });

      var presets = h('div.chips', PRESETS.map(function (p) { return h('button.chip', { type: 'button', onclick: function () { input.value = p; st.q = p; compute(); paint(); } }, p); }));
      var sizeSl = ui.slider({ label: 'Chunk size (characters)', min: 40, max: 400, step: 20, value: st.size, help: 'Rebuilds the index: smaller chunks are more precise but carry less context.', oninput: function (v) { st.size = v; compute(); paint(); } });
      var kSl = ui.slider({ label: 'Candidates from vector search (k)', min: 1, max: 6, step: 1, value: st.k, oninput: function (v) { st.k = v; if (st.top > v) st.top = v; compute(); paint(); } });
      var topSl = ui.slider({ label: 'Chunks sent to the model', min: 1, max: 4, step: 1, value: st.top, oninput: function (v) { st.top = v; compute(); paint(); } });

      el.appendChild(ui.honest('Chunking, TF-IDF vectors, cosine search, reranking and prompt building are real computations. The embedding is word-based, not a neural model, and the answer is extracted, not generated.', cd.src, ctx.repo));
      el.appendChild(h('div.lab-grid',
        h('div.controls', h('div.control', h('label', { 'for': inputId }, h('span', 'Your question')), input, presets), sizeSl, kSl, topSl,
          h('div.row-btns', backBtn, playBtn, nextBtn, resetBtn)),
        h('div', strip, detail)));
      compute(); setPlay(false); paint();
      return stop;
    }
  };

  AIER.labs.ragpipe.tour = function (el) {
    var q = function (s) { return function () { return el.querySelector(s); }; };
    return [
      { el: q('input.textin'), title: 'Ask anything', text: 'Type a question about tokens, caching, Docker, RAG or agents, or pick one of the examples.' },
      { el: q('.pipe-strip'), title: 'Click a stage', text: 'Each button is one stage of the pipeline. Click it to see its input and output.' },
      { el: q('.row-btns'), title: 'Play, step, reset', text: 'Play runs through the stages. Back and Next move one stage. Reset starts again.' },
      { el: function () { return el.querySelectorAll('input[type=range]')[0].closest('.control'); }, title: 'Change the chunk size', text: 'The index is rebuilt with the new size. Watch what lands in the retrieved chunks.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
