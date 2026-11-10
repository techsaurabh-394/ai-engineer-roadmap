/* Search lab: one query through BM25, the toy "dense" search and hybrid (reciprocal rank fusion).
   Maths: scripts/site/core/search.js (a tested port of rag_bm25.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, Q = AIER.search, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var PRESETS = ['rotate refresh token', 'token lifetime', 'age', 'ETag 304', 'caching'];

  AIER.labs.search = {
    mount: function (el, ctx) {
      var sdta = ctx.data.search, docs = sdta.DOCS, stop = sdta.STOP, texts = docs.map(function (d) { return d.text; });
      var st = { q: PRESETS[0], team: '', year: '' };
      if (ctx.lesson && ctx.lesson.name === '20-metadata-filtering') st.year = '2025';
      var teams = [], years = [];
      docs.forEach(function (d) { if (teams.indexOf(d.meta.team) === -1) teams.push(d.meta.team); if (years.indexOf(d.meta.year) === -1) years.push(d.meta.year); });
      years.sort();

      var inputId = ui.id('q');
      var input = h('input.textin', { id: inputId, type: 'search', value: st.q, autocomplete: 'off', spellcheck: 'false' });
      input.addEventListener('input', function () { st.q = input.value; render(); });
      var teamSel = h('select', { 'aria-label': 'Team filter', onchange: function () { st.team = teamSel.value; render(); } }, h('option', { value: '' }, 'any team'), teams.map(function (t) { return h('option', { value: t }, 'team = ' + t); }));
      var yearSel = h('select', { 'aria-label': 'Year filter', onchange: function () { st.year = yearSel.value; render(); } }, h('option', { value: '' }, 'any year'), years.map(function (y) { return h('option', { value: y }, 'year ≥ ' + y); }));
      yearSel.value = st.year;
      var explain = h('p.summary-line', { role: 'status' });
      var cols = h('div.search-cols');
      var math = h('div');

      function highlight(text, terms) {
        var out = h('span'), re = /[A-Za-z0-9]+/g, last = 0, m;
        while ((m = re.exec(text)) !== null) {
          if (terms.indexOf(m[0].toLowerCase()) !== -1) {
            out.appendChild(root.document.createTextNode(text.slice(last, m.index)));
            out.appendChild(h('mark', m[0])); last = m.index + m[0].length;
          }
        }
        out.appendChild(root.document.createTextNode(text.slice(last)));
        return out;
      }
      function column(title, sub, rows, empty) {
        var list = h('ol.hits');
        rows.forEach(function (r, i) {
          list.appendChild(h('li' + (i === 0 ? '.top' : ''), h('span.rk', { 'aria-hidden': 'true' }, String(i + 1)),
            h('span.body', h('span.doc', r.body), h('span.meta', 'doc ' + r.id + ' · ' + docs[r.id].meta.source + ' · ' + docs[r.id].meta.team + ' · ' + docs[r.id].meta.year + ' · ', h('b', r.score)))));
        });
        return h('section.search-col', h('h3', title), h('p.help', sub), rows.length ? list : h('p.none', empty));
      }

      function render() {
        var filters = {};
        if (st.team) filters.team = st.team;
        if (st.year) filters.year = { op: 'gte', target: Number(st.year) };
        var hyb = Q.hybridSearch(st.q, docs, stop, docs.length, filters), allowed = hyb.allowed;
        var engine = new Q.BM25(texts, stop), terms = engine.tokenize(st.q);
        var filteredOut = allowed === null ? 0 : docs.length - allowed.length;
        var sparse = allowed !== null && !allowed.length ? [] : engine.search(st.q, docs.length, allowed);
        var dense = allowed !== null && !allowed.length ? [] : Q.toyDenseScores(st.q, texts, allowed).filter(function (x) { return x[0] > 0; });
        var rrf = new Map(), fused = Q.reciprocalRankFusion([hyb.sparse, hyb.dense], 60, rrf);

        ui.clear(cols);
        cols.appendChild(column('Keyword search (BM25)', 'Scores documents that contain the exact query words.',
          sparse.map(function (x) { return { id: x[1], body: highlight(texts[x[1]], terms), score: 'score ' + py.fixed(x[0], 3) }; }),
          terms.length ? 'No document contains ' + (terms.length === 1 ? 'the word "' + terms[0] + '"' : 'any of these exact words') + '.' : 'Type a query. Very common words like "the" are ignored.'));
        cols.appendChild(column('Toy vector search', 'Compares 3-letter pieces of words, so "token" is close to "tokens". Not real embeddings.',
          dense.map(function (x) { return { id: x[1], body: texts[x[1]], score: 'similarity ' + py.fixed(x[0], 3) }; }), 'Nothing looks similar.'));
        cols.appendChild(column('Hybrid (rank fusion)', 'Each list gives a document 1 / (60 + its rank). Add them up.',
          fused.map(function (id) {
            var a = hyb.sparse.indexOf(id), b = hyb.dense.indexOf(id);
            return { id: id, body: texts[id], score: 'RRF ' + py.fixed(rrf.get(id), 4) + ' (keyword ' + (a === -1 ? 'none' : '#' + (a + 1)) + ', vector ' + (b === -1 ? 'none' : '#' + (b + 1)) + ')' };
          }), 'Nothing to merge.'));

        ui.clear(explain);
        var msg;
        if (allowed !== null && !allowed.length) msg = 'The filter removed every document before any ranking happened.';
        else if (!sparse.length && !dense.length) msg = 'Neither search found anything.';
        else if (!sparse.length) msg = 'Keyword search found nothing, because no document has these exact words. The toy vector search still found look-alike words, so hybrid has an answer. Check it: look-alike is not always right.';
        else if (!dense.length) msg = 'Only keyword search found something, so hybrid simply returns its list.';
        else if (sparse[0][1] !== dense[0][1]) msg = 'The two searches disagree about the best document (keyword: doc ' + sparse[0][1] + ', vector: doc ' + dense[0][1] + '). Hybrid puts doc ' + fused[0] + ' first because it ranks high in both lists.';
        else if (sparse.length > 1 && sparse[0][0] === sparse[1][0]) msg = 'Keyword search cannot choose: its top documents have the same score. The vector list breaks the tie in the hybrid result.';
        else msg = 'Both agree on doc ' + sparse[0][1] + '. Keyword search stops there; the vector search also returns ' + (dense.length - 1) + ' weaker look-alikes, and hybrid keeps them below the winner.';
        explain.appendChild(h('span', msg, filteredOut ? ' The filter removed ' + filteredOut + ' of ' + docs.length + ' documents BEFORE ranking.' : ''));

        ui.clear(math);
        if (sparse.length) {
          var detail = [], top = sparse[0][1];
          engine.score(st.q, top, detail);
          math.appendChild(h('p', h('b', 'BM25 for doc ' + top + ': '), 'for each query word in the document, IDF × tf × (k1 + 1) / (tf + k1 × (1 − b + b × length / average length)), with k1 = ' + engine.k1 + ', b = ' + engine.b +
            ', length = ' + engine.lengths[top] + ', average = ' + py.fixed(engine.avgdl, 2) + '.'));
          math.appendChild(h('table.data', h('thead', h('tr', h('th', 'word'), h('th.num', 'times in doc (tf)'), h('th.num', 'docs with it'), h('th.num', 'IDF'), h('th.num', 'adds'))),
            h('tbody', detail.map(function (d) { return h('tr', h('th', { scope: 'row' }, h('code', d.term)), h('td.num', String(d.tf)), h('td.num', String(engine.df.get(d.term))), h('td.num', py.fixed(d.idf, 3)), h('td.num', py.fixed(d.part, 3))); }),
              h('tr', h('th', { scope: 'row' }, 'total'), h('td'), h('td'), h('td'), h('td.num', py.fixed(sparse[0][0], 3))))));
        }
        if (fused.length) {
          math.appendChild(h('p', { style: { 'margin-top': '12px' } }, h('b', 'Rank fusion: ')));
          math.appendChild(h('table.data', h('thead', h('tr', h('th', 'doc'), h('th.num', 'from keyword list'), h('th.num', 'from vector list'), h('th.num', 'sum'))),
            h('tbody', fused.map(function (id) {
              var a = hyb.sparse.indexOf(id), b = hyb.dense.indexOf(id);
              return h('tr', h('th', { scope: 'row' }, 'doc ' + id), h('td.num', a === -1 ? '0' : '1/(60+' + (a + 1) + ') = ' + py.fixed(1 / (60 + a + 1), 4)),
                h('td.num', b === -1 ? '0' : '1/(60+' + (b + 1) + ') = ' + py.fixed(1 / (60 + b + 1), 4)), h('td.num', py.fixed(rrf.get(id), 4)));
            }))));
        }
        if (!sparse.length && !fused.length) math.appendChild(h('p', 'Nothing to show for this query.'));
      }

      el.appendChild(ui.honest('Five one-sentence documents, and the "vector" side is a character-trigram stand-in so it runs offline. Real dense search uses an embedding model and understands meaning far better.', sdta.src, ctx.repo));
      el.appendChild(h('div.controls', { style: { 'margin-bottom': '16px' } },
        h('div.control', h('label', { 'for': inputId }, h('span', 'Search these 5 documents')), input,
          h('div.wordchips', { role: 'group', 'aria-label': 'Example queries' }, PRESETS.map(function (p) { return h('button', { type: 'button', onclick: function () { input.value = p; st.q = p; render(); } }, p); }))),
        h('div.control', h('span.lbl', 'Metadata filter (applied before ranking)'), h('div.row-btns', { style: { 'margin-top': '0' } }, teamSel, yearSel))));
      el.appendChild(explain); el.appendChild(cols);
      el.appendChild(ui.disclosure('Show the math', math));
      el.appendChild(h('p.help', { style: { 'margin-top': '12px' } }, 'Hybrid is not magic. It helps when the two searches fail on different queries, which on real data they usually do: keyword search misses reworded questions, vector search can miss exact codes and names.'));
      render();
    }
  };
})(typeof self !== 'undefined' ? self : globalThis);
