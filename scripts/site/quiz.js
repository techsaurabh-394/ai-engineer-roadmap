/* Quiz page: self-graded flash cards with spaced repetition (Leitner boxes), a strengths view and
   links from every question to the animation that explains it. Progress stays in this browser. */
(function (root) {
  'use strict';
  var AIER = root.AIER, D = root.AIER_DATA, ui = AIER.ui, h = ui.h, icon = ui.icon, doc = root.document;
  var $ = function (s) { return doc.querySelector(s); };
  var Q = D.quiz, QBY = {}, LESSON = {}, PHASE = {};
  Q.forEach(function (q) { QBY[q.id] = q; });
  D.lessons.forEach(function (l) { LESSON[l.name] = l; });
  D.phases.forEach(function (p) { PHASE[p.dir] = p; });

  // ---------------------------------------------------------------- stored progress
  var KEY = 'aier-quiz-v2', OLD_KEY = 'aier-quiz-v1';
  var DAYS = [0, 1, 2, 4, 8, 16];                    // box n comes back after DAYS[n] days
  function today() { var d = new Date(); return Math.floor((d.getTime() - d.getTimezoneOffset() * 60000) / 86400000); }
  var st = ui.store.get(KEY, null);
  if (!st || typeof st !== 'object' || !st.cards) {
    st = { cards: {} };
    var old = ui.store.get(OLD_KEY, null);            // the first version stored {question index: 'k' | 'n'}
    if (old && typeof old === 'object') {
      Object.keys(old).forEach(function (i) {
        var q = Q[Number(i)];
        if (!q) return;
        if (old[i] === 'k') st.cards[q.id] = { box: 2, due: today() + DAYS[2], seen: 1, right: 1, last: 'k' };
        else if (old[i] === 'n') st.cards[q.id] = { box: 1, due: today(), seen: 1, right: 0, last: 'n' };
      });
    }
  }
  function save() { ui.store.set(KEY, st); }
  function rec(id) { return st.cards[id] || null; }
  function status(id) { var r = rec(id); return !r ? 'new' : r.last === 'k' ? 'known' : 'review'; }
  function isDue(id) { var r = rec(id); return !!r && r.due <= today(); }
  function grade(id, knew) {
    var r = rec(id) || { box: 1, due: 0, seen: 0, right: 0, last: null };
    r.seen += 1;
    if (knew) { r.right += 1; r.box = Math.min(5, r.box + 1); r.last = 'k'; r.due = today() + DAYS[r.box]; }
    else { r.box = 1; r.last = 'n'; r.due = today(); }
    st.cards[id] = r; save();
    return r;
  }
  var STATUS = { 'new': ['New', 'spark', ''], known: ['Knew it', 'check', 'good'], review: ['Review', 'redo', 'warn'] };
  function statusChip(id) {
    var s = STATUS[status(id)], r = rec(id);
    return h('span.chip' + (s[2] ? '.' + s[2] : ''), icon(s[1], 'sm'), s[0] + (r && r.last === 'k' ? ' · box ' + r.box : ''));
  }
  function phaseChip(p) { return h('span.chip', { dataset: { phase: p.id } }, h('span.phase-dot'), 'Phase ' + p.n + ' · ' + p.name); }
  function toast(msg) {
    var old = $('.toast'); if (old) old.parentNode.removeChild(old);
    var t = h('div.toast', { role: 'status' }, msg);
    doc.body.appendChild(t);
    root.setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 2400);
  }
  function when(days) { return days <= 0 ? 'today' : days === 1 ? 'tomorrow' : 'in ' + days + ' days'; }

  // ---------------------------------------------------------------- view state
  var view = 'practice', phase = 'all', lesson = null, browse = 'all';
  var session = null;      // {queue: [ids], pos, first: {id: 'k'|'n'}, shown, label}

  function pool() {
    return Q.filter(function (q) { return (phase === 'all' || q.phase === phase) && (!lesson || q.anims.indexOf(lesson) !== -1); });
  }
  function startSession(kind) {
    var items = pool(), ids, label;
    if (kind === 'missed') { ids = items.filter(function (q) { return status(q.id) === 'review'; }).map(function (q) { return q.id; }); label = 'Review what I missed'; }
    else if (kind === 'all') { ids = items.map(function (q) { return q.id; }); label = 'Everything in this set'; }
    else {
      var due = items.filter(function (q) { return isDue(q.id); }).sort(function (a, b) { return rec(a.id).box - rec(b.id).box; });
      var fresh = items.filter(function (q) { return !rec(q.id); });
      ids = due.concat(fresh).map(function (q) { return q.id; }); label = 'Due today and new';
    }
    session = { queue: ids, pos: 0, first: {}, shown: false, label: label, kind: kind, total: ids.length };
  }

  // ---------------------------------------------------------------- chrome: tabs and filters
  function tabs() {
    var missed = Q.filter(function (q) { return status(q.id) === 'review'; }).length;
    var items = [['practice', 'Practice', 'play'], ['missed', 'Review missed' + (missed ? ' (' + missed + ')' : ''), 'redo'], ['browse', 'Browse all', 'book'], ['strengths', 'Strengths', 'star']];
    var el = ui.clear($('#tabs'));
    items.forEach(function (it) {
      el.appendChild(h('button.tab', { type: 'button', 'aria-pressed': String(view === it[0]), onclick: function () { go(it[0]); } }, icon(it[2], 'sm'), it[1]));
    });
  }
  function filters() {
    var el = ui.clear($('#filters'));
    if (view === 'strengths') { el.hidden = true; return; }
    el.hidden = false;
    var mk = function (key, label, p) {
      var n = Q.filter(function (q) { return key === 'all' || q.phase === key; }).length;
      el.appendChild(h('button.fchip', { type: 'button', 'aria-pressed': String(phase === key), dataset: p ? { phase: p.id } : {},
        onclick: function () { phase = key; session = null; render(); } }, p ? h('span.phase-dot') : null, label, h('small', String(n))));
    };
    mk('all', 'All phases', null);
    D.phases.forEach(function (p) { mk(p.dir, 'Phase ' + p.n, p); });
    if (lesson && LESSON[lesson]) {
      el.appendChild(h('button.fchip.on', { type: 'button', 'aria-label': 'Showing only questions about ' + LESSON[lesson].title + '. Click to show all.',
        onclick: function () { lesson = null; session = null; render(); } }, 'Only: ' + LESSON[lesson].title, icon('close', 'sm')));
    }
  }
  function go(v) { view = v; session = null; render(); root.scrollTo(0, 0); }

  // ---------------------------------------------------------------- the animation links on a card
  function watchLinks(q) {
    var box = h('div.watch');
    q.anims.forEach(function (name, i) {
      var l = LESSON[name]; if (!l) return;
      box.appendChild(h('a.btn' + (i === 0 ? '' : '.small.ghost'), { href: '../#' + name },
        icon('play', i === 0 ? '' : 'sm'), i === 0 ? 'Watch the animation that explains this: ' + l.title : 'Also: ' + l.title));
    });
    return box;
  }

  // ---------------------------------------------------------------- practice card
  function renderCard() {
    var main = ui.clear($('#view'));
    if (!session) startSession(view === 'missed' ? 'missed' : 'practice');
    if (!session.queue.length) { renderEmpty(main); return; }
    if (session.pos >= session.queue.length) { renderDone(main); return; }
    var id = session.queue[session.pos], q = QBY[id], p = PHASE[q.phase];
    var done = Object.keys(session.first).length, pct = session.total ? done / session.total * 100 : 0;
    var known = 0, missed = 0;
    Object.keys(session.first).forEach(function (k) { if (session.first[k] === 'k') known += 1; else missed += 1; });

    var answerId = ui.id('ans'), noteId = ui.id('mine');
    var mine = h('textarea.textin', { id: noteId, rows: 3, placeholder: 'Say it in your own words first. Nothing you type is checked or saved.' });
    var answer = h('div.answer', { id: answerId, hidden: !session.shown, tabindex: '-1' },
      h('p.k', 'Model answer'), h('div.a', { html: q.a }),
      h('p.self', 'Compare it with what you had in mind, then mark yourself honestly. Only you see this.'),
      h('div.grade', h('button.btn.good', { type: 'button', id: 'knew', onclick: function () { mark(true); } }, icon('check'), 'I knew it', h('kbd', '1')),
        h('button.btn.warn', { type: 'button', id: 'notyet', onclick: function () { mark(false); } }, icon('redo'), 'Not yet', h('kbd', '2'))),
      watchLinks(q));
    var showBtn = h('button.btn.primary', { type: 'button', id: 'show', hidden: session.shown, 'aria-controls': answerId, 'aria-expanded': String(session.shown), onclick: reveal }, 'Show answer', h('kbd', 'Space'));

    function reveal() { session.shown = true; answer.hidden = false; showBtn.hidden = true; showBtn.setAttribute('aria-expanded', 'true'); answer.focus(); }
    function mark(knew) {
      var r = grade(id, knew);
      if (session.first[id] === undefined) session.first[id] = knew ? 'k' : 'n';
      if (!knew && session.queue.indexOf(id, session.pos + 1) === -1) session.queue.push(id);     // see it again this round
      toast(knew ? 'Knew it. It comes back ' + when(r.due - today()) + ' (box ' + r.box + ').' : 'Marked for review. You will see it again at the end of this round.');
      session.pos += 1; session.shown = false; render();
    }
    renderCard.reveal = reveal; renderCard.mark = mark;

    main.appendChild(h('div.session', h('div.session-top', h('span', session.label + ' · card ' + Math.min(done + 1, session.total) + ' of ' + session.total),
      h('span.tally', h('span.ok', icon('check', 'sm'), String(known)), h('span.no', icon('redo', 'sm'), String(missed)))),
      h('div.meter', { role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': session.total, 'aria-valuenow': done, 'aria-label': 'Cards done this round' }, h('i', { style: { width: pct + '%' } }))));
    main.appendChild(h('article.card.qcard', { dataset: { phase: p.id } },
      h('header', phaseChip(p), statusChip(id), session.first[id] === 'n' ? h('span.chip.warn', 'second try') : null),
      h('h2.q', { id: 'qtext', tabindex: '-1' }, q.q),
      h('label.mine-l', { 'for': noteId }, 'Your answer (optional)'), mine,
      h('div.actions', showBtn), answer,
      h('footer', h('button.btn.small.ghost', { type: 'button', id: 'prev', disabled: session.pos === 0, onclick: function () { session.pos -= 1; session.shown = false; render(); } }, icon('prev', 'sm'), 'Back'),
        h('button.btn.small.ghost', { type: 'button', id: 'skip', onclick: function () { session.pos += 1; session.shown = false; render(); } }, 'Skip', icon('next', 'sm')))));
    main.appendChild(h('p.hint', 'Answers are free text, so nothing here can check them for you: this is self-assessment. Keys: ', h('kbd', 'Space'), ' shows the answer, ', h('kbd', '1'), ' knew it, ', h('kbd', '2'), ' not yet, ', h('kbd', '←'), ' ', h('kbd', '→'), ' move.'));
  }
  function renderEmpty(main) {
    var items = pool(), next = null;
    items.forEach(function (q) { var r = rec(q.id); if (r && (next === null || r.due < next)) next = r.due; });
    if (view === 'missed') {
      main.appendChild(h('div.card.empty', icon('check'), h('h2', 'Nothing to review'), h('p', 'You have no questions marked "Not yet" in this set.'),
        h('div.row-btns', h('button.btn.primary', { type: 'button', onclick: function () { go('practice'); } }, 'Practice'), h('button.btn', { type: 'button', onclick: function () { go('strengths'); } }, 'See strengths'))));
    } else if (!items.length) {
      main.appendChild(h('div.card.empty', h('h2', 'No questions in this set'), h('p', 'Pick another phase above.')));
    } else {
      main.appendChild(h('div.card.empty', icon('check'), h('h2', 'All caught up'), h('p', 'Nothing is due in this set. The next card comes back ' + when(next - today()) + '. Spacing the reviews out is what makes them stick.'),
        h('div.row-btns', h('button.btn.primary', { type: 'button', onclick: function () { startSession('all'); render(); } }, 'Practice anyway'),
          h('button.btn', { type: 'button', onclick: function () { go('strengths'); } }, 'See strengths'))));
    }
  }
  function renderDone(main) {
    var ids = Object.keys(session.first), known = ids.filter(function (k) { return session.first[k] === 'k'; }), missed = ids.filter(function (k) { return session.first[k] === 'n'; });
    var pct = ids.length ? Math.round(known.length / ids.length * 100) : 0;
    var card = h('div.card.done', h('div.bigring', { style: { '--v': String(pct) }, role: 'img', 'aria-label': known.length + ' of ' + ids.length + ' known on the first try' }, h('b', known.length + '/' + ids.length), h('span', 'first try')),
      h('h2', !ids.length ? 'Round skipped' : !missed.length ? 'Clean round' : pct >= 60 ? 'Good round' : 'Round done'),
      h('p', !ids.length ? 'You skipped every card. That is fine: come back when you want to mark them.'
        : !missed.length ? 'You knew every card on the first try. They are scheduled to come back later, further apart each time.'
        : 'You marked ' + missed.length + (missed.length === 1 ? ' card' : ' cards') + ' "Not yet". Watch the matching animation, then try again.'));
    if (missed.length) {
      card.appendChild(h('ul.missed', missed.map(function (id) {
        var q = QBY[id], l = LESSON[q.anims[0]];
        return h('li', h('span', q.q), l ? h('a.btn.small', { href: '../#' + l.name }, icon('play', 'sm'), 'Watch: ' + l.title) : null);
      })));
    }
    card.appendChild(h('div.row-btns', { style: { 'justify-content': 'center' } },
      missed.length ? h('button.btn.primary', { type: 'button', onclick: function () { view = 'missed'; session = null; render(); } }, icon('redo'), 'Review what I missed') : null,
      h('button.btn', { type: 'button', onclick: function () { view = 'practice'; session = null; render(); } }, 'Next round'),
      h('button.btn', { type: 'button', onclick: function () { go('strengths'); } }, icon('star'), 'See strengths')));
    main.appendChild(card);
  }

  // ---------------------------------------------------------------- browse
  function renderBrowse() {
    var main = ui.clear($('#view'));
    main.appendChild(h('div.browse-bar', ui.seg({ label: 'Show', options: [['all', 'All'], ['new', 'New'], ['review', 'Review'], ['known', 'Knew it']], value: browse, onchange: function (v) { browse = v; renderBrowse(); } })));
    var any = false;
    D.phases.forEach(function (p) {
      var items = pool().filter(function (q) { return q.phase === p.dir && (browse === 'all' || status(q.id) === browse); });
      if (!items.length) return;
      any = true;
      main.appendChild(h('h2.phase-h', { dataset: { phase: p.id } }, h('span.phase-dot'), 'Phase ' + p.n + ': ' + p.name));
      main.appendChild(h('div.qlist', items.map(function (q) {
        var s = STATUS[status(q.id)];
        return h('details.qrow', h('summary', h('span.st' + (s[2] ? '.' + s[2] : ''), { title: s[0] }, icon(s[1], 'sm'), h('span.sr-only', s[0] + ': ')), h('span.t', q.q), icon('chevron', 'sm')),
          h('div.body', h('div.a', { html: q.a }), watchLinks(q)));
      })));
    });
    if (!any) main.appendChild(h('div.card.empty', h('h2', 'Nothing here'), h('p', 'No questions match this filter.')));
  }

  // ---------------------------------------------------------------- strengths
  function counts(items) {
    var c = { known: 0, review: 0, 'new': 0, due: 0, total: items.length };
    items.forEach(function (q) { c[status(q.id)] += 1; if (isDue(q.id)) c.due += 1; });
    return c;
  }
  function stack(c) {
    var seg = function (cls, n, label) { return n ? h('i.' + cls, { style: { flex: String(n) }, title: n + ' ' + label }) : null; };
    return h('div.stack', { role: 'img', 'aria-label': c.known + ' known, ' + c.review + ' to review, ' + c['new'] + ' not tried' }, seg('k', c.known, 'known'), seg('r', c.review, 'to review'), seg('n', c['new'], 'not tried'));
  }
  function renderStrengths() {
    var main = ui.clear($('#view')), all = counts(Q);
    main.appendChild(h('div.tiles',
      h('div.tile', h('span.k', icon('check', 'sm'), 'Knew it'), h('b', String(all.known)), h('span.n', 'of ' + all.total + ' questions')),
      h('div.tile', h('span.k', icon('redo', 'sm'), 'To review'), h('b', String(all.review)), h('span.n', 'marked "Not yet"')),
      h('div.tile', h('span.k', icon('spark', 'sm'), 'Not tried'), h('b', String(all['new'])), h('span.n', 'still new')),
      h('div.tile', h('span.k', icon('clock', 'sm'), 'Due today'), h('b', String(all.due)), h('span.n', 'ready to practice'))));

    main.appendChild(h('h2.sec', 'By phase'));
    main.appendChild(h('p.legend', h('span', h('i.sw.k'), icon('check', 'sm'), 'knew it'), h('span', h('i.sw.r'), icon('redo', 'sm'), 'to review'), h('span', h('i.sw.n'), 'not tried')));
    main.appendChild(h('div.card.phase-rows', D.phases.map(function (p) {
      var c = counts(Q.filter(function (q) { return q.phase === p.dir; }));
      return h('div.prow', { dataset: { phase: p.id } }, h('div.name', h('span.phase-dot'), h('span', 'Phase ' + p.n + ': ' + p.name)), stack(c),
        h('span.nums', c.known + ' / ' + c.total), h('button.btn.small', { type: 'button', 'aria-label': 'Practice phase ' + p.n, onclick: function () { phase = p.dir; lesson = null; go('practice'); } }, 'Practice'));
    })));

    // weakest lessons: every question counts for its best-matching animation
    var by = {};
    Q.forEach(function (q) { var n = q.anims[0]; if (!by[n]) by[n] = []; by[n].push(q); });
    var rows = Object.keys(by).map(function (n) { return { name: n, c: counts(by[n]) }; }).filter(function (r) { return r.c.known + r.c.review > 0 && LESSON[r.name]; });
    rows.sort(function (a, b) { return (b.c.review - a.c.review) || (a.c.known / a.c.total - b.c.known / b.c.total) || (a.name < b.name ? -1 : 1); });
    main.appendChild(h('h2.sec', 'Weakest topics first'));
    if (!rows.length) {
      main.appendChild(h('div.card.empty', h('h2', 'No answers yet'), h('p', 'Mark a few questions and the topics you find hardest will be listed here, each with its animation.'),
        h('div.row-btns', h('button.btn.primary', { type: 'button', onclick: function () { go('practice'); } }, 'Start practising'))));
    } else {
      main.appendChild(h('div.card.topic-rows-q', rows.slice(0, 12).map(function (r) {
        var l = LESSON[r.name], p = D.phases.filter(function (x) { return x.id === l.phase; })[0];
        return h('div.trow', { dataset: { phase: p.id } }, h('div.name', h('span.phase-dot'), h('span', h('b', l.title), h('br'),
          h('span.muted', r.c.review ? r.c.review + ' to review, ' + r.c.known + ' of ' + r.c.total + ' known' : 'all ' + r.c.known + ' of ' + r.c.total + ' known' + (r.c['new'] ? ', ' + r.c['new'] + ' not tried' : '')))),
          r.c.review ? h('span.chip.warn', icon('redo', 'sm'), 'weak spot') : h('span.chip.good', icon('check', 'sm'), 'solid'),
          h('a.btn.small', { href: '../#' + l.name }, icon('play', 'sm'), 'Watch'),
          h('button.btn.small', { type: 'button', onclick: function () { lesson = l.name; phase = 'all'; go('practice'); startSession('all'); render(); } }, 'Practice'));
      })));
    }

    var boxes = [0, 0, 0, 0, 0];
    Object.keys(st.cards).forEach(function (id) { if (QBY[id]) boxes[st.cards[id].box - 1] += 1; });
    main.appendChild(h('h2.sec', 'Spaced repetition boxes'));
    main.appendChild(h('p.muted', { style: { 'font-size': 'var(--fs-sm)' } }, '"I knew it" moves a card one box to the right, so it comes back later. "Not yet" sends it back to box 1.'));
    main.appendChild(h('div.boxes', boxes.map(function (n, i) { return h('div.box', h('b', String(n)), h('span', 'Box ' + (i + 1)), h('small', i === 0 ? 'again today' : 'every ' + DAYS[i + 1] + ' days')); })));

    var resetWrap = h('div.reset');
    var ask = function () {
      ui.clear(resetWrap);
      resetWrap.appendChild(h('span', 'Delete all quiz progress in this browser?'));
      resetWrap.appendChild(h('button.btn.small.warn', { type: 'button', onclick: function () { st = { cards: {} }; save(); session = null; render(); toast('Quiz progress cleared.'); } }, 'Yes, delete'));
      resetWrap.appendChild(h('button.btn.small', { type: 'button', onclick: idle }, 'Cancel'));
    };
    var idle = function () { ui.clear(resetWrap); resetWrap.appendChild(h('button.btn.small.ghost', { type: 'button', onclick: ask }, 'Reset my progress')); };
    idle(); main.appendChild(resetWrap);
  }

  // ---------------------------------------------------------------- render + wiring
  function render() {
    tabs(); filters();
    if (view === 'browse') renderBrowse(); else if (view === 'strengths') renderStrengths(); else renderCard();
  }
  function readHash() {
    var m, hash = decodeURIComponent(root.location.hash.slice(1));
    if ((m = /(?:^|&)phase=([^&]+)/.exec(hash)) && PHASE[m[1]]) phase = m[1];
    if ((m = /(?:^|&)lesson=([^&]+)/.exec(hash)) && LESSON[m[1]]) { lesson = m[1]; phase = 'all'; }
    if ((m = /(?:^|&)view=([^&]+)/.exec(hash)) && ['practice', 'missed', 'browse', 'strengths'].indexOf(m[1]) !== -1) view = m[1];
  }
  function init() {
    ui.initTheme();
    readHash();
    if (lesson) startSession('all');
    render();
    root.addEventListener('hashchange', function () { readHash(); session = null; if (lesson) startSession('all'); render(); });
    doc.addEventListener('keydown', function (e) {
      var t = e.target, tag = t && t.tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
      if (view !== 'practice' && view !== 'missed') return;
      if (!session || session.pos >= session.queue.length) return;
      var onControl = tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY';
      if (e.key === ' ' && !onControl) { e.preventDefault(); if (!session.shown) renderCard.reveal(); }
      else if (e.key === '1' && session.shown) renderCard.mark(true);
      else if (e.key === '2' && session.shown) renderCard.mark(false);
      else if (e.key === 'ArrowRight') { var s = $('#skip'); if (s) s.click(); }
      else if (e.key === 'ArrowLeft') { var b = $('#prev'); if (b && !b.disabled) b.click(); }
    });
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})(typeof self !== 'undefined' ? self : globalThis);
