/* Lessons viewer: home, lesson pages for every animation, lab pages. Vanilla JS.
   Data comes from assets/data.js (built by scripts/build_site.py from the repo's own files). */
(function (root) {
  'use strict';
  var AIER = root.AIER, D = root.AIER_DATA, ui = AIER.ui, h = ui.h, icon = ui.icon, doc = root.document;
  var $ = function (s) { return doc.querySelector(s); };
  var LESSONS = D.lessons, BY = {}, PHASE = {};
  LESSONS.forEach(function (l, i) { BY[l.name] = l; l.index = i; });
  D.phases.forEach(function (p) { PHASE[p.id] = p; });
  var LABS = (D.labs && D.labs.meta) || {};
  AIER.labs = AIER.labs || {};
  var LEVELS = { Beginner: 1, Intermediate: 2, Advanced: 3 };

  // ---------------------------------------------------------------- progress (this browser only)
  var PKEY = 'aier-progress-v1';
  var progress = ui.store.get(PKEY, {});
  if (!progress || typeof progress !== 'object') progress = {};
  progress.seen = progress.seen || {}; progress.aced = progress.aced || {};
  function saveProgress() { ui.store.set(PKEY, progress); }
  function seenIn(phaseId) { return LESSONS.filter(function (l) { return l.phase === phaseId && progress.seen[l.name]; }).length; }

  // ---------------------------------------------------------------- small pieces
  function labsFor(lesson) {
    return Object.keys(LABS).filter(function (id) { return AIER.labs[id] && LABS[id].lessons.indexOf(lesson.name) !== -1; });
  }
  function availableLabs() { return Object.keys(LABS).filter(function (id) { return AIER.labs[id]; }); }
  function minutes(lesson) { var m = 0; lesson.topics.forEach(function (t) { m += t.min; }); return m; }
  function level(lesson) { return lesson.topics.length ? lesson.topics[0].level : ''; }
  function levelEl(name) {
    var n = LEVELS[name] || 0;
    return h('span.lvl', { title: 'Level: ' + name }, h('i', { 'aria-hidden': 'true' }, [1, 2, 3].map(function (k) { return h('b' + (k <= n ? '.on' : '')); })), name);
  }
  function phaseChip(p) { return h('span.chip', { dataset: { phase: p.id } }, h('span.phase-dot'), 'Phase ' + p.n + ' · ' + p.name); }
  function stepTitle(s) { return s[2].replace(/^\d+\.\s*/, ''); }
  function toast(msg) {
    var t = h('div.toast', { role: 'status' }, msg);
    doc.body.appendChild(t);
    root.setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 2600);
  }

  // ---------------------------------------------------------------- sidebar
  var open = {}, query = '';
  function matches(l, q) {
    if (!q) return true;
    var hay = (l.title + ' ' + l.sub + ' ' + l.topics.map(function (t) { return t.title + ' ' + t.plain; }).join(' ')).toLowerCase();
    return q.split(/\s+/).every(function (w) { return hay.indexOf(w) !== -1; });
  }
  function renderSidebar(current) {
    var list = $('#nav'), q = query.trim().toLowerCase(), total = 0;
    ui.clear(list);
    D.phases.forEach(function (p) {
      var items = LESSONS.filter(function (l) { return l.phase === p.id && matches(l, q); });
      if (!items.length) return;
      total += items.length;
      var expanded = q ? true : (open[p.id] !== undefined ? open[p.id] : (current ? current.phase === p.id : p.id === '01'));
      var listId = 'nav-' + p.id;
      var btn = h('button', { type: 'button', 'aria-expanded': String(expanded), 'aria-controls': listId,
        onclick: function () { open[p.id] = !expanded; renderSidebar(current); } },
        h('span.phase-dot'), h('span.t', 'Phase ' + p.n + ': ' + p.name), h('span.n', seenIn(p.id) + '/' + LESSONS.filter(function (l) { return l.phase === p.id; }).length), icon('chevron', 'sm'));
      var ul = h('ul.nav-list', { id: listId, hidden: !expanded });
      items.forEach(function (l) {
        ul.appendChild(h('li', h('a', { href: '#' + l.name, 'aria-current': current && current.name === l.name ? 'page' : null },
          h('span.t', l.title),
          labsFor(l).length ? h('span.has-lab', { title: 'Has a hands-on lab' }, icon('flask', 'sm'), h('span.sr-only', ' (has a lab)')) : null,
          progress.seen[l.name] ? h('span.tick', { title: 'Viewed' }, icon('check', 'sm'), h('span.sr-only', ' (viewed)')) : h('span.m', minutes(l) + ' min'))));
      });
      list.appendChild(h('div.nav-phase', { dataset: { phase: p.id } }, btn, ul));
    });
    if (!total) list.appendChild(h('p.nav-empty', 'No lesson matches "' + query + '". Try a shorter word, for example "chunk" or "agent".'));
    var cur = list.querySelector('a[aria-current="page"]');
    if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest' });
  }
  function setQuery(v, from) {
    query = v;
    ['#q', '#q2'].forEach(function (s) { var el = $(s); if (el && el !== from) el.value = v; });
    renderSidebar(cur);
  }
  function closeNav() { doc.body.classList.remove('nav-open'); var s = $('.scrim'); if (s) s.hidden = true; var b = $('#menu'); if (b) b.setAttribute('aria-expanded', 'false'); }
  function openNav() { doc.body.classList.add('nav-open'); $('.scrim').hidden = false; $('#menu').setAttribute('aria-expanded', 'true'); var q2 = $('#q2'); if (q2) q2.focus(); }

  // ---------------------------------------------------------------- lesson state
  var cur = null, step = 0, timer = null, view = null, cleanup = [];
  function teardown() { pause(); cleanup.forEach(function (fn) { try { fn(); } catch (e) { /* ignore */ } }); cleanup = []; view = null; }
  function pause() {
    if (timer) { root.clearInterval(timer); timer = null; }
    doc.querySelectorAll('[data-act="play"]').forEach(function (b) { ui.clear(b); b.appendChild(icon('play')); b.appendChild(h('span', 'Play')); b.setAttribute('aria-label', 'Play the steps'); });
  }
  function play() {
    if (timer) { pause(); return; }
    if (!cur) return;
    if (step >= cur.steps.length - 1) show(0);
    doc.querySelectorAll('[data-act="play"]').forEach(function (b) { ui.clear(b); b.appendChild(icon('pause')); b.appendChild(h('span', 'Pause')); b.setAttribute('aria-label', 'Pause'); });
    var speed = parseFloat(($('#speed') || { value: '1' }).value);
    timer = root.setInterval(function () { if (step >= cur.steps.length - 1) { pause(); return; } show(step + 1); }, 2600 * speed);
  }
  function show(i) {
    if (!cur || !view) return;
    step = Math.max(0, Math.min(cur.steps.length - 1, i));
    view.setStep(step);
    if (step === cur.steps.length - 1 && !progress.seen[cur.name]) {
      progress.seen[cur.name] = 1; saveProgress(); renderSidebar(cur);
    }
    var fin = doc.getElementById('finish');
    if (fin) fin.hidden = step !== cur.steps.length - 1;
  }

  // ---------------------------------------------------------------- diagrams
  var LIGHT = { '#0d1117': '#ffffff', '#161b22': '#f1f3f1', '#30363d': '#c4cac3', '#e6edf3': '#14181f', '#9da7b3': '#5b6470',
    '#2dd4bf': '#0f766e', '#6ea8fe': '#2a78d6', '#f5b042': '#b86e00', '#f472b6': '#c2377a' };
  function recolor(svgText) {
    if (ui.isDark()) return svgText;
    return svgText.replace(/#[0-9a-fA-F]{6}\b/g, function (c) { return LIGHT[c.toLowerCase()] || c; });
  }
  function flowDiagram(lesson, onPick) {
    var el = h('div.flowviz' + (lesson.kind === 'loop' ? '.loop' : ''), { role: 'group', 'aria-label': lesson.title + ': the steps, in order' });
    lesson.steps.forEach(function (s, i) {
      if (i) el.appendChild(h('span.arrow', { 'aria-hidden': 'true' }, icon('next')));
      el.appendChild(h('button.node', { type: 'button', onclick: function () { onPick(i); }, 'aria-label': 'Step ' + (i + 1) + ': ' + s[0] + (s[1] ? ', ' + s[1] : '') },
        h('span.badge', { 'aria-hidden': 'true' }, String(i + 1)), h('span.lbl', s[0]), s[1] ? h('span.sub', s[1]) : null,
        h('span.cap', h('b', stepTitle(s)), h('span', s[3]))));
    });
    if (lesson.kind === 'loop') el.appendChild(h('div.loopback', h('span', '↻ then it repeats from step 1')));
    return {
      el: el,
      set: function (n) {
        var nodes = el.querySelectorAll('.node'), arrows = el.querySelectorAll('.arrow');
        nodes.forEach(function (node, i) {
          node.classList.toggle('done', i < n);
          if (i === n) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current');
        });
        arrows.forEach(function (a, i) { a.classList.toggle('lit', i < n); });
      }
    };
  }
  function barsDiagram(lesson) {
    var name = h('p.frame-name'), list = h('div.bars'), el = h('div.barsviz', name, list,
      h('p.legend', h('span', h('i.sw'), 'kept'), h('span', h('i.sw.dim'), 'removed before sampling')));
    return {
      el: el,
      set: function (n) {
        var f = lesson.frames[n];
        name.textContent = f.name;
        ui.bars(list, lesson.labels.map(function (lab, j) {
          return { label: lab, value: f.probs[j], ghost: null, dim: !!(f.keep && !f.keep[j]) };
        }), { d: 2 });
      }
    };
  }
  function svgDiagram(lesson, stage) {
    var svg = null, raw = null, wanted = 0, dead = false;
    function mount() {
      if (dead || raw === null) return;
      stage.innerHTML = recolor(raw);
      svg = stage.querySelector('svg');
      if (!svg) return;
      svg.removeAttribute('width'); svg.removeAttribute('height');
      svg.setAttribute('aria-label', lesson.title + '. The text beside the picture describes each step.');
      set(wanted);
    }
    function set(n) {
      wanted = n;
      if (!svg || !svg.pauseAnimations) return;
      try { svg.pauseAnimations(); svg.setCurrentTime(lesson.marks[n]); } catch (e) { /* SMIL not supported: the still picture stays */ }
    }
    stage.appendChild(h('div.loading', 'Loading the picture…'));
    root.fetch('anim/' + lesson.name + '.svg').then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
      .then(function (t) { raw = t; mount(); })
      .catch(function () {
        if (dead) return;
        ui.clear(stage).appendChild(h('img', { alt: lesson.title + ': ' + lesson.sub, src: 'anim/' + lesson.name + '.svg', style: { width: '100%' } }));
      });
    var onTheme = function () { mount(); };
    doc.addEventListener('aier:theme', onTheme);
    cleanup.push(function () { dead = true; doc.removeEventListener('aier:theme', onTheme); });
    return { el: null, set: set };
  }

  // ---------------------------------------------------------------- "check yourself": put the steps in order
  function shuffled(n) {
    var idx = [], i, j, t;
    for (i = 0; i < n; i++) idx.push(i);
    for (var tries = 0; tries < 8; tries++) {
      for (i = n - 1; i > 0; i--) { j = Math.floor(Math.random() * (i + 1)); t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      if (idx.some(function (v, k) { return v !== k; })) break;
    }
    return idx;
  }
  function checkCard(lesson) {
    var labels = lesson.steps.map(function (s) { return s[0]; });
    var card = h('section.card.check', { 'aria-labelledby': 'check-h' });
    function start() {
      var placed = 0, misses = 0, order = shuffled(labels.length);
      var slots = h('ol.slots', { 'aria-label': 'Your order so far', style: { 'list-style': 'none', padding: '0' } }, labels.map(function () { return h('li.slot', '…'); }));
      var pool = h('div.pool', { role: 'group', 'aria-label': 'Steps to place' });
      var verdict = h('p.verdict', { role: 'status' });
      function mark() { slots.querySelectorAll('.slot').forEach(function (s, i) { s.classList.toggle('next', i === placed); }); }
      order.forEach(function (k) {
        var b = h('button', { type: 'button' }, labels[k]);
        b.addEventListener('click', function () {
          if (labels[k] === labels[placed]) {
            var slot = slots.children[placed];
            slot.textContent = labels[k]; slot.classList.add('filled');
            b.classList.add('used'); b.disabled = true; placed += 1; mark();
            if (placed === labels.length) {
              verdict.className = 'verdict ok'; ui.clear(verdict);
              verdict.appendChild(icon('check')); verdict.appendChild(h('span', misses ? 'Done, with ' + misses + (misses === 1 ? ' slip.' : ' slips.') + ' Try once more for a clean run.' : 'Clean run. You know the order.'));
              verdict.appendChild(h('button.btn.small', { type: 'button', onclick: start }, icon('redo', 'sm'), 'Shuffle again'));
              if (!misses) { progress.aced[lesson.name] = 1; saveProgress(); }
            } else {
              verdict.textContent = 'Right. What comes next?';
              var nextBtn = pool.querySelector('button:not(.used)'); if (nextBtn) nextBtn.focus();
            }
          } else {
            misses += 1; b.classList.remove('wrong'); void b.offsetWidth; b.classList.add('wrong');
            verdict.className = 'verdict'; verdict.textContent = 'Not yet: "' + labels[k] + '" comes later. Which step is number ' + (placed + 1) + '?';
          }
        });
        pool.appendChild(b);
      });
      ui.clear(card);
      card.appendChild(h('h2', { id: 'check-h' }, icon('shuffle'), 'Check yourself: put the steps in order'));
      card.appendChild(h('p.how', 'Tap the step that comes first, then the next one. No peeking at the picture.'));
      card.appendChild(slots); card.appendChild(pool); card.appendChild(verdict);
      mark();
    }
    start();
    return card;
  }

  // ---------------------------------------------------------------- lab sections
  function labSection(id, standalone) {
    var meta = LABS[id];
    var body = h('div');
    var section = h('section.card.lab', { id: 'lab-' + id, 'aria-labelledby': 'lab-h-' + id },
      h('div.lab-head', h('div', h(standalone ? 'h1' : 'h2', { id: 'lab-h-' + id, style: standalone ? { 'font-size': 'var(--fs-2xl)' } : null }, icon('flask'), meta.title), h('p', meta.blurb)),
        h('div.lab-actions', AIER.labs[id].tour ? h('button.btn.small.primary', { type: 'button', onclick: function () {
          var steps = AIER.labs[id].tour(body);
          if (steps && steps.length) cleanup.push(ui.tour(steps, { label: meta.title + ': guided tour' }));
        } }, icon('spark', 'sm'), 'Guided tour') : null,
          standalone ? null : h('a.btn.small.ghost', { href: '#lab/' + id }, 'Open on its own page'))),
      body);
    try {
      var stop = AIER.labs[id].mount(body, { data: D.labs.data, repo: D.repo, lesson: cur });
      if (typeof stop === 'function') cleanup.push(stop);
    } catch (e) {
      body.appendChild(h('p.note-inline', 'This lab could not start: ' + e.message));
      if (root.console) root.console.error(e);
    }
    return section;
  }

  // ---------------------------------------------------------------- lesson page
  function renderLesson(lesson) {
    teardown(); cur = lesson; step = 0;
    var main = ui.clear($('#main')), p = PHASE[lesson.phase], n = lesson.steps.length;
    doc.title = lesson.title + ' | AI Engineer Roadmap';
    progress.last = lesson.name; saveProgress();

    var head = h('header.lesson-head',
      h('nav.crumbs', { 'aria-label': 'Breadcrumb' }, h('a', { href: '#' }, 'All lessons'), h('span', { 'aria-hidden': 'true' }, '/'), h('span', 'Phase ' + p.n + ': ' + p.name)),
      h('h1', lesson.title), h('p.sub', lesson.sub),
      h('div.chips', phaseChip(p), level(lesson) ? h('span.chip', levelEl(level(lesson))) : null,
        h('span.chip', icon('clock', 'sm'), minutes(lesson) + ' min read'), h('span.chip', n + ' steps'),
        labsFor(lesson).map(function (id) { return h('a.chip.accent', { href: '#lab-' + id, onclick: function (e) { e.preventDefault(); var t = doc.getElementById('lab-' + id); if (t) t.scrollIntoView({ behavior: ui.reducedMotion() ? 'auto' : 'smooth' }); } }, icon('flask', 'sm'), LABS[id].title); })));
    if (lesson.topics.length && lesson.topics[0].one) head.appendChild(h('p.oneliner', h('span', h('b', 'In one line: '), h('span', { html: lesson.topics[0].one }))));
    main.appendChild(head);

    // stage
    var stage = h('div.stage'), diagram;
    if (lesson.kind === 'pipeline' || lesson.kind === 'loop') { diagram = flowDiagram(lesson, function (i) { pause(); show(i); }); stage.classList.add('native'); stage.appendChild(diagram.el); }
    else if (lesson.kind === 'bars') { diagram = barsDiagram(lesson); stage.classList.add('native'); stage.appendChild(diagram.el); }
    else diagram = svgDiagram(lesson, stage);

    var range = h('input', { type: 'range', min: 1, max: n, step: 1, value: 1, id: 'stepper', 'aria-label': 'Step' });
    var count = h('span.count', { 'aria-hidden': 'true' });
    var prevB = h('button.btn.dockable', { type: 'button', 'data-act': 'prev', 'aria-label': 'Previous step' }, icon('prev'), h('span', 'Back'));
    var playB = h('button.btn.primary.play.dockable', { type: 'button', 'data-act': 'play' });
    var nextB = h('button.btn.dockable', { type: 'button', 'data-act': 'next', 'aria-label': 'Next step' }, h('span', 'Next'), icon('next'));
    var transport = h('div.transport', prevB, playB, nextB, h('div.progress', range, count),
      h('label.speed', { 'for': 'speed' }, 'Speed', h('select', { id: 'speed', onchange: function () { if (timer) { pause(); play(); } } },
        h('option', { value: '1.6' }, 'Slow'), h('option', { value: '1', selected: true }, 'Normal'), h('option', { value: '0.6' }, 'Fast'))));
    var stageCard = h('div.card.stage-card', stage, transport);

    // step panel
    var k = h('p.k'), title = h('h2'), c2 = h('p.c2');
    var now = h('div.now', { 'aria-live': 'polite', 'aria-atomic': 'true' }, k, title, c2);
    var stepList = h('ol.steps', { 'aria-label': 'All steps' });
    lesson.steps.forEach(function (s, i) {
      stepList.appendChild(h('li', h('button', { type: 'button', onclick: function () { pause(); show(i); } }, h('span.n', { 'aria-hidden': 'true' }, String(i + 1)), h('span', s[0] + (lesson.kind === 'custom' || lesson.kind === 'bars' ? '' : s[1] ? ' · ' + s[1] : '')))));
    });
    var panel = h('aside.card.step-panel' + (lesson.kind === 'pipeline' || lesson.kind === 'loop' ? '.for-flow' : ''), { 'aria-label': 'What is happening in this step' }, now, stepList);

    // progressive disclosure: the demo's real output
    lesson.demos.forEach(function (path, i) {
      var out = D.demos[path], file = path.split('/').pop();
      var body = [h('p', 'This is what the repo’s own demo prints. Run it yourself: ', h('code', 'python ' + path))];
      if (out) body.push(h('pre.out', { tabindex: '0', 'aria-label': 'Output of ' + file }, out));
      else body.push(h('p.src', 'Its output changes on every run (timings or ids), so it is not frozen here.'));
      body.push(h('p.src', h('a', { href: D.repo + '/blob/main/' + path, rel: 'noopener' }, 'Read ' + file + ' on GitHub')));
      panel.appendChild(ui.disclosure(i === 0 ? 'Show the math: real output of ' + file : 'More: output of ' + file, body));
    });
    var links = h('div.panel-links');
    lesson.topics.forEach(function (t) {
      links.appendChild(h('a', { href: D.repo + '/blob/main/' + t.path, rel: 'noopener' }, icon('book', 'sm'), 'Read the full lesson: ' + t.title, icon('ext', 'sm')));
    });
    if (D.quizFor && D.quizFor[lesson.name]) links.appendChild(h('a', { href: 'quiz/#lesson=' + lesson.name }, icon('quiz', 'sm'), 'Quiz yourself on this (' + D.quizFor[lesson.name] + (D.quizFor[lesson.name] === 1 ? ' question)' : ' questions)')));
    panel.appendChild(links);

    main.appendChild(h('div.lesson-wrap', h('div.lesson-grid', stageCard, panel)));
    var nextL0 = LESSONS[lesson.index + 1], phaseItems = LESSONS.filter(function (x) { return x.phase === lesson.phase; });
    main.appendChild(h('div.card.finish', { id: 'finish', hidden: true, role: 'status' },
      h('span.finish-ic', { 'aria-hidden': 'true' }, icon('check')),
      h('div', h('p.t', 'Lesson complete'), h('p.muted', 'You have seen ' + (seenIn(lesson.phase) + (progress.seen[lesson.name] ? 0 : 1)) + ' of ' + phaseItems.length + ' lessons in Phase ' + p.n + '. Check yourself below, or keep going.')),
      h('div.row-btns', { style: { 'margin-top': '0' } },
        h('a.btn.small', { href: '#check-h', onclick: function (e) { e.preventDefault(); var c = doc.getElementById('check-h'); if (c) c.scrollIntoView({ behavior: ui.reducedMotion() ? 'auto' : 'smooth', block: 'center' }); } }, icon('shuffle', 'sm'), 'Check yourself'),
        nextL0 ? h('a.btn.small.primary', { href: '#' + nextL0.name }, 'Next: ' + nextL0.title, icon('next', 'sm')) : null)));
    main.appendChild(h('p.kbd-hint', 'Keyboard: ', h('kbd', '←'), ' ', h('kbd', '→'), ' move one step, ', h('kbd', 'Space'), ' plays or pauses, ', h('kbd', '/'), ' searches.'));
    main.appendChild(checkCard(lesson));
    labsFor(lesson).forEach(function (id) { main.appendChild(labSection(id, false)); });

    // previous / next lesson
    var prevL = LESSONS[lesson.index - 1], nextL = LESSONS[lesson.index + 1];
    main.appendChild(h('nav.lesson-nav', { 'aria-label': 'Previous and next lesson' },
      prevL ? h('a.card', { href: '#' + prevL.name }, h('span.k', '← Previous lesson'), h('span.t', prevL.title)) : h('span'),
      nextL ? h('a.card.next', { href: '#' + nextL.name }, h('span.k', 'Next lesson →'), h('span.t', nextL.title)) : h('span')));
    main.appendChild(footer());

    // mobile dock
    var dock = $('#dock'); ui.clear(dock); dock.hidden = false;
    dock.appendChild(h('button.btn', { type: 'button', 'data-act': 'prev', 'aria-label': 'Previous step' }, icon('prev'), h('span', 'Back')));
    dock.appendChild(h('button.btn.primary', { type: 'button', 'data-act': 'play' }));
    dock.appendChild(h('button.btn', { type: 'button', 'data-act': 'next', 'aria-label': 'Next step' }, h('span', 'Next'), icon('next')));

    view = {
      setStep: function (i) {
        var s = lesson.steps[i];
        diagram.set(i);
        k.textContent = 'Step ' + (i + 1) + ' of ' + n; title.textContent = stepTitle(s); c2.textContent = s[3];
        stepList.querySelectorAll('button').forEach(function (b, j) {
          b.classList.toggle('done', j < i);
          if (j === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
        });
        range.value = i + 1; range.style.setProperty('--fill', (n > 1 ? i / (n - 1) * 100 : 100) + '%');
        range.setAttribute('aria-valuetext', 'Step ' + (i + 1) + ' of ' + n + ': ' + s[0]);
        count.textContent = (i + 1) + ' / ' + n;
        doc.querySelectorAll('[data-act="prev"]').forEach(function (b) { b.disabled = i === 0; });
        doc.querySelectorAll('[data-act="next"]').forEach(function (b) { b.disabled = i === n - 1; });
      }
    };
    range.addEventListener('input', function () { pause(); show(Number(range.value) - 1); });
    pause(); show(0); renderSidebar(lesson);
    root.scrollTo(0, 0);
  }

  function footer() {
    return h('footer.site', h('span', 'Free and open source. No tracking, no sign-up. Your progress stays in this browser.'),
      h('span', h('a', { href: D.repo, rel: 'noopener' }, 'GitHub repo'), ' · ', h('a', { href: D.repo + '/issues', rel: 'noopener' }, 'Report a mistake')));
  }

  // ---------------------------------------------------------------- lab page
  function renderLab(id) {
    teardown(); cur = null;
    var main = ui.clear($('#main')), meta = LABS[id];
    $('#dock').hidden = true;
    doc.title = meta.title + ' | AI Engineer Roadmap';
    main.appendChild(h('nav.crumbs', { 'aria-label': 'Breadcrumb' }, h('a', { href: '#' }, 'Home'), h('span', { 'aria-hidden': 'true' }, '/'), h('span', 'Labs')));
    main.appendChild(labSection(id, true));
    var uses = meta.lessons.filter(function (n) { return BY[n]; });
    if (uses.length) {
      main.appendChild(h('div.section-head', h('h2', 'Lessons this lab belongs to')));
      main.appendChild(h('ul.topic-rows', uses.map(function (n) { return topicRow(BY[n]); })));
    }
    main.appendChild(footer());
    renderSidebar(null); root.scrollTo(0, 0);
  }

  // ---------------------------------------------------------------- home
  function topicRow(l) {
    return h('li', h('a', { href: '#' + l.name }, h('span.t', l.title, labsFor(l).length ? h('span.chip.accent', { style: { 'margin-left': '8px' } }, icon('flask', 'sm'), 'lab') : null),
      levelEl(level(l)), h('span.m', progress.seen[l.name] ? 'viewed' : minutes(l) + ' min')));
  }
  function heroArt() {
    var S = AIER.sampling, sd = D.labs && D.labs.data && D.labs.data.sampling;
    if (!S || !sd) {
      return h('div.hero-art', h('p.mini-title', 'What is inside'), h('div.stats',
        h('div', h('b', String(D.counts.lessons)), h('span', 'animated lessons')), h('div', h('b', String(D.counts.questions)), h('span', 'quiz questions')),
        h('div', h('b', String(D.phases.length)), h('span', 'phases'))));
    }
    var list = h('div.bars'), cap = h('p.caption', { role: 'status' });
    var base = S.softmax(sd.LOGITS, 1.0);
    function paint(t) {
      var probs = S.softmax(sd.LOGITS, t), top = 0;
      probs.forEach(function (p, i) { if (p > probs[top]) top = i; });
      ui.bars(list, sd.TOKENS.map(function (tok, i) { return { label: tok, value: probs[i], ghost: base[i] }; }), { d: 2 });
      cap.textContent = t < 0.5 ? 'Low temperature: "' + sd.TOKENS[top] + '" wins almost every time.' : t > 1.5 ? 'High temperature: the odds flatten, so odd picks get more likely.' : 'Around 1.0 the model uses its own odds.';
    }
    var sl = ui.slider({ label: 'Temperature', min: 0.1, max: 3, step: 0.1, value: 1, format: function (v) { return AIER.py.fixed(v, 1); }, oninput: paint });
    paint(1);
    return h('div.hero-art.hero-mini', h('p.mini-title', h('span', 'Try it: next word after “The capital of France is”'), h('span.chip', 'toy numbers')),
      sl, list, cap, h('p', { style: { 'margin-top': '12px' } }, h('a.btn.small', { href: '#lab/sampling' }, icon('flask', 'sm'), 'Open the full Sampling lab')));
  }
  function renderHome() {
    teardown(); cur = null;
    var main = ui.clear($('#main'));
    $('#dock').hidden = true;
    doc.title = 'AI Engineer Roadmap: interactive lessons';
    var labs = availableLabs();
    var first = LESSONS[0];
    main.appendChild(h('section.hero',
      h('div', h('span.eyebrow', icon('spark', 'sm'), 'Free · open source · no sign-up'),
        h('h1', 'Learn AI engineering by ', h('em', 'playing with it')),
        h('p.lead', D.counts.lessons + ' short step-by-step lessons' + (labs.length ? ', ' + labs.length + ' hands-on labs' : '') + ' and ' + D.counts.questions + ' quiz questions on RAG, LLMs, agents, MCP and production AI.'),
        h('div.cta', h('a.btn.primary', { href: '#' + first.name }, icon('play'), 'Start with lesson 1'),
          labs.length ? h('a.btn', { href: '#lab/' + labs[0] }, icon('flask'), 'Open a lab') : null,
          h('a.btn', { href: 'quiz/' }, icon('quiz'), 'Take the quiz')),
        labs.length ? h('p.note', icon('info', 'sm'), h('span', 'The labs run small teaching models with hand-picked numbers from this repo’s Python demos. They are not real LLMs.')) : null),
      heroArt()));

    if (progress.last && BY[progress.last]) {
      var l = BY[progress.last];
      main.appendChild(h('a.card.continue', { href: '#' + l.name }, icon('play'), h('span', h('span.k', 'Continue where you left off'), h('br'), h('span.t', l.title)), h('span.grow', { style: { flex: '1' } }), phaseChip(PHASE[l.phase])));
    }

    if (labs.length) {
      main.appendChild(h('div.section-head', h('h2', 'Hands-on labs'), h('p', 'Change an input and watch the numbers move.')));
      main.appendChild(h('div.grid-labs', labs.map(function (id) {
        var m = LABS[id], p = PHASE[m.phase];
        return h('a.card.lab-card', { href: '#lab/' + id }, h('h3', m.title), h('p', m.blurb),
          h('div.foot', p ? phaseChip(p) : null, h('span.chip.accent', icon('flask', 'sm'), m.verb || 'Interactive')));
      })));
    }

    main.appendChild(h('div.section-head', h('h2', 'The roadmap'), h('p', 'Seven phases, from your first RAG pipeline to running AI in production.')));
    main.appendChild(h('div.grid-phases', D.phases.map(function (p) {
      var items = LESSONS.filter(function (x) { return x.phase === p.id; }), mins = 0, lv = { Beginner: 0, Intermediate: 0, Advanced: 0 };
      items.forEach(function (x) { mins += minutes(x); if (lv[level(x)] !== undefined) lv[level(x)] += 1; });
      var seen = seenIn(p.id), pct = items.length ? Math.round(seen / items.length * 100) : 0;
      return h('article.card.phase-card', { dataset: { phase: p.id } },
        h('div', { style: { display: 'flex', 'justify-content': 'space-between', gap: '12px', 'align-items': 'flex-start' } },
          h('div', h('p.ph', h('span.phase-dot'), 'Phase ' + p.n), h('h3', p.name)),
          h('span.ring', { style: { '--v': String(pct) }, role: 'img', 'aria-label': seen + ' of ' + items.length + ' lessons viewed' }, seen + '/' + items.length)),
        h('p.meta', h('span', items.length + ' lessons'), h('span', '≈ ' + (mins >= 90 ? AIER.py.fixed(mins / 60, 1) + ' h' : mins + ' min') + ' of reading')),
        h('div.levelbar', { role: 'img', 'aria-label': lv.Beginner + ' beginner, ' + lv.Intermediate + ' intermediate, ' + lv.Advanced + ' advanced lessons' },
          h('i.b', { style: { flex: String(lv.Beginner) } }), h('i.i', { style: { flex: String(lv.Intermediate) } }), h('i.a', { style: { flex: String(lv.Advanced) } })),
        h('p.muted', { style: { 'font-size': 'var(--fs-xs)' } }, lv.Beginner + ' beginner · ' + lv.Intermediate + ' intermediate · ' + lv.Advanced + ' advanced'),
        h('div', { style: { display: 'flex', gap: '8px', 'flex-wrap': 'wrap', 'margin-top': '12px' } },
          h('a.btn.small.primary', { href: '#' + (items.filter(function (x) { return !progress.seen[x.name]; })[0] || items[0]).name }, seen ? 'Continue' : 'Start', icon('next', 'sm')),
          h('a.btn.small', { href: 'quiz/#phase=' + p.dir }, icon('quiz', 'sm'), 'Quiz')),
        h('details', h('summary', icon('chevron', 'sm'), 'All ' + items.length + ' lessons'), h('ul.topic-rows', items.map(topicRow))));
    })));
    main.appendChild(footer());
    renderSidebar(null); root.scrollTo(0, 0);
  }

  // ---------------------------------------------------------------- router
  function route() {
    closeNav();
    var hash = decodeURIComponent(root.location.hash.slice(1));
    if (hash.indexOf('lab/') === 0 && LABS[hash.slice(4)] && AIER.labs[hash.slice(4)]) { renderLab(hash.slice(4)); return; }
    if (BY[hash]) { renderLesson(BY[hash]); return; }
    renderHome();
    if (hash && hash.indexOf('lab/') !== 0) toast('No lesson called "' + hash + '". Showing the home page.');
  }

  // ---------------------------------------------------------------- wiring
  function init() {
    ui.initTheme();
    doc.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('[data-act]') : null;
      if (!b) return;
      var act = b.getAttribute('data-act');
      if (act === 'play') play();
      else if (act === 'prev') { pause(); show(step - 1); }
      else if (act === 'next') { pause(); show(step + 1); }
    });
    ['#q', '#q2'].forEach(function (s) {
      var el = $(s); if (!el) return;
      el.addEventListener('input', function () { setQuery(el.value, el); });
      el.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { var a = $('#nav .nav-list a'); if (a) { root.location.hash = a.getAttribute('href'); el.blur(); } }
        if (e.key === 'Escape') { el.value = ''; setQuery('', null); el.blur(); }
      });
    });
    $('#menu').addEventListener('click', function () { if (doc.body.classList.contains('nav-open')) closeNav(); else openNav(); });
    $('.scrim').addEventListener('click', closeNav);
    doc.addEventListener('keydown', function (e) {
      var t = e.target, tag = t && t.tagName;
      var typing = tag === 'INPUT' && t.type !== 'range' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable);
      if (e.key === 'Escape') { closeNav(); return; }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '/') {
        e.preventDefault();
        var q = $('#q');
        if (q && q.offsetParent !== null) q.focus(); else openNav();
        return;
      }
      if (!cur || !view) return;
      if (tag === 'INPUT' && t.type === 'range' && t.id !== 'stepper') return;       // lab sliders keep their arrow keys
      if (t && t.closest && t.closest('.lab')) return;
      if (e.key === 'ArrowRight') { pause(); show(step + 1); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { pause(); show(step - 1); e.preventDefault(); }
      else if (e.key === ' ' && tag !== 'BUTTON' && tag !== 'A' && tag !== 'SUMMARY') { play(); e.preventDefault(); }
    });
    root.addEventListener('hashchange', route);
    route();
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})(typeof self !== 'undefined' ? self : globalThis);
