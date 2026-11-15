/* Request lifecycle lab: replay traffic through a rate limiter, a cache and a (simulated) model.
   Logic: scripts/site/core/sims.js (a tested port of 07-production-ai/code/prod_request_sim.py,
   which reuses TTLCache and TokenBucket from prod_cache_ratelimit.py). Time is virtual: nothing sleeps. */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, S = AIER.reqsim, py = AIER.py;
  AIER.labs = AIER.labs || {};
  var START = { '14-cost-optimization': 'popular', '04-redis': 'popular', '13-observability': 'steady', '01-fastapi-for-ai': 'burst', '05-celery': 'burst' };
  var COLORS = { hit: 'var(--good)', llm: 'var(--p2)', limited: 'var(--bad)' };
  var NAMES = { hit: 'Served from cache', llm: 'Sent to the model', limited: 'Rejected: rate limit (HTTP 429)' };

  AIER.labs.reqsim = {
    mount: function (el, ctx) {
      var st = { pattern: (ctx.lesson && START[ctx.lesson.name]) || 'steady', n: 60, ttl: 30, rate: 1, cap: 3, order: 'limit_first', llmMs: 1800, price: 0.01, seed: 7, at: 0, timer: null };
      var res = null, traffic = null;
      var pat = ui.seg({ label: 'Traffic pattern', options: Object.keys(S.PATTERNS).map(function (k) { return [k, S.PATTERNS[k].title]; }), value: st.pattern, onchange: function (v) { st.pattern = v; st.at = 0; stop(); compute(); paint(); } });
      var patNote = h('p.help');
      var ord = ui.seg({ label: 'Order of checks', options: [['limit_first', 'Rate limit, then cache'], ['cache_first', 'Cache, then rate limit']], value: st.order, onchange: function (v) { st.order = v; compute(); paint(); } });
      var ttlSl = ui.slider({ label: 'Cache lifetime (TTL, seconds)', min: 0, max: 120, step: 5, value: st.ttl, help: '0 turns the cache off. Longer means more hits but staler answers.', oninput: function (v) { st.ttl = v; compute(); paint(); } });
      var rateSl = ui.slider({ label: 'Allowed requests per second', min: 0.2, max: 5, step: 0.2, value: st.rate, format: function (v) { return py.fixed(v, 1); }, oninput: function (v) { st.rate = py.round(v, 1); compute(); paint(); } });
      var capSl = ui.slider({ label: 'Burst allowance (bucket size)', min: 1, max: 20, step: 1, value: st.cap, help: 'How many requests can arrive at once before limiting starts.', oninput: function (v) { st.cap = v; compute(); paint(); } });
      var llmSl = ui.slider({ label: 'Model latency (ms, illustrative)', min: 200, max: 6000, step: 100, value: st.llmMs, oninput: function (v) { st.llmMs = v; compute(); paint(); } });
      var priceSl = ui.slider({ label: 'Price per model call (illustrative, $)', min: 0.001, max: 0.1, step: 0.001, value: st.price, format: function (v) { return '$' + py.fixed(v, 3); }, oninput: function (v) { st.price = py.round(v, 3); compute(); paint(); } });
      var timeline = h('div.timeline-req', { role: 'img', 'aria-label': 'Each bar is one request, coloured by what happened to it.' });
      var legend = h('p.legend', Object.keys(NAMES).map(function (k) { return h('span', h('i.sw', { style: { background: COLORS[k] } }), NAMES[k]); }));
      var meters = h('div.meters'), now = h('div.now-card', { 'aria-live': 'polite' });
      var playBtn = h('button.btn.small.primary', { type: 'button' }), resetBtn = h('button.btn.small.ghost', { type: 'button' }, icon('redo', 'sm'), 'Replay from the start'),
        allBtn = h('button.btn.small', { type: 'button' }, 'Show all');

      function compute() {
        traffic = S.makeTraffic(st.pattern, st.n, st.seed);
        res = S.simulate(traffic, { ttl: st.ttl, rate: st.rate, capacity: st.cap, order: st.order, llmMs: st.llmMs, cacheMs: 15, price: st.price });
        st.at = Math.min(st.at, res.events.length);
      }
      function setPlay(on) { ui.clear(playBtn); playBtn.appendChild(icon(on ? 'pause' : 'play', 'sm')); playBtn.appendChild(h('span', on ? 'Pause' : 'Play')); }
      function stop() { if (st.timer) { root.clearInterval(st.timer); st.timer = null; } setPlay(false); }
      playBtn.addEventListener('click', function () {
        if (st.timer) { stop(); return; }
        if (st.at >= res.events.length) st.at = 0;
        setPlay(true);
        st.timer = root.setInterval(function () { st.at = Math.min(res.events.length, st.at + (ui.reducedMotion() ? 6 : 2)); paint(); if (st.at >= res.events.length) stop(); }, 120);
      });
      resetBtn.addEventListener('click', function () { stop(); st.at = 0; paint(); });
      allBtn.addEventListener('click', function () { stop(); st.at = res.events.length; paint(); });

      function meter(label, value, text, color, hint) {
        return h('div.meter', { title: hint || '' }, h('span.ml', label), h('span.mt', h('i', { style: { width: Math.max(0, Math.min(100, value * 100)) + '%', background: color } })), h('b.mv', text));
      }
      function paint() {
        patNote.textContent = S.PATTERNS[st.pattern].blurb;
        var evs = res.events.slice(0, st.at), total = res.events.length, hits = 0, llm = 0, lim = 0, ms = 0;
        evs.forEach(function (e) { if (e.kind === 'hit') hits++; else if (e.kind === 'llm') llm++; else lim++; ms += e.ms; });
        var served = hits + llm;
        ui.clear(timeline);
        res.events.forEach(function (e, i) {
          var h0 = e.kind === 'limited' ? 18 : Math.max(10, Math.min(100, 12 + e.ms / st.llmMs * 88));
          timeline.appendChild(h('i.bar' + (i >= st.at ? '.future' : ''), { style: { background: COLORS[e.kind], height: h0 + '%' }, title: 't=' + py.fixed(e.t, 1) + 's · ' + S.questionText(e.q) + ' · ' + NAMES[e.kind] }));
        });
        ui.clear(meters);
        meters.appendChild(meter('Cache hit rate', served ? hits / served : 0, py.fixed(served ? hits / served * 100 : 0, 0) + '%', 'var(--good)', 'Hits divided by requests that were served'));
        meters.appendChild(meter('Requests rejected', st.at ? lim / st.at : 0, lim + ' of ' + st.at, 'var(--bad)', 'Users who got an HTTP 429'));
        meters.appendChild(meter('Average latency', served ? Math.min(1, ms / served / st.llmMs) : 0, py.fixed(served ? ms / served : 0, 0) + ' ms', 'var(--p2)', 'Mean over served requests'));
        meters.appendChild(meter('Model spend', Math.min(1, llm * st.price / (total * st.price || 1)), '$' + py.fixed(llm * st.price, 2), 'var(--p3)', 'Calls to the model times the price you set'));
        meters.appendChild(meter('Money the cache saved', Math.min(1, hits / (total || 1)), '$' + py.fixed(hits * st.price, 2), 'var(--accent)', 'Each hit is one model call you did not pay for'));
        ui.clear(now);
        var last = evs[evs.length - 1];
        if (!last) now.appendChild(h('p', 'Press Play to replay ' + total + ' requests, or Show all.'));
        else now.appendChild(h('p', h('b', 'Request ' + evs.length + ': '), '“' + S.questionText(last.q) + '” at t = ' + py.fixed(last.t, 1) + ' s → ', h('b', { style: { color: COLORS[last.kind] } }, NAMES[last.kind]), '. ',
          last.kind === 'hit' ? 'Same question seen within the last ' + st.ttl + ' s, so no model call.' : last.kind === 'llm' ? 'Not in the cache, so the model answers and the answer is stored.' : 'The bucket had no token left, so the request is turned away before any work is done.'));
      }
      var insight = h('section.card.mini', h('h3', 'What to try'), h('ul.ticks',
        h('li', 'Choose “Bursts”, then switch between the two orders. Checking the cache first lets hits through even when the limiter is empty.'),
        h('li', 'Choose “Every question is different” and notice how little the cache helps.'),
        h('li', 'Set the cache lifetime to 0 and watch the spend jump.')));
      el.appendChild(ui.honest('The cache and the token-bucket rate limiter are the real classes from prod_cache_ratelimit.py, replayed on a virtual clock. Latency and price are numbers you choose: they are illustrations, not measurements of any provider. The model call itself is simulated.', '07-production-ai/code/prod_request_sim.py', ctx.repo));
      el.appendChild(h('div.lab-grid',
        h('div.controls', h('div.control', h('span.lbl', 'Traffic pattern'), pat, patNote), h('div.control', h('span.lbl', 'Order of checks'), ord), ttlSl, rateSl, capSl, llmSl, priceSl),
        h('div', h('div.req-head', h('h3', 'Every request, in time order'), h('div.row-btns', playBtn, resetBtn, allBtn)), timeline, legend, now, meters, insight)));
      compute(); st.at = res.events.length; setPlay(false); paint();
      return stop;
    }
  };
  AIER.labs.reqsim.tour = function (el) {
    var q = function (s) { return function () { return el.querySelector(s); }; };
    return [
      { el: q('.seg'), title: 'Pick the traffic', text: 'Real traffic is uneven. Steady, bursty, one-hit-wonder and all-different traffic behave very differently.' },
      { el: q('.timeline-req'), title: 'One bar per request', text: 'Green was served from the cache, blue went to the model, red was rejected by the rate limiter.' },
      { el: q('.meters'), title: 'The numbers that matter', text: 'Hit rate, rejections, latency and spend update as you change anything on the left.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
