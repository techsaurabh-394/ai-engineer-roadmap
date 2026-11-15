/* Agent loop lab: watch a scripted model, its tools and the conversation, one step at a time.
   Maths/logic: scripts/site/core/sims.js (a tested port of 04-agentic-ai/code/agent_trace.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, A = AIER.agent;
  AIER.labs = AIER.labs || {};

  var SCEN = {
    math_note: { title: 'Calculate, then save a note', goal: 'What is 12*7+5? Save the answer as a note.', approve: ['save_note'],
      script: [{ text: 'I will use the calculator.', tool_calls: [{ name: 'calculator', input: { expression: '12*7+5' } }] },
        { text: 'Now I will save the result.', tool_calls: [{ name: 'save_note', input: { text: '12*7+5 = 89' } }] },
        { text: '12*7+5 is 89, and I saved it as a note.' }] },
    country_math: { title: 'Look something up, then calculate', goal: 'France has about 68 million people. How many people are in three Frances?', approve: [],
      script: [{ text: 'I need the population first.', tool_calls: [{ name: 'lookup', input: { topic: 'france' } }] },
        { text: 'The population is 68 million. Three times that:', tool_calls: [{ name: 'calculator', input: { expression: '68*3' } }] },
        { text: 'Three Frances would be about 204 million people.' }] },
    bad_tool: { title: 'The model calls a tool that does not exist', goal: 'Email me the result of 6*7.', approve: [],
      script: [{ text: 'I will send an email.', tool_calls: [{ name: 'send_email', input: { to: 'me', body: '6*7' } }] },
        { text: 'There is no email tool. I can still compute the number.', tool_calls: [{ name: 'calculator', input: { expression: '6*7' } }] },
        { text: '6*7 is 42. I cannot send email with the tools I have.' }] },
    tool_error: { title: 'A tool fails and the model recovers', goal: 'Split 10 into 2 equal parts.', approve: [],
      script: [{ text: 'Divide 10 by 0 parts?', tool_calls: [{ name: 'calculator', input: { expression: '10/0' } }] },
        { text: 'That was a mistake. Two parts means divide by 2.', tool_calls: [{ name: 'calculator', input: { expression: '10/2' } }] },
        { text: 'Each part is 5.' }] },
    loop_forever: { title: 'A confused model that never finishes', goal: 'Keep checking until you are sure.', approve: [],
      script: [{ text: 'Let me check again.', tool_calls: [{ name: 'calculator', input: { expression: '1+1' } }] }] }
  };
  var LESSON_START = { 'agent-loop-anim': 'math_note', 'tool-calling-flow': 'country_math', 'react-loop': 'country_math', '14-human-in-the-loop': 'math_note', '16-agent-guardrails': 'loop_forever', '12-guardrails': 'bad_tool' };

  AIER.labs.agent = {
    mount: function (el, ctx) {
      var st = { key: (ctx.lesson && LESSON_START[ctx.lesson.name]) || 'math_note', cap: 6, deny: false, at: 0, timer: null, run: null };
      var seg = ui.seg({ label: 'Scenario', options: Object.keys(SCEN).map(function (k) { return [k, SCEN[k].title]; }), value: st.key, onchange: function (v) { st.key = v; st.at = 0; stop(); compute(); paint(); } });
      var capSl = ui.slider({ label: 'Step limit (max_steps)', min: 1, max: 8, step: 1, value: st.cap, help: 'The loop stops here even if the model has not finished. Try 2 on the first scenario.', oninput: function (v) { st.cap = v; compute(); paint(); } });
      var denyBox = h('input', { type: 'checkbox', id: ui.id('deny'), onchange: function () { st.deny = denyBox.checked; compute(); paint(); } });
      var denyRow = h('label.check', { 'for': denyBox.id }, denyBox, h('span', 'Act as the human reviewer and DENY risky actions'));
      var transcript = h('ol.transcript', { 'aria-label': 'What happened, in order' });
      var side = h('div.agent-side');
      var summary = h('p.summary-line', { role: 'status' });
      var playBtn = h('button.btn.small.primary', { type: 'button' }), backBtn = h('button.btn.small', { type: 'button' }, icon('prev', 'sm'), 'Back'),
        nextBtn = h('button.btn.small', { type: 'button' }, 'Next', icon('next', 'sm')), resetBtn = h('button.btn.small.ghost', { type: 'button' }, icon('redo', 'sm'), 'Reset');

      function compute() {
        var sc = SCEN[st.key], dec = {};
        sc.approve.forEach(function (n) { dec[n] = !st.deny; });
        st.run = A.runTrace(sc.script, sc.goal, st.cap, sc.approve, dec);
        st.at = Math.min(st.at, st.run.events.length - 1);
      }
      function setPlay(on) { ui.clear(playBtn); playBtn.appendChild(icon(on ? 'pause' : 'play', 'sm')); playBtn.appendChild(h('span', on ? 'Pause' : 'Play')); }
      function stop() { if (st.timer) { root.clearInterval(st.timer); st.timer = null; } setPlay(false); }
      function go(n) { st.at = Math.max(0, Math.min(st.run.events.length - 1, n)); paint(); }
      playBtn.addEventListener('click', function () {
        if (st.timer) { stop(); return; }
        if (st.at >= st.run.events.length - 1) st.at = 0;
        setPlay(true); paint();
        st.timer = root.setInterval(function () { if (st.at >= st.run.events.length - 1) { stop(); return; } go(st.at + 1); if (st.at >= st.run.events.length - 1) stop(); }, ui.reducedMotion() ? 2400 : 1500);
      });
      backBtn.addEventListener('click', function () { stop(); go(st.at - 1); });
      nextBtn.addEventListener('click', function () { stop(); go(st.at + 1); });
      resetBtn.addEventListener('click', function () { stop(); go(0); });

      function row(e, active) {
        var cls = 'li.ev.' + e.type + (e.status ? '.' + e.status : '') + (active ? '.now' : '');
        if (e.type === 'user') return h(cls, h('span.who', icon('home', 'sm'), 'You'), h('p', e.text));
        if (e.type === 'model') return h(cls, h('span.who', icon('spark', 'sm'), 'Model · step ' + e.step), e.text ? h('p', e.text) : null,
          e.calls.length ? h('p.call', icon('code', 'sm'), 'asks to run ', h('code', e.calls.map(function (c) { return c.name + '(' + JSON.stringify(c.input) + ')'; }).join(', '))) : null);
        if (e.type === 'approval') return h(cls + (e.approved ? '.ok' : '.denied'), h('span.who', icon('info', 'sm'), 'Human approval'),
          h('p', (e.approved ? 'Approved: ' : 'Denied: '), h('code', e.name + '(' + JSON.stringify(e.input) + ')')));
        if (e.type === 'tool') return h(cls, h('span.who', icon('flask', 'sm'), 'Tool · ' + e.name),
          h('p', h('b', e.status === 'ok' ? 'Result: ' : e.status === 'denied' ? 'Blocked: ' : 'Error: '), e.output));
        if (e.type === 'final') return h(cls, h('span.who', icon('check', 'sm'), 'Final answer'), h('p', e.text));
        return h(cls, h('span.who', icon('close', 'sm'), 'Loop stopped'), h('p', e.text));
      }
      function paint() {
        var r = st.run, evs = r.events, i = st.at, e = evs[i], sc = SCEN[st.key];
        ui.clear(transcript);
        evs.slice(0, i + 1).forEach(function (x, k) { transcript.appendChild(row(x, k === i)); });
        var last = transcript.lastElementChild; if (last && last.scrollIntoView && st.timer) last.scrollIntoView({ block: 'nearest' });
        // what the model "sees": the message list so far
        var sent = 1, tools = 0;
        evs.slice(0, i + 1).forEach(function (x) { if (x.type === 'model') sent++; if (x.type === 'tool') { tools++; } });
        var results = evs.slice(0, i + 1).filter(function (x) { return x.type === 'tool'; }).length;
        ui.clear(side);
        side.appendChild(h('section.card.mini', h('h3', 'Why this step happened'), h('p', why(e))));
        side.appendChild(h('section.card.mini', h('h3', 'Tools the model can use'), h('ul.tool-specs', A.SPECS.map(function (t) {
          var used = e.type === 'tool' && e.name === t.name || (e.type === 'model' && e.calls.some(function (c) { return c.name === t.name; }));
          return h('li' + (used ? '.on' : ''), h('code', t.name), h('span', t.description), t.risk === 'writes' ? h('span.chip.warn', 'changes state') : null);
        }))));
        side.appendChild(h('section.card.mini', h('h3', 'Loop state'), h('dl.state',
          h('div', h('dt', 'Step'), h('dd', String(e.step || 0) + ' of at most ' + st.cap)),
          h('div', h('dt', 'Tool calls so far'), h('dd', String(results))),
          h('div', h('dt', 'Notes saved'), h('dd', String(evs.slice(0, i + 1).filter(function (x) { return x.type === 'tool' && x.name === 'save_note' && x.status === 'ok'; }).length))))));
        var done = i === evs.length - 1;
        summary.textContent = done ? (r.stopped ? 'The loop hit the step limit after ' + r.steps_used + ' steps and stopped without an answer. This is what the cap is for.'
          : 'Finished in ' + r.steps_used + ' model calls and ' + evs.filter(function (x) { return x.type === 'tool'; }).length + ' tool calls.') : 'Event ' + (i + 1) + ' of ' + evs.length + '.';
        backBtn.disabled = i === 0; nextBtn.disabled = done;
      }
      function why(e) {
        if (e.type === 'user') return 'The loop starts with the user’s goal as the first message. The model also receives the list of tools and their JSON schemas.';
        if (e.type === 'model') return e.calls.length ? 'The whole conversation so far is sent to the model. It answers with text and one or more tool calls instead of a final answer. The model only asks; your code decides whether to run it.' : 'The model returned no tool calls. That is the signal that it is done, so the loop ends.';
        if (e.type === 'approval') return 'A tool that changes something can be paused for a person. Nothing runs until they answer. Use the checkbox above to deny it and see what the model receives.';
        if (e.type === 'tool') return e.status === 'ok' ? 'Your code ran the tool and appends the result to the conversation as a new message. The model reads it on the next turn.'
          : e.status === 'denied' ? 'The result tells the model the action was refused, so it can explain or choose another way. Silence would make it retry.'
          : 'A failing or unknown tool is returned as text, not thrown. The model can read the error and try something else. That is how agents recover.';
        if (e.type === 'final') return 'No tool call, so this text is the answer. Everything before it was the agent working.';
        return 'max_steps was reached. Without this cap a confused model could call tools forever and run up your bill.';
      }
      el.appendChild(ui.honest('The model here is scripted: it replays fixed replies so the lab is repeatable. The loop, tools, approval and step cap are the real logic. A real model decides each reply itself.', '04-agentic-ai/code/agent_trace.py', ctx.repo));
      el.appendChild(h('div.lab-grid',
        h('div.controls', h('div.control', h('span.lbl', 'Scenario'), seg), capSl, denyRow, h('div.row-btns', backBtn, playBtn, nextBtn, resetBtn), summary),
        h('div.agent-main', transcript, side)));
      compute(); setPlay(false); paint();
      return stop;
    }
  };
  AIER.labs.agent.tour = function (el) {
    var q = function (s) { return function () { return el.querySelector(s); }; };
    return [
      { el: q('.seg'), title: 'Pick a scenario', text: 'Each one shows a different thing agents run into: a normal task, a missing tool, a failing tool and a loop that never ends.' },
      { el: q('.transcript'), title: 'The conversation', text: 'Every model reply, tool call and tool result in order. The agent loop is this list growing.' },
      { el: function () { return el.querySelectorAll('input[type=range]')[0].closest('.control'); }, title: 'The step limit', text: 'Lower it and replay the first scenario: the loop stops before the model finishes.' },
      { el: q('.check'), title: 'Be the human reviewer', text: 'Tick this, then play the first scenario. The note is never saved and the model is told why.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
