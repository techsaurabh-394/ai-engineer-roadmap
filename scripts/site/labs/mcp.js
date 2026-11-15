/* MCP lab: a client and a server exchanging real JSON-RPC messages.
   Logic: scripts/site/core/sims.js (a tested port of 06-mcp/code/mcp_mini_jsonrpc.py). */
(function (root) {
  'use strict';
  var AIER = root.AIER, ui = AIER.ui, h = ui.h, icon = ui.icon, M = AIER.mcp;
  AIER.labs = AIER.labs || {};

  var LIFE = [
    { label: 'initialize', actor: 'Handshake', why: 'The client says hello and states which protocol version it speaks. The server answers with its own version, what it can do (capabilities) and its name.', msg: { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: M.PROTOCOL_VERSION, capabilities: {}, clientInfo: { name: 'mini-client', version: '0.1.0' } } } },
    { label: 'initialized', actor: 'Handshake', why: 'The client confirms it is ready. This is a notification: it has no "id", so JSON-RPC says the server must not reply. Until this arrives the server refuses other requests.', msg: { jsonrpc: '2.0', method: 'notifications/initialized' } },
    { label: 'tools/list', actor: 'Discovery', why: 'The client asks what tools exist. Each comes with a description and a JSON Schema. This is how a model learns what it may call, without anyone hard-coding tools into the app.', msg: { jsonrpc: '2.0', id: 2, method: 'tools/list' } },
    { label: 'tools/call', actor: 'Invocation', why: 'The client runs one tool with arguments that match its schema. The result is a list of content blocks plus an isError flag.', msg: { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'add_numbers', arguments: { a: 2, b: 40 } } } },
    { label: 'resources/read', actor: 'Resources', why: 'Resources are data the server exposes by URI (a file, a record, a note). Unlike a tool, reading one has no side effects, and the application decides when to read it.', msg: { jsonrpc: '2.0', id: 4, method: 'resources/read', params: { uri: 'notes://welcome' } } },
    { label: 'prompts/get', actor: 'Prompts', why: 'Prompts are reusable templates the server offers. The client fills the arguments and receives ready-to-send messages.', msg: { jsonrpc: '2.0', id: 5, method: 'prompts/get', params: { name: 'summarize', arguments: { text: 'MCP is a protocol.' } } } }
  ];
  var MISTAKES = {
    before_init: { title: 'Call a tool before the handshake', init: false, msg: { jsonrpc: '2.0', id: 1, method: 'tools/list' }, lesson: 'The server insists on the handshake first. -32600 means "invalid request" here.' },
    unknown_method: { title: 'Call a method that does not exist', init: true, msg: { jsonrpc: '2.0', id: 9, method: 'tools/destroy' }, lesson: '-32601 is JSON-RPC’s "method not found". It is a protocol error.' },
    unknown_tool: { title: 'Call a tool that does not exist', init: true, msg: { jsonrpc: '2.0', id: 9, method: 'tools/call', params: { name: 'nope', arguments: {} } }, lesson: '-32602 means invalid parameters: the method exists but the tool name is wrong.' },
    missing_arg: { title: 'Leave out a required argument', init: true, msg: { jsonrpc: '2.0', id: 9, method: 'tools/call', params: { name: 'add_numbers', arguments: { a: 1 } } }, lesson: 'The server checks required arguments from the schema before running anything.' },
    tool_fails: { title: 'The tool runs but fails inside', init: true, msg: { jsonrpc: '2.0', id: 9, method: 'tools/call', params: { name: 'add_numbers', arguments: { a: 1, b: 'x' } } }, lesson: 'NOT a protocol error: the reply is a normal result with isError: true. The model can read the message and try again.' },
    bad_json: { title: 'Send text that is not JSON', init: true, raw: '{not json', lesson: '-32700 is "parse error". The server cannot even tell which request this was, so the id is null.' }
  };
  var INIT = [JSON.stringify(LIFE[0].msg), JSON.stringify(LIFE[1].msg)];

  function pretty(o) { return JSON.stringify(o, null, 2); }

  AIER.labs.mcp = {
    mount: function (el, ctx) {
      var st = { mode: 'life', at: 0, key: 'before_init', timer: null, custom: '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"word_count","arguments":{"text":"learn MCP step by step"}}}' };
      var modeSeg = ui.seg({ label: 'What to explore', options: [['life', 'Guided lifecycle'], ['mistakes', 'What goes wrong'], ['own', 'Write your own request']], value: st.mode, onchange: function (v) { st.mode = v; stop(); layout(); paint(); } });
      var chooser = h('div.control'), stage = h('div.mcp-stage'), notes = h('div.mcp-notes');
      var playBtn = h('button.btn.small.primary', { type: 'button' }), backBtn = h('button.btn.small', { type: 'button' }, icon('prev', 'sm'), 'Back'),
        nextBtn = h('button.btn.small', { type: 'button' }, 'Next', icon('next', 'sm')), resetBtn = h('button.btn.small.ghost', { type: 'button' }, icon('redo', 'sm'), 'Reset');
      var transport = h('div.row-btns', backBtn, playBtn, nextBtn, resetBtn);

      function setPlay(on) { ui.clear(playBtn); playBtn.appendChild(icon(on ? 'pause' : 'play', 'sm')); playBtn.appendChild(h('span', on ? 'Pause' : 'Play')); }
      function stop() { if (st.timer) { root.clearInterval(st.timer); st.timer = null; } setPlay(false); }
      function go(n) { st.at = Math.max(0, Math.min(LIFE.length - 1, n)); paint(); }
      playBtn.addEventListener('click', function () {
        if (st.timer) { stop(); return; }
        if (st.at >= LIFE.length - 1) st.at = 0;
        setPlay(true); paint();
        st.timer = root.setInterval(function () { if (st.at >= LIFE.length - 1) { stop(); return; } go(st.at + 1); if (st.at >= LIFE.length - 1) stop(); }, ui.reducedMotion() ? 3200 : 2400);
      });
      backBtn.addEventListener('click', function () { stop(); go(st.at - 1); });
      nextBtn.addEventListener('click', function () { stop(); go(st.at + 1); });
      resetBtn.addEventListener('click', function () { stop(); go(0); });

      function lane(title, sub) { return h('div.lane', h('strong', title), h('span', sub)); }
      function msgBox(dir, label, obj, raw, kind) {
        return h('div.msg.' + dir + (kind ? '.' + kind : ''), h('p.mh', icon(dir === 'out' ? 'next' : 'prev', 'sm'), label),
          h('pre.json', { tabindex: '0' }, raw !== undefined ? raw : pretty(obj)));
      }
      function exchange(msgs, upto) {
        // msgs: [{raw, label}] sent in order to a fresh server; show every exchange up to index `upto`
        var log = M.replay(msgs.map(function (m) { return m.raw; })), out = h('div.exchanges');
        log.slice(0, upto + 1).forEach(function (e, i) {
          var req = null; try { req = JSON.parse(e['in']); } catch (x) { /* shown raw */ }
          var isNote = req && typeof req === 'object' && !('id' in req);
          out.appendChild(h('div.ex' + (i === upto ? '.now' : ''),
            msgBox('out', 'Client → Server' + (isNote ? '  (notification: no id)' : ''), req, req ? undefined : e['in']),
            e.out ? msgBox('in', 'Server → Client' + (e.out.error ? '  (error ' + e.out.error.code + ')' : ''), e.out, undefined, e.out.error ? 'err' : (e.out.result && e.out.result.isError ? 'warn' : 'ok'))
              : h('div.msg.in.none', h('p.mh', icon('prev', 'sm'), 'Server → Client'), h('p.silent', 'No reply. Notifications never get one.'))));
        });
        return out;
      }
      function layout() {
        ui.clear(chooser);
        if (st.mode === 'life') chooser.appendChild(h('p.help', 'Six messages, in the order a real MCP session sends them. Step through, or press Play.'));
        else if (st.mode === 'mistakes') {
          var sel = h('select', { 'aria-label': 'Mistake', onchange: function () { st.key = sel.value; paint(); } }, Object.keys(MISTAKES).map(function (k) { return h('option', { value: k, selected: k === st.key }, MISTAKES[k].title); }));
          chooser.appendChild(h('label', h('span', 'Pick a mistake')));
          chooser.appendChild(sel);
        } else {
          var ta = h('textarea.textin', { rows: 6, spellcheck: 'false', 'aria-label': 'Your JSON-RPC request' }); ta.value = st.custom;
          ta.addEventListener('input', function () { st.custom = ta.value; paint(); });
          chooser.appendChild(h('label', h('span', 'A JSON-RPC message (the handshake is done for you)')));
          chooser.appendChild(ta);
          chooser.appendChild(h('div.chips', [['List tools', '{"jsonrpc":"2.0","id":7,"method":"tools/list"}'], ['Word count', '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"word_count","arguments":{"text":"learn MCP step by step"}}}'],
            ['List prompts', '{"jsonrpc":"2.0","id":7,"method":"prompts/list"}'], ['Ping', '{"jsonrpc":"2.0","id":7,"method":"ping"}']].map(function (p) {
            return h('button.chip', { type: 'button', onclick: function () { ta.value = p[1]; st.custom = p[1]; paint(); } }, p[0]);
          })));
        }
        transport.hidden = st.mode !== 'life';
      }
      function paint() {
        ui.clear(stage); ui.clear(notes);
        backBtn.disabled = st.at === 0; nextBtn.disabled = st.at === LIFE.length - 1;
        if (st.mode === 'life') {
          var step = LIFE[st.at];
          stage.appendChild(h('ol.pipe-strip', { 'aria-label': 'Session steps' }, LIFE.map(function (s, i) {
            return h('li', h('button' + (i === st.at ? '.on' : i < st.at ? '.done' : ''), { type: 'button', 'aria-label': 'Step ' + (i + 1) + ': ' + s.label, 'aria-current': i === st.at ? 'step' : null, onclick: function () { stop(); go(i); } },
              h('span.n', { 'aria-hidden': 'true' }, String(i + 1)), h('span.l', s.label)));
          })));
          stage.appendChild(h('div.lanes', lane('MCP client', 'the AI app'), h('span.wire', { 'aria-hidden': 'true' }, '⇄ JSON-RPC 2.0'), lane('MCP server', 'tools · resources · prompts')));
          stage.appendChild(exchange(LIFE.map(function (s) { return { raw: JSON.stringify(s.msg) }; }), st.at));
          notes.appendChild(h('section.card.mini', h('span.k', step.actor), h('h3', step.label), h('p', step.why)));
        } else if (st.mode === 'mistakes') {
          var m = MISTAKES[st.key], raws = (m.init ? INIT.slice() : []).concat([m.raw !== undefined ? m.raw : JSON.stringify(m.msg)]);
          stage.appendChild(h('div.lanes', lane('MCP client', 'the AI app'), h('span.wire', { 'aria-hidden': 'true' }, '⇄ JSON-RPC 2.0'), lane('MCP server', m.init ? 'handshake already done' : 'NOT initialized')));
          var log = M.replay(raws), e = log[log.length - 1];
          var out = h('div.exchanges', h('div.ex.now', msgBox('out', 'Client → Server', null, m.raw !== undefined ? m.raw : pretty(m.msg)),
            e.out ? msgBox('in', 'Server → Client' + (e.out.error ? '  (error ' + e.out.error.code + ')' : ''), e.out, undefined, e.out.error ? 'err' : 'warn') : h('p', 'No reply.')));
          stage.appendChild(out);
          notes.appendChild(h('section.card.mini', h('span.k', 'What to learn'), h('h3', m.title), h('p', m.lesson)));
        } else {
          var parsed = null, bad = null;
          try { parsed = JSON.parse(st.custom); } catch (x) { bad = x.message; }
          var raws2 = INIT.concat([st.custom]), log2 = M.replay(raws2), e2 = log2[log2.length - 1];
          stage.appendChild(h('div.lanes', lane('MCP client', 'you'), h('span.wire', { 'aria-hidden': 'true' }, '⇄ JSON-RPC 2.0'), lane('MCP server', 'handshake already done')));
          stage.appendChild(h('div.exchanges', h('div.ex.now', parsed ? msgBox('out', 'Client → Server', parsed) : msgBox('out', 'Client → Server (not valid JSON)', null, st.custom),
            e2.out ? msgBox('in', 'Server → Client' + (e2.out.error ? '  (error ' + e2.out.error.code + ')' : ''), e2.out, undefined, e2.out.error ? 'err' : (e2.out.result && e2.out.result.isError ? 'warn' : 'ok')) : h('div.msg.in.none', h('p.silent', 'No reply: a message without an "id" is a notification.')))));
          notes.appendChild(h('section.card.mini', h('span.k', 'Try this'), h('h3', 'Break it on purpose'),
            h('p', 'Change the method to something unknown, remove "arguments", or delete the id. Each one shows a different reply from the real server logic.'),
            bad ? h('p.note-inline', 'Not valid JSON yet: ' + bad) : null));
        }
        notes.appendChild(h('section.card.mini', h('h3', 'Real world'), h('p', 'Real MCP servers speak this same JSON-RPC over stdio or Streamable HTTP, and an SDK handles it for you. This lab’s server is a teaching toy with two tools; it has no authentication, which production servers over HTTP must have.')));
      }
      el.appendChild(ui.honest('The server here is a small teaching server (the same logic as mcp_mini_jsonrpc.py), not the official SDK. The messages follow the shapes of the Model Context Protocol; check the spec for the current protocol version.', '06-mcp/code/mcp_mini_jsonrpc.py', ctx.repo));
      el.appendChild(h('div.lab-grid', h('div.controls', h('div.control', h('span.lbl', 'What to explore'), modeSeg), chooser, transport),
        h('div.mcp-main', stage, notes)));
      layout(); setPlay(false); paint();
      return stop;
    }
  };
  AIER.labs.mcp.tour = function (el) {
    var q = function (s) { return function () { return el.querySelector(s); }; };
    return [
      { el: q('.seg'), title: 'Three ways in', text: 'Follow a normal session, trigger common mistakes, or write your own request.' },
      { el: q('.lanes'), title: 'Two sides, one wire', text: 'A client (the AI app) and a server (tools, resources, prompts) swap JSON-RPC messages.' },
      { el: q('.exchanges'), title: 'Real messages', text: 'Every box is a message the server logic actually received or produced.' }
    ];
  };
})(typeof self !== 'undefined' ? self : globalThis);
