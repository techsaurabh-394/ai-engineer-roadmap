/* JavaScript ports of three small Python modules, so the labs run the same logic in the browser:
     04-agentic-ai/code/agent_trace.py     -> AIER.agent   (the agent loop as a list of events)
     06-mcp/code/mcp_mini_jsonrpc.py       -> AIER.mcp     (a JSON-RPC server with MCP-shaped methods)
     07-production-ai/code/prod_request_sim.py + prod_cache_ratelimit.py -> AIER.reqsim
   No DOM. scripts/test_site_parity.py runs these under node and compares them with the Python. */
(function (root) {
  'use strict';
  var AIER = root.AIER || (root.AIER = {});
  if (typeof module === 'object' && module.exports && !AIER.py) require('./py.js');
  var py = AIER.py;

  // ================================================================== agent
  var SPECS = [
    { name: 'calculator', description: "Evaluate an arithmetic expression such as '12*7+5'.", required: ['expression'], risk: 'safe' },
    { name: 'lookup', description: 'Look up a short fact sheet about a country.', required: ['topic'], risk: 'safe' },
    { name: 'save_note', description: 'Save a short text note (changes state, so a person may want to approve it).', required: ['text'], risk: 'writes' }
  ];
  var FACTS = { france: 'France has a population of about 68 million. Capital: Paris.', japan: 'Japan has a population of about 124 million. Capital: Tokyo.' };
  function Parser(text) { this.s = text; this.i = 0; }
  Parser.prototype.peek = function () { while (this.i < this.s.length && this.s.charAt(this.i) === ' ') this.i++; return this.i < this.s.length ? this.s.charAt(this.i) : ''; };
  Parser.prototype.expr = function () {
    var v = this.term();
    while (this.peek() === '+' || this.peek() === '-') { var op = this.s.charAt(this.i++), r = this.term(); v = op === '+' ? v + r : v - r; }
    return v;
  };
  Parser.prototype.term = function () {
    var v = this.factor();
    while (this.peek() === '*' || this.peek() === '/') {
      var op = this.s.charAt(this.i++), r = this.factor();
      if (op === '/' && r === 0) throw new Error('division by zero');
      v = op === '*' ? v * r : v / r;
    }
    return v;
  };
  Parser.prototype.factor = function () {
    var c = this.peek();
    if (c === '-') { this.i++; return -this.factor(); }
    if (c === '(') { this.i++; var v = this.expr(); if (this.peek() !== ')') throw new Error('missing )'); this.i++; return v; }
    var j = this.i;
    while (j < this.s.length && /[0-9.]/.test(this.s.charAt(j))) j++;
    if (j === this.i) throw new Error('expected a number');
    var token = this.s.slice(this.i, j);
    if (token.split('.').length - 1 > 1 || token === '.') throw new Error('bad number');
    this.i = j;
    return parseFloat(token);
  };
  function fmt(v) {
    if (v === Infinity) return 'inf';
    if (v === -Infinity) return '-inf';
    if (Math.floor(v) === v && Math.abs(v) < 1e15) return String(v);
    return py.repr(v);
  }
  function calculator(expression) {
    if (!expression || !/^[0-9+\-*/(). ]+$/.test(expression)) return 'Error: only numbers and + - * / ( ) are allowed.';
    var p = new Parser(expression);
    try { var v = p.expr(); if (p.peek() !== '') throw new Error('unexpected character'); return fmt(v); } catch (e) { return 'Error: ' + e.message; }
  }
  function lookup(topic) { var k = topic.toLowerCase(); return Object.prototype.hasOwnProperty.call(FACTS, k) ? FACTS[k] : "No entry for '" + topic + "'."; }
  /** runTrace(script, goal, maxSteps, approve, decisions) -> {events, final, steps_used, notes, messages, stopped} */
  function runTrace(script, goal, maxSteps, approve, decisions) {
    approve = approve || []; decisions = decisions || {};
    var known = {}; SPECS.forEach(function (t) { known[t.name] = t; });
    var notes = [], events = [{ type: 'user', text: goal }], messages = 1, i = 0, step, n;
    for (step = 1; step <= maxSteps; step++) {
      var item = script[Math.min(i, script.length - 1)]; i++;
      var calls = (item.tool_calls || []).map(function (c, k) { return { id: 'call_' + i + '_' + k, name: c.name, input: c.input }; });
      events.push({ type: 'model', step: step, text: item.text || '', calls: calls });
      messages++;
      if (!calls.length) {
        events.push({ type: 'final', step: step, text: item.text || '' });
        return { events: events, final: item.text || '', steps_used: step, notes: notes, messages: messages, stopped: false };
      }
      for (n = 0; n < calls.length; n++) {
        var c = calls[n], spec = known[c.name], out, status;
        if (!spec) { events.push({ type: 'tool', step: step, id: c.id, name: c.name, input: c.input, output: 'Error: unknown tool ' + c.name, status: 'error' }); continue; }
        if (approve.indexOf(c.name) !== -1) {
          var ok = Object.prototype.hasOwnProperty.call(decisions, c.name) ? !!decisions[c.name] : true;
          events.push({ type: 'approval', step: step, id: c.id, name: c.name, input: c.input, approved: ok });
          if (!ok) { events.push({ type: 'tool', step: step, id: c.id, name: c.name, input: c.input, output: 'Error: a person denied this action.', status: 'denied' }); continue; }
        }
        var missing = spec.required.filter(function (k) { return !(k in c.input); });
        if (missing.length) { out = "Error: missing argument '" + missing[0] + "'"; status = 'error'; }
        else if (c.name === 'calculator') { out = calculator(c.input.expression); status = out.indexOf('Error') === 0 ? 'error' : 'ok'; }
        else if (c.name === 'lookup') { out = lookup(c.input.topic); status = 'ok'; }
        else { notes.push(c.input.text); out = 'Saved note #' + notes.length + '.'; status = 'ok'; }
        events.push({ type: 'tool', step: step, id: c.id, name: c.name, input: c.input, output: out, status: status });
      }
      messages++;
    }
    events.push({ type: 'stopped', step: maxSteps, text: 'Stopped: reached max steps without finishing.' });
    return { events: events, final: null, steps_used: maxSteps, notes: notes, messages: messages, stopped: true };
  }
  AIER.agent = { SPECS: SPECS, calculator: calculator, lookup: lookup, runTrace: runTrace };

  // ================================================================== mcp
  var PROTOCOL_VERSION = '2025-06-18';
  function RpcError(code, message) { this.code = code; this.message = message; }
  function McpServer() {
    this.initialized = false;
    this.tools = {
      add_numbers: { description: 'Add two numbers and return the sum.', required: ['a', 'b'],
        inputSchema: { type: 'object', properties: { a: { type: 'number' }, b: { type: 'number' } }, required: ['a', 'b'] },
        fn: function (a) { if (typeof a.a !== 'number' || typeof a.b !== 'number') throw new Error('a and b must be numbers'); return fmt(a.a + a.b); } },
      word_count: { description: 'Count the words in a piece of text.', required: ['text'],
        inputSchema: { type: 'object', properties: { text: { type: 'string', minLength: 1 } }, required: ['text'] },
        fn: function (a) { if (typeof a.text !== 'string') throw new Error('text must be a string'); return String(py.split(a.text).length); } }
    };
    this.resources = { 'notes://welcome': ['Welcome', 'Hello from a resource.'] };
    this.prompts = { summarize: { description: 'Ask the model to summarize some text.', arguments: [{ name: 'text', required: true }] } };
  }
  McpServer.prototype.error = function (id, code, message) { return { jsonrpc: '2.0', id: id, error: { code: code, message: message } }; };
  /** handle(raw JSON string) -> reply object, or null for a notification */
  McpServer.prototype.handle = function (raw) {
    var msg;
    try { msg = JSON.parse(raw); } catch (e) { return this.error(null, -32700, 'Parse error'); }
    var isObj = msg && typeof msg === 'object' && !Array.isArray(msg);
    if (!isObj || msg.jsonrpc !== '2.0' || !('method' in msg)) return this.error(isObj && 'id' in msg ? msg.id : null, -32600, 'Invalid Request');
    var isNote = !('id' in msg), result;
    try { result = this.dispatch(msg.method, msg.params || {}); }
    catch (e) { if (!(e instanceof RpcError)) throw e; return isNote ? null : this.error(msg.id, e.code, e.message); }
    if (isNote) return null;
    return { jsonrpc: '2.0', id: msg.id, result: result };
  };
  McpServer.prototype.dispatch = function (method, params) {
    var self = this, names = Object.keys(this.tools);
    if (method === 'initialize') return { protocolVersion: PROTOCOL_VERSION, capabilities: { tools: { listChanged: false }, resources: {}, prompts: {} }, serverInfo: { name: 'mini-mcp', version: '0.1.0' } };
    if (method === 'notifications/initialized') { this.initialized = true; return null; }
    if (method === 'ping') return {};
    if (!this.initialized) throw new RpcError(-32600, 'Server not initialized');
    if (method === 'tools/list') return { tools: names.map(function (n) { return { name: n, description: self.tools[n].description, inputSchema: self.tools[n].inputSchema }; }) };
    if (method === 'tools/call') {
      var tool = this.tools[params.name];
      if (!tool || !Object.prototype.hasOwnProperty.call(this.tools, params.name)) throw new RpcError(-32602, 'Unknown tool: ' + params.name);
      var args = params.arguments || {};
      for (var k = 0; k < tool.required.length; k++) if (!(tool.required[k] in args)) throw new RpcError(-32602, 'Missing argument: ' + tool.required[k]);
      try { return { content: [{ type: 'text', text: tool.fn(args) }], isError: false }; }
      catch (e) { return { content: [{ type: 'text', text: 'Tool failed: ' + e.message }], isError: true }; }
    }
    if (method === 'resources/list') return { resources: Object.keys(this.resources).map(function (u) { return { uri: u, name: self.resources[u][0], mimeType: 'text/plain' }; }) };
    if (method === 'resources/read') {
      if (!Object.prototype.hasOwnProperty.call(this.resources, params.uri)) throw new RpcError(-32002, 'Resource not found');
      return { contents: [{ uri: params.uri, mimeType: 'text/plain', text: this.resources[params.uri][1] }] };
    }
    if (method === 'prompts/list') return { prompts: Object.keys(this.prompts).map(function (n) { var p = self.prompts[n]; return { name: n, description: p.description, arguments: p.arguments }; }) };
    if (method === 'prompts/get') {
      if (!Object.prototype.hasOwnProperty.call(this.prompts, params.name)) throw new RpcError(-32602, 'Unknown prompt: ' + params.name);
      var text = (params.arguments || {}).text;
      if (text === undefined || text === null) throw new RpcError(-32602, 'Missing argument: text');
      return { description: this.prompts[params.name].description, messages: [{ role: 'user', content: { type: 'text', text: 'Summarize this:\n' + text } }] };
    }
    throw new RpcError(-32601, 'Method not found: ' + method);
  };
  /** replay([raw, ...]) on a fresh server -> [{in, out}] */
  function replay(raws) { var s = new McpServer(); return raws.map(function (r) { return { 'in': r, out: s.handle(r) }; }); }
  AIER.mcp = { PROTOCOL_VERSION: PROTOCOL_VERSION, McpServer: McpServer, replay: replay };

  // ================================================================== request simulator
  var QUESTIONS = ['What are your opening hours?', 'How do I reset my password?', 'Where is my order?', 'Do you ship abroad?',
    'How do I cancel?', 'Is there a student discount?', 'Can I change my email?', 'What is the refund policy?'];
  var ones = []; for (var u = 0; u < 80; u++) ones.push(1);
  var PATTERNS = {
    steady: { title: 'Steady traffic', weights: [8, 5, 3, 2, 1, 1, 1, 1], blurb: 'One request about every second. Some questions are asked far more than others.' },
    burst: { title: 'Bursts', weights: [8, 5, 3, 2, 1, 1, 1, 1], blurb: 'Requests arrive in rushes of about ten per second, with quiet gaps between them.' },
    popular: { title: 'One question goes viral', weights: [30, 1, 1, 1, 1, 1, 1, 1], blurb: 'Almost everyone asks the same thing. Caching shines here.' },
    unique: { title: 'Every question is different', weights: ones, blurb: 'Eighty different questions asked evenly: a repeat is rare, so caching helps much less.' }
  };
  function questionText(q) { return q < QUESTIONS.length ? QUESTIONS[q] : 'Question number ' + (q + 1); }
  function makeTraffic(pattern, n, seed) {
    var rng = new py.Random(seed), weights = PATTERNS[pattern].weights, ids = weights.map(function (_, i) { return i; }), t = 0, out = [];
    for (var k = 0; k < n; k++) {
      var q = rng.choice(ids, weights), gap;
      if (pattern === 'burst') gap = (k + 1) % 12 ? 0.05 + rng.random() * 0.1 : 6.0 + rng.random() * 4.0;
      else gap = 0.5 + rng.random() * 1.5;
      t = t + gap; out.push([t, q]);
    }
    return out;
  }
  /** simulate(traffic, opts) with opts {ttl, rate, capacity, order, llmMs, cacheMs, price} */
  function simulate(traffic, o) {
    o = o || {};
    var ttl = o.ttl === undefined ? 30.0 : o.ttl, rate = o.rate === undefined ? 1.0 : o.rate, cap = o.capacity === undefined ? 3 : o.capacity;
    var order = o.order || 'limit_first', llmMs = o.llmMs === undefined ? 1800 : o.llmMs, cacheMs = o.cacheMs === undefined ? 15 : o.cacheMs, price = o.price === undefined ? 0.01 : o.price;
    var data = new Map(), tokens = cap, last = 0, events = [], now = 0;     // TTLCache + TokenBucket on a virtual clock (the bucket starts at t = 0)
    function cacheGet(key) {
      var item = data.get(key);
      if (item !== undefined) { if (now < item) return true; data.delete(key); }
      return false;
    }
    function allow() {
      tokens = Math.min(cap, tokens + (now - last) * rate); last = now;
      if (tokens >= 1.0) { tokens -= 1.0; return true; }
      return false;
    }
    traffic.forEach(function (tq) {
      now = tq[0]; var q = tq[1], key = questionText(q);
      if (order === 'limit_first') {
        if (!allow()) { events.push({ t: now, q: q, kind: 'limited', ms: 0 }); return; }
        if (cacheGet(key)) events.push({ t: now, q: q, kind: 'hit', ms: cacheMs });
        else { data.set(key, now + ttl); events.push({ t: now, q: q, kind: 'llm', ms: llmMs }); }
      } else {
        if (cacheGet(key)) events.push({ t: now, q: q, kind: 'hit', ms: cacheMs });
        else if (!allow()) events.push({ t: now, q: q, kind: 'limited', ms: 0 });
        else { data.set(key, now + ttl); events.push({ t: now, q: q, kind: 'llm', ms: llmMs }); }
      }
    });
    var hits = 0, llm = 0, limited = 0;
    events.forEach(function (e) { if (e.kind === 'hit') hits++; else if (e.kind === 'llm') llm++; else limited++; });
    var served = hits + llm;
    return { events: events, summary: { total: events.length, hits: hits, llm: llm, limited: limited, hit_rate: served ? hits / served : 0.0,
      cost: llm * price, saved: hits * price, avg_ms: served ? (hits * cacheMs + llm * llmMs) / served : 0.0 } };
  }
  AIER.reqsim = { QUESTIONS: QUESTIONS, PATTERNS: PATTERNS, questionText: questionText, makeTraffic: makeTraffic, simulate: simulate };

  if (typeof module === 'object' && module.exports) module.exports = { agent: AIER.agent, mcp: AIER.mcp, reqsim: AIER.reqsim };
})(typeof self !== 'undefined' ? self : globalThis);
