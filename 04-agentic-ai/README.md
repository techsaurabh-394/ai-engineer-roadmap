# Phase 4 · Agentic AI

**Goal:** build AI agents from scratch (LLM + tools + loop + memory + safety) so you understand exactly what agent frameworks do for you, and can make agents that are reliable enough to ship.

**Who it's for:** beginners who know how LLMs pick tokens and how prompts work ([Phase 3](../03-llm-concepts/README.md)) and now want models that **take actions**, not just write text.

**Prerequisites:**
- Python functions, dicts, and JSON.
- [Phase 3 · LLM Concepts](../03-llm-concepts/README.md), especially [context window](../03-llm-concepts/topics/08-context-window.md), [hallucination](../03-llm-concepts/topics/13-hallucination.md), and [prompt engineering](../03-llm-concepts/topics/14-prompt-engineering.md).
- Helpful: [Phase 1 · RAG Foundations](../01-rag-foundations/README.md) (long-term memory is retrieval).

**Time:** about 8-10 hours, including the code.

## Big idea
An **agent** is an LLM in a loop. At each step the model decides what to do next, usually by calling a **tool** (a function you wrote). Your code runs the tool, hands back the result, and the model decides again, until it's done or a limit is hit.

![The agent loop](../assets/agent-loop.svg)

The model never runs anything itself. It *asks*; your code *acts*. That is where you put memory, approvals, and guardrails.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | What is an AI Agent? | Beginner | [01-what-is-an-ai-agent.md](topics/01-what-is-an-ai-agent.md) |
| 02 | Tool Calling | Beginner | [02-tool-calling.md](topics/02-tool-calling.md) |
| 03 | Function Calling | Beginner | [03-function-calling.md](topics/03-function-calling.md) |
| 04 | ReAct Pattern | Intermediate | [04-react-pattern.md](topics/04-react-pattern.md) |
| 05 | Planning Agents | Intermediate | [05-planning-agents.md](topics/05-planning-agents.md) |
| 06 | Reflection | Intermediate | [06-reflection.md](topics/06-reflection.md) |
| 07 | Memory | Beginner | [07-memory.md](topics/07-memory.md) |
| 08 | Short-term Memory | Beginner | [08-short-term-memory.md](topics/08-short-term-memory.md) |
| 09 | Long-term Memory | Intermediate | [09-long-term-memory.md](topics/09-long-term-memory.md) |
| 10 | Semantic Memory | Intermediate | [10-semantic-memory.md](topics/10-semantic-memory.md) |
| 11 | Episodic Memory | Intermediate | [11-episodic-memory.md](topics/11-episodic-memory.md) |
| 12 | Agent Loops | Intermediate | [12-agent-loops.md](topics/12-agent-loops.md) |
| 13 | Multi-Agent Systems | Advanced | [13-multi-agent-systems.md](topics/13-multi-agent-systems.md) |
| 14 | Human in the Loop | Intermediate | [14-human-in-the-loop.md](topics/14-human-in-the-loop.md) |
| 15 | Agent Evaluation | Advanced | [15-agent-evaluation.md](topics/15-agent-evaluation.md) |
| 16 | Agent Guardrails | Advanced | [16-agent-guardrails.md](topics/16-agent-guardrails.md) |

## Learning order
1. **Foundations (01-03).** What an agent is, and how a model asks your code to run a function. Run `agent.py` while reading.
2. **Reasoning patterns (04-06).** ReAct (think → act → observe), planning up front, and reflection (critique and retry). Run `agent_react.py`.
3. **Memory (07-11).** Start with the overview (07), then the four types. Run `agent_memory.py`.
4. **Putting it together (12-14).** The loop and its stop conditions, multiple agents, and human approval. Run `agent_multi.py`.
5. **Shipping it (15-16).** How to measure an agent and how to keep it safe.

Short on time? Read 01, 02, 04, 07, 12, 16 first.

## Run the code
Everything runs offline with scripted models: standard-library Python, no API key.

```bash
# from the repo root
python 04-agentic-ai/code/agent.py           # the core loop with two tools (scripted model)
python 04-agentic-ai/code/agent_react.py     # ReAct loop, JSON tool schemas, validated dispatch
python 04-agentic-ai/code/agent_memory.py    # short-term window, long-term store, episodic log, semantic facts
python 04-agentic-ai/code/agent_multi.py     # planner -> worker -> critic, approval gate, guardrails

python -m pytest -q 04-agentic-ai/code       # run the tests (needs: pip install pytest)

# Optional, with real Claude tool use:
pip install anthropic
ANTHROPIC_API_KEY=... python 04-agentic-ai/code/agent.py "What is 12*7+5? Then save it as a note."
```

| File | What it shows | Used in topics |
|---|---|---|
| [`agent.py`](code/agent.py) | minimal agent loop, tool specs, scripted and Anthropic backends | 01, 02, 12 |
| [`agent_react.py`](code/agent_react.py) | ReAct text loop, JSON Schema validation, function-calling dispatch | 03, 04, 15 |
| [`agent_memory.py`](code/agent_memory.py) | four memory types and prompt assembly | 07-11 |
| [`agent_multi.py`](code/agent_multi.py) | planner/worker/critic, reflection, human approval, allowlist, max steps | 05, 06, 13, 14, 16 |

## Rules for good tools
- One clear job per tool, with a name and description the model can choose from.
- A strict input schema. Validate before running.
- Return short, readable results and readable errors (the model reads them and recovers).
- Mark dangerous tools and require confirmation for them.

## Rules for reliable agents
- **Always cap steps** (`max_steps`). An agent without a cap can loop and spend money.
- Log every step so you can replay failures.
- Make actions idempotent where possible (a retry must not double-charge).
- Treat text returned by tools or web pages as data, never as instructions (prompt injection).
- Put guardrails and approvals in code, not just in the prompt.

## Try it
1. Add a `current_time` tool to `agent.py`. Give it a schema and description.
2. In `agent_react.py`, add a `divide` tool that returns a readable error for division by zero. Write a test.
3. Set `max_steps=1` in `run_react` with `DEMO_SCRIPT`. What does it return?
4. Change `ShortTermMemory(max_messages=3)` to `1`. What context does `build_context` produce now?
5. In `agent_multi.py`, add a critic rule "must mention a risk", and a matching worker fix. Does the pipeline still finish within `max_steps`?

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. Agent or fixed workflow: when is each right?**
<details><summary>Answer</summary>
Use a workflow when the steps are known in advance: it is cheaper, more predictable, and easier to test. Use an agent when the steps depend on intermediate results and can't be scripted.
</details>

**2. Who executes a tool call, the model or your code? Why does it matter?**
<details><summary>Answer</summary>
Your code. The model only emits a name and arguments. That boundary is where you validate arguments, enforce permissions, and require approvals.
</details>

**3. What makes a tool easy for a model to use?**
<details><summary>Answer</summary>
A distinct name, a description that says what it does and when to use it, a strict JSON schema with described fields, and short, readable outputs and error messages.
</details>

**4. Describe one ReAct step.**
<details><summary>Answer</summary>
The model writes a Thought, then an Action with an Action Input (or a Final Answer). Your code runs the action and appends an Observation with the result. Repeat until a final answer or the step limit.
</details>

**5. When does reflection actually improve results?**
<details><summary>Answer</summary>
When there is a real feedback signal: failing tests, validation errors, a rubric-based critic, or tool results. Without new information, self-critique helps much less. Always cap revision rounds.
</details>

**6. Name the four memory types in this phase and one use for each.**
<details><summary>Answer</summary>
Short-term (recent turns in the prompt), long-term (searchable notes across sessions), semantic (stable facts like a user's timezone), episodic (past tasks and outcomes, such as lessons from a failed deploy).
</details>

**7. List the stop conditions every agent loop should have.**
<details><summary>Answer</summary>
Normal finish (no tool calls), max steps, a token or cost budget, a timeout, and detection of repeated identical actions.
</details>

**8. Which actions need human approval, and how should the gate be built?**
<details><summary>Answer</summary>
Irreversible or external actions (send, pay, delete, deploy). Build the gate in code around the tool, show the exact arguments, default to "no", and return rejections to the agent.
</details>

**9. How do you evaluate an agent beyond "the final answer looks right"?**
<details><summary>Answer</summary>
Run a fixed task set with mocked tools and record traces. Grade outcome, trajectory (right tools, valid arguments, no forbidden actions), and efficiency (steps, tokens, cost, latency). Repeat runs to measure consistency.
</details>

**10. Name three controls that make agents safe to run.**
<details><summary>Answer</summary>
Any three of: a tool allowlist with least privilege, argument and business-rule validation, max steps / cost limits, human approval for risky actions, treating tool output as untrusted data, sandboxing, and audit logs.
</details>

More practice: [interview prep · agents and MCP](../08-interview-prep/03-agents-and-mcp.md).

**Prev:** [Phase 3 · LLM Concepts](../03-llm-concepts/README.md) · **Next:** [Phase 5 · Frameworks](../05-frameworks/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
