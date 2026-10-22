# 03 · AI agents

**Goal:** build an agent from scratch so you understand what frameworks do for you. Time: 2 hours.

## Big idea
An **agent** is an LLM in a loop. At each step the model decides what to do next, usually by calling a **tool** (a function you wrote). You run the tool, hand back the result, and the model decides again, until it's done.

![The agent loop](../assets/agent-loop.svg)

The model never runs anything itself. It *asks*; your code *acts*. That is where you put safety.

## Run it
```bash
cd 03-agents/code
python agent.py                  # offline demo with a scripted model
pytest -q                        # tests for the loop, step cap and error handling

# With real Claude tool use:
pip install anthropic
ANTHROPIC_API_KEY=... python agent.py "What is 12*7+5? Then save it as a note."
```

Read `agent.py` top to bottom. The heart is `run_agent()`: about 15 lines.

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

## Try it
1. Add a `current_time` tool. Give it a schema and description.
2. Make `calculator` reject expressions longer than 50 characters.
3. Set `max_steps=1` and see what the agent returns.
4. Write a test where the model calls an unknown tool. Is the error readable?

## Checkpoint
- Agent or fixed workflow: when is each right?
- What makes a tool easy for a model to use?
- Name three controls that make agents safe to run.

(Answers: [interview prep, agents](../07-interview-prep/03-agents-and-mcp.md))

**Next:** [04 · MCP](../04-mcp/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
