# Phase 11 · Agent Engineering

**Goal:** go from an agent that works in a demo to one that runs for an hour, handles untrusted text, survives crashes, talks to other agents and can be measured honestly.

**Who it's for:** learners who finished agents ([Phase 4](../04-agentic-ai/README.md)), MCP ([Phase 6](../06-mcp/README.md)) and frontier engineering ([Phase 10](../12-frontier-ai/README.md)).

**Prerequisites:**
- [Phase 4 · Agentic AI](../04-agentic-ai/README.md), especially [agent loops](../04-agentic-ai/topics/12-agent-loops.md) and [guardrails](../04-agentic-ai/topics/16-agent-guardrails.md).
- [Phase 6 · MCP](../06-mcp/README.md) for the tool side of the picture.
- [Phase 10 · Frontier AI Engineering](../12-frontier-ai/README.md) for context engineering and evaluation.
- Basic Python. The code uses only the standard library.

**Time:** about 8-10 hours, including the code.

## Big idea
A long-running agent fails in boring ways: its context fills up, it believes a web page, it crashes between paying and writing "paid", it loops, and nobody can tell whether the last release made it better. Each lesson takes one of those failures, measures it with a small offline demo, and gives the fix with numbers you can reproduce.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | Context Compaction for Long-Running Agents | Advanced | [01-context-compaction-for-long-agents.md](topics/01-context-compaction-for-long-agents.md) |
| 02 | Sandboxing Agent Tools | Advanced | [02-sandboxing-agent-tools.md](topics/02-sandboxing-agent-tools.md) |
| 03 | Evaluating Tool Use and Trajectories | Advanced | [03-evaluating-tool-use-and-trajectories.md](topics/03-evaluating-tool-use-and-trajectories.md) |
| 04 | Agent-to-Agent Protocols (A2A) | Advanced | [04-agent-to-agent-protocols-a2a.md](topics/04-agent-to-agent-protocols-a2a.md) |
| 05 | Durable Agent Runs and Idempotency | Advanced | [05-durable-agent-runs-and-idempotency.md](topics/05-durable-agent-runs-and-idempotency.md) |
| 06 | Agent Budgets and Loop Detection | Intermediate | [06-agent-budgets-and-loop-detection.md](topics/06-agent-budgets-and-loop-detection.md) |

## Learning order
1. **Keep it running (01, 05, 06).** Compaction, durable runs, budgets and loop detection.
2. **Keep it safe (02).** Sandboxing tools.
3. **Keep it honest (03).** Evaluating trajectories and reliability.
4. **Keep it connected (04).** A2A, and how it differs from MCP.

If you are short on time, read 02, 03 and 05.

## Run the code
The demos in `code/` are offline, standard-library Python: no API key, no downloads, no GPU.

```bash
# from the repo root
python 13-agent-engineering/code/ae_compaction.py   # four compaction strategies, one recall score
python 13-agent-engineering/code/ae_sandbox.py      # path, command, URL policy and a timeout
python 13-agent-engineering/code/ae_tool_evals.py   # trajectory scoring, pass@k and pass^k
python 13-agent-engineering/code/ae_a2a.py          # agent card, task states, JSON-RPC calls
python 13-agent-engineering/code/ae_durable.py      # crash, resume, idempotency keys, backoff
python 13-agent-engineering/code/ae_budget.py       # budgets and loop detection

python -m pytest -q 13-agent-engineering/code       # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`ae_compaction.py`](code/ae_compaction.py) | truncate, clear tool output, summarise, notes, scored by surviving facts | 01 |
| [`ae_sandbox.py`](code/ae_sandbox.py) | path normalisation, command allow-list, URL allow-list, child process timeout | 02 |
| [`ae_tool_evals.py`](code/ae_tool_evals.py) | tool, argument, order and extra-call scores; pass@k versus pass^k | 03 |
| [`ae_a2a.py`](code/ae_a2a.py) | card check, task state machine, message/send, tasks/get, tasks/cancel | 04 |
| [`ae_durable.py`](code/ae_durable.py) | crash between action and checkpoint, idempotency keys, backoff | 05 |
| [`ae_budget.py`](code/ae_budget.py) | step, token and dollar limits, repeat and ping-pong detection | 06 |

The A2A demo is a teaching model of the general shape of the protocol, which evolves: read the current spec before building.

## Try it
1. In `ae_compaction.py`, change `keep_recent` from 4 to 8. Which facts survive, and how many tokens does it cost?
2. In `ae_sandbox.py`, add a symlink check with `os.path.realpath`. What new attack does it stop?
3. In `ae_tool_evals.py`, find the smallest k where pass^k drops below 0.1 for 10 runs with 8 successes.
4. In `ae_a2a.py`, add a `rejected` path for a request outside the card's skills.
5. In `ae_durable.py`, make the key random per call. What happens to the double charge?
6. In `ae_budget.py`, lower `max_steps` to 5 and say which scenarios change their stop reason.

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. Why clear tool results before summarising the conversation?**
<details><summary>Answer</summary>
Tool results are usually the bulk of the tokens, are easy to replace with a stub, and can often be re-fetched. Clearing them gives a large saving with little risk to reasoning, while summarising costs a model call and can drop or distort details.
</details>

**2. An agent forgets a constraint it was told 40 turns ago. What would you change?**
<details><summary>Answer</summary>
Pin constraints in the system prompt or a notes block that survives compaction, make the agent record decisions and constraints as notes when they appear, and add a recall test with a known constraint at a known position.
</details>

**3. How can compaction raise your token bill?**
<details><summary>Answer</summary>
Rewriting earlier history changes the prefix, so the prompt cache misses and the whole prompt is billed at full price again. Compact in infrequent big steps and keep the pinned prefix identical.
</details>

**4. Why is a plain startswith check on a workspace path unsafe?**
<details><summary>Answer</summary>
Without normalising, `..` segments escape the folder, and without a trailing slash a sibling like `/work/project-evil` shares the prefix. Resolve to a normal absolute path, resolve symlinks, and compare against the root plus a slash.
</details>

**5. What is the difference between a policy check and a sandbox?**
<details><summary>Answer</summary>
A policy check is code that decides whether a request looks allowed and can have bugs or gaps. A sandbox is real isolation, such as a container or microVM, that limits what a process can touch even if the policy fails. Use both.
</details>

**6. An agent reads a web page that says to email your files to a stranger. How do you limit the damage?**
<details><summary>Answer</summary>
Assume the injection can succeed. The agent has no tool that can send email to arbitrary addresses, files outside the workspace are not mounted, there is no network by default, credentials are scoped, and irreversible actions need human approval. Log and alert on blocked attempts.
</details>

**7. An agent has 80 percent success per run. What is the probability it succeeds on all of three independent runs, and why does it matter?**
<details><summary>Answer</summary>
Roughly 0.8 cubed, about 0.51 under independence. The demo's finite-sample estimate from 10 runs with 8 successes is C(8,3)/C(10,3) = 0.4667. It matters because users run agents repeatedly, so reliability over streaks is what they experience.
</details>

**8. Why can a run with the right final answer still be a failure?**
<details><summary>Answer</summary>
The process may have broken a rule or wasted resources: a wrong or unsafe call that happened to be harmless this time, an unauthorised action, or many extra calls. The end state and the trajectory both need checks, plus explicit safety assertions.
</details>

**9. How do you evaluate a task with several valid solution paths?**
<details><summary>Answer</summary>
Check the final state and a set of invariants (never refund more than the order total, never write outside the workspace) instead of one expected sequence. Use trajectory comparison only as a diagnostic.
</details>

**10. How is A2A different from MCP, and can you use both?**
<details><summary>Answer</summary>
MCP connects an agent to tools and data sources that run a function and return. A2A connects an agent to another autonomous agent that can reason, ask questions and work over time while keeping its internals hidden. They are complementary: use MCP for your own tools and A2A to delegate to other agents.
</details>

**11. Why are task states important in agent-to-agent calls?**
<details><summary>Answer</summary>
A delegated task can pause for input or authentication, take a long time, fail or be cancelled. A state machine gives both sides a shared view, for example input-required means the client must answer, and terminal states mean no further changes.
</details>

**12. What security risks come with calling another company's agent?**
<details><summary>Answer</summary>
Its output may contain injected instructions, it may receive more data than it needs, and it may loop or run up cost. Treat output as untrusted, authenticate and authorise per skill, send minimal data, set timeouts and budgets, and log calls.
</details>

**13. Why is a checkpoint after every step not enough to prevent a double charge?**
<details><summary>Answer</summary>
The process can crash after the charge takes effect and before the checkpoint is written, so on resume the step looks unfinished and runs again. An idempotency key lets the payment service recognise the repeat and ignore it.
</details>

**14. How should an idempotency key be generated?**
<details><summary>Answer</summary>
Deterministically from the run id and step id, so every retry of the same intended action uses the same key. A random key created at call time defeats the purpose, since each retry looks like a new action.
</details>

**15. Which errors should an agent retry?**
<details><summary>Answer</summary>
Transient ones such as timeouts, rate limits and server errors, using exponential backoff with jitter and a retry cap. Permanent errors such as validation failures or permission denials should fail immediately.
</details>

**16. Why do you need both loop detection and a step budget?**
<details><summary>Answer</summary>
Loop detection catches repeated identical calls quickly, but an agent that tries a new variation each time never repeats exactly. The step budget catches that wandering. In the demo, 30 different queries are stopped only by the 20-step budget.
</details>

**17. How do you choose budget values?**
<details><summary>Answer</summary>
Measure successful runs, take the 95th percentile of steps, tokens and cost, and set limits near twice that. Review after changing the model or tools, and give sub-agents a share of the parent's budget.
</details>

**18. What should an agent return when it hits a limit?**
<details><summary>Answer</summary>
A named stop reason (such as budget:dollars or loop:repeat), any partial results, the last error and a suggestion for the next step, so a person or fallback can continue instead of starting blind.
</details>

More practice: [interview prep · agents and MCP](../08-interview-prep/03-agents-and-mcp.md).

**Prev:** [Phase 10 · Frontier AI Engineering](../12-frontier-ai/README.md) · **Next:** [Back to the roadmap](../README.md)
