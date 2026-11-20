"""Animation content for Phase 11 (agent engineering)."""
from anim_core import P, L, B  # noqa: F401

_T = "13-agent-engineering/topics/"

P("ae-context-compaction", "Context compaction", "Shrink the history, choose what you lose.", """
Grow|15,300 tokens|1. A 30-message run outgrows a 6,000-token budget|Tool results are 88 percent of it.
Pin|task + rules|2. Pin the system prompt and the task|These must never be dropped.
Clear|old tool output|3. Replace old tool results with short stubs|Keeps the chat, loses facts that lived only there.
Notes|agent writes|4. The agent notes facts as it finds them|A pinned note survives every compaction.
Summary|last resort|5. Summarise what is left, keep recent turns exact|A summary keeps what it judged important.
Score|facts kept|6. Test recall: truncate 1 of 3, notes 3 of 3|Compact in big steps to protect the cache.
""", color="teal", topic=_T + "01-context-compaction-for-long-agents")

P("ae-sandboxing-tools", "Sandboxing tools", "Policy first, a real wall underneath.", """
Assume|fooled model|1. Assume untrusted text can trick the agent|So make the tools safe, not only the prompt.
Path|stay inside|2. Normalise the path, then check the root plus a slash|../../etc/passwd resolves to /etc/passwd: blocked.
Command|allow-list|3. Allow few programs, scan the raw text for operators|cat a.txt; rm -rf / is blocked.
Network|exact hosts|4. https and exact host names only|api.example.com.evil.io is blocked.
Wall|container or VM|5. Real isolation: no network, no secrets, limits|Policy has bugs, a wall holds anyway.
Test|attack it|6. Run a suite of hostile requests on every change|Each must fail safely.
""", color="pink", topic=_T + "02-sandboxing-agent-tools")

P("ae-tool-use-evals", "Tool-use evals", "Score the drive, not just the arrival.", """
End state|did it work|1. Check the world: refund once, file changed|The hardest check to fake.
Trace|calls + args|2. Compare tool calls to the expected ones|Refund 400 instead of 40 fails despite right tools.
Extra|wasted calls|3. Count extra calls as a cost|Two retries passed but cost more.
Repeat|n runs|4. Run each task several times|Agents are not deterministic.
pass^k|all k succeed|5. 80 percent per run is 46.7 percent at k = 3|pass@3 would say 1.0, a misleading number.
Lock|test set|6. Keep a locked set, add production failures|Add safety checks: no action without approval.
""", color="amber", topic=_T + "03-evaluating-tool-use-and-trajectories")

P("ae-a2a-protocol", "Agent-to-agent (A2A)", "Delegate a task to an opaque agent.", """
Card|who and what|1. Read the agent card: skills, URL, version|A bad card is missing skills or uses http.
Send|message/send|2. Send a request, get back a task id|The remote agent starts working.
Ask|input-required|3. The agent can pause and ask a question|Reply on the same task.
Done|completed|4. A terminal state holds the artifacts|Artifacts are the outputs, such as a quote.
Rules|state machine|5. Illegal moves are errors|Cancelling a completed task returns -32002.
Trust|untrusted|6. Authenticate, scope data, budget, log|Remote output can contain injection.
""", color="blue", topic=_T + "04-agent-to-agent-protocols-a2a")

P("ae-durable-runs", "Durable runs", "Survive a crash without a double charge.", """
Steps|4 in a run|1. Read, email, charge card, update the CRM|Each real action happens outside your process.
Crash|after charge|2. The process dies after the charge, before the save|The checkpoint does not know it happened.
Resume|checkpoint only|3. Resume from the saved step: charge repeats|The customer is charged 2x.
Key|run:step:name|4. Send a deterministic idempotency key|The repeat returns duplicate-ignored.
Both|1x each|5. Checkpoint saves effort, the key prevents damage|4 effects in total, each once.
Retry|backoff|6. Retry transient errors: 1, 2, 4, 8, 16, 30 s|Stop at once on bad input.
""", color="teal", topic=_T + "05-durable-agent-runs-and-idempotency")

P("ae-budgets-loops", "Budgets and loops", "Stop a stuck agent, and say why.", """
Limit|3 budgets|1. Count all three in code around every call|Never ask the model to watch its own budget.
Repeat|same call 3x|2. Same tool and arguments three times: loop|Sort argument keys before comparing.
Ping-pong|A B A B|3. Alternating two calls is also a loop|Here the repeat rule fired first, at call 5.
Wander|new each time|4. Endless variety never repeats exactly|Only the step budget stops it at 20.
Reason|named stop|5. Return a stop reason and partial work|budget:tokens, loop:repeat, finished.
Tune|2x p95|6. Set limits near twice the 95th percentile|Track stop reasons on a dashboard.
""", color="pink", topic=_T + "06-agent-budgets-and-loop-detection")
