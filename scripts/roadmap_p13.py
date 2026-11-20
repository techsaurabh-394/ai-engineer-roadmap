"""Roadmap text for Phase 11 (Agent engineering, folder 13-agent-engineering); imported by roadmap_data.py as PHASES["13"]."""

PHASE = dict(
    tagline="Make agents that run long, stay safe and can be trusted",
    why="A demo agent finishes in five steps. A real one runs for an hour, reads untrusted text, touches money and talks to other agents. "
        "This phase covers what keeps it alive and honest: compacting context, sandboxing tools, scoring trajectories, delegating to other agents, "
        "surviving crashes without repeating side effects, and stopping runaway runs.",
    outcomes=["Compare compaction strategies by how many needed facts survive a token budget",
              "Write path, command and network checks, and explain why they are not a sandbox by themselves",
              "Score a tool-call trajectory and report reliability with pass^k instead of pass@k"],
    ideas=["context compaction", "agent notes", "sandboxing and least privilege", "trajectory evals", "pass^k",
           "A2A and MCP", "checkpoints and idempotency keys", "budgets and loop detection"],
    builds=["ae_compaction.py: four compaction strategies, one recall score", "ae_sandbox.py: path, command, URL policy and a timeout",
            "ae_tool_evals.py: trajectory scorer, pass@k and pass^k", "ae_a2a.py: agent card, task states, JSON-RPC calls",
            "ae_durable.py: crash, resume, idempotency keys, backoff", "ae_budget.py: step, token and dollar budgets, loop rules"],
    demos=["13-agent-engineering/code/ae_compaction.py", "13-agent-engineering/code/ae_sandbox.py", "13-agent-engineering/code/ae_tool_evals.py",
           "13-agent-engineering/code/ae_a2a.py", "13-agent-engineering/code/ae_durable.py", "13-agent-engineering/code/ae_budget.py"],
    labs=[],
    needs="Phase 4 (agents), Phase 6 (MCP) and Phase 10 (context engineering and evals). Basic Python. No API key or GPU is needed.",
)
