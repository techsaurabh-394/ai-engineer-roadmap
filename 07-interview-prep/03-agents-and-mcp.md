# 03 · AI Agents & MCP

### 1. What is an AI agent?
A system where an LLM decides what to do next in a loop: observe the state, choose an action (often a tool call), see the result, and continue until the goal is met or a limit is hit. The difference from a fixed pipeline is that the model, not your code, chooses the steps.

### 2. When should you use an agent instead of a fixed workflow?
Use a workflow when the steps are known and stable: it's cheaper, faster and easier to test. Use an agent when the path depends on intermediate results and can't be enumerated up front. Start with the simplest design that works and add autonomy only where it earns its cost.

### 3. What makes a good tool for an agent?
A clear name and description (the model chooses tools from these), a narrow single purpose, a strict input schema, compact outputs with only needed fields, readable error messages the model can act on, and safe defaults. Mark destructive tools clearly and require explicit parameters.

### 4. What is the Model Context Protocol (MCP)?
An open protocol that standardises how AI applications connect to external tools and data. An MCP server exposes capabilities (tools, resources, prompts) and an MCP client inside the host application (such as Claude Desktop) discovers and calls them. It replaces one-off integrations with a common interface.

### 5. What are MCP's main primitives and transports?
Primitives: tools (actions the model can invoke), resources (data the application can read) and prompts (reusable templates). Common transports: stdio for local servers launched by the client, and HTTP-based transports for remote servers. Check the current spec for transport details, since they have evolved.

### 6. MCP vs plain function calling vs a REST API?
Function calling is how a single model request invokes your functions; the integration lives in your app. A REST API is an interface for software in general. MCP standardises the connection between any compatible client and any compatible server, so one server works across many AI clients without custom glue.

### 7. What are the security risks with MCP and tool-using agents?
Prompt injection through tool results or fetched content, over-privileged credentials, destructive tools without confirmation, data exfiltration through tool arguments, and untrusted third-party servers. Mitigate with least privilege, per-tool permissions, human confirmation for risky actions, input validation, audit logs and vetting servers before installing.

### 8. What multi-agent patterns exist?
Orchestrator-worker (a lead agent delegates subtasks), handoff (one agent passes control to another specialist), and parallel agents exploring independently then merging. Multi-agent designs add cost, latency and failure modes; use them when subtasks are genuinely separable, not by default.

### 9. How do agents get memory?
Short-term memory is the conversation and scratch state in the context window. Long-term memory is external storage (database or vector store) that the agent reads and writes through tools. Summarise or prune context to stay within limits, and decide explicitly what is worth remembering.

### 10. How do you make agents reliable?
Cap steps and cost, set timeouts, retry transient failures with backoff, make tool calls idempotent where possible, validate every tool input and output, log full traces, add human approval for high-impact actions, and test with scripted scenarios so behaviour changes are caught.

### 11. How do you evaluate an agent?
Measure task success on a fixed set of scenarios, plus process quality: correct tool choices, number of steps, cost, latency and recovery from errors. Inspect trajectories, not only final answers, because an agent can succeed for the wrong reasons.

### 12. What is agent-to-agent communication?
Agents built by different people or on different platforms exchanging messages directly. Open efforts such as A2A (Agent2Agent) aim to standardise discovery and messaging, while MCP focuses on connecting agents to tools and data. Key problems are identity, trust and abuse prevention: how an agent knows who it's talking to and whether to act on a message.
