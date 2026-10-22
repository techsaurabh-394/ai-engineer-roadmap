# 04 · MCP (Model Context Protocol)

**Goal:** build an MCP server and connect it to Claude Desktop. Time: 2 hours.

## Big idea
Before MCP, every AI app needed custom code for every tool. **MCP** is an open standard: you write a **server** that exposes **tools** (actions), **resources** (data) and **prompts** (templates). Any MCP-compatible **client** can then use it.

![MCP architecture](../assets/mcp-architecture.svg)

Servers usually run locally over **stdio** (the app launches them) or remotely over HTTP. Messages are JSON-RPC.

## Build it
The smallest useful server is in [`code/hello-mcp-server`](code/hello-mcp-server/src/index.ts): two tools, validated with `zod`, about 40 lines.

```bash
cd 04-mcp/code/hello-mcp-server
npm install
npm run build
```

### Connect it to Claude Desktop
Add this to `claude_desktop_config.json` (use your absolute path), then restart Claude Desktop:
```json
{
  "mcpServers": {
    "hello": {
      "command": "node",
      "args": ["/absolute/path/to/04-mcp/code/hello-mcp-server/dist/index.js"]
    }
  }
}
```
Ask: "Use the hello server to add 2 and 40", then "How many words are in: learn MCP step by step?"

## Try it
1. Add a third tool, `reverse_text`. Give it a clear description.
2. Send invalid input (an empty string to `word_count`). What does the client see?
3. Why does the server log with `console.error` and not `console.log`? (Hint: stdout *is* the protocol.)

## Security checklist
- Give each server the least access it needs.
- Mark read-only and destructive tools (`annotations`).
- Never run shell or SQL built from model output.
- Treat tool results as untrusted text (prompt injection).
- Only install servers you've read.

## Level up
A fuller server with persistence, search, tests and a Claude Desktop config is the [mcp-notes-server](https://github.com/techsaurabh-394/mcp-notes-server) repo.

## Checkpoint
- What are tools, resources and prompts?
- How is MCP different from plain function calling?
- What are the biggest security risks?

(Answers: [interview prep, MCP](../07-interview-prep/03-agents-and-mcp.md))

**Next:** [05 · Evaluation](../05-evaluation/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
