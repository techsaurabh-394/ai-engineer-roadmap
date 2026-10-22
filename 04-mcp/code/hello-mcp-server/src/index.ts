#!/usr/bin/env node
// The smallest useful MCP server. Two tools, validated inputs, stdio transport.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "hello-mcp-server", version: "1.0.0" });

// Tool 1: the model reads the description to decide when to call this.
server.registerTool(
  "add_numbers",
  {
    title: "Add numbers",
    description: "Add two numbers and return the sum.",
    inputSchema: { a: z.number(), b: z.number() },
    annotations: { readOnlyHint: true },
  },
  async ({ a, b }) => ({ content: [{ type: "text", text: String(a + b) }] })
);

// Tool 2: input is validated by zod before this function runs.
server.registerTool(
  "word_count",
  {
    title: "Word count",
    description: "Count the words in a piece of text.",
    inputSchema: { text: z.string().min(1) },
    annotations: { readOnlyHint: true },
  },
  async ({ text }) => ({ content: [{ type: "text", text: String(text.trim().split(/\s+/).length) }] })
);

await server.connect(new StdioServerTransport());
// stdout is the protocol channel, so log to stderr only.
console.error("hello-mcp-server running on stdio");
