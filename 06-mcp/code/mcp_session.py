"""Replay a conversation against MiniMcpServer and return the wire log (standard library only).

The site's "MCP lab" shows each JSON-RPC message that crosses between a client and a server. This module is the
reference: it feeds raw JSON strings to the real MiniMcpServer in mcp_mini_jsonrpc.py and records the replies.
A JavaScript port of the server runs in the browser; tests compare its replies with these.

Run:  python mcp_session.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mcp_mini_jsonrpc import PROTOCOL_VERSION, MiniMcpServer  # noqa: E402

# The guided lifecycle: (label, why, message). Messages without an "id" are notifications and get no reply.
LIFECYCLE = [
    ("initialize", "The client says hello and states which protocol version it speaks.",
     {"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "mini-client", "version": "0.1.0"}}}),
    ("initialized", "The client confirms. This is a notification: it has no id, so the server sends no reply.",
     {"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ("tools/list", "The client asks what tools exist. This is tool discovery: names, descriptions and JSON Schemas.",
     {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ("tools/call", "The client runs a tool with arguments that match its schema.",
     {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "add_numbers", "arguments": {"a": 2, "b": 40}}}),
    ("resources/read", "The client reads a resource: data the server exposes by URI.",
     {"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "notes://welcome"}}),
    ("prompts/get", "The client fetches a ready-made prompt template and fills it in.",
     {"jsonrpc": "2.0", "id": 5, "method": "prompts/get", "params": {"name": "summarize", "arguments": {"text": "MCP is a protocol."}}}),
]

# Things that go wrong, each replayed on a fresh, already-initialized server unless "init" is False.
MISTAKES = {
    "before_init": dict(title="Call a tool before the handshake", init=False,
                        msg={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    "unknown_method": dict(title="Call a method that does not exist", init=True,
                           msg={"jsonrpc": "2.0", "id": 9, "method": "tools/destroy"}),
    "unknown_tool": dict(title="Call a tool that does not exist", init=True,
                         msg={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}),
    "missing_arg": dict(title="Leave out a required argument", init=True,
                        msg={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "add_numbers", "arguments": {"a": 1}}}),
    "tool_fails": dict(title="The tool runs but fails inside", init=True,
                       msg={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "add_numbers", "arguments": {"a": 1, "b": "x"}}}),
    "bad_json": dict(title="Send text that is not JSON", init=True, raw="{not json"),
}


def init_messages():
    return [LIFECYCLE[0][2], LIFECYCLE[1][2]]


def replay(raws):
    """Feed raw JSON strings to a fresh server. Returns [{"in": raw, "out": parsed reply or None}]."""
    server = MiniMcpServer()
    log = []
    for raw in raws:
        reply = server.handle(raw)
        log.append({"in": raw, "out": None if reply is None else json.loads(reply)})
    return log


if __name__ == "__main__":
    log = replay([json.dumps(m) for _, _, m in LIFECYCLE])
    for (label, _why, _m), entry in zip(LIFECYCLE, log):
        out = entry["out"]
        shown = "(no reply: notification)" if out is None else json.dumps(out)[:100]
        print(f"{label:15} -> {shown}")
    for key, m in MISTAKES.items():
        raws = (([json.dumps(x) for x in init_messages()]) if m["init"] else []) + [m.get("raw") or json.dumps(m["msg"])]
        print(f"{key:15} -> {json.dumps(replay(raws)[-1]['out'])}")
