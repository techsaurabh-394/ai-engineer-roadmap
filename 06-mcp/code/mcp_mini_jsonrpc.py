"""A tiny in-process MCP-style JSON-RPC 2.0 server and client (stdlib only).

It mimics the *message shapes* of the Model Context Protocol:
initialize, notifications/initialized, tools/list, tools/call,
resources/list, resources/read, prompts/list, prompts/get.
It is a teaching toy, not the real SDK. The real transport (stdio or
Streamable HTTP) is replaced by a direct function call.
"""
import json

PROTOCOL_VERSION = "2025-06-18"  # check the current spec for the latest version

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class MiniMcpServer:
    def __init__(self):
        self.initialized = False
        self.tools = {
            "add_numbers": {
                "description": "Add two numbers and return the sum.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
                "fn": lambda a: str(a["a"] + a["b"]),
            },
            "word_count": {
                "description": "Count the words in a piece of text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "minLength": 1}},
                    "required": ["text"],
                },
                "fn": lambda a: str(len(a["text"].split())),
            },
        }
        self.resources = {"notes://welcome": ("Welcome", "Hello from a resource.")}
        self.prompts = {
            "summarize": {
                "description": "Ask the model to summarize some text.",
                "arguments": [{"name": "text", "required": True}],
            }
        }

    # ---- transport entry point: one JSON string in, one JSON string (or None) out
    def handle(self, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return self._error(None, PARSE_ERROR, "Parse error")
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return self._error(msg.get("id") if isinstance(msg, dict) else None,
                               INVALID_REQUEST, "Invalid Request")
        msg_id = msg.get("id")
        is_notification = "id" not in msg
        try:
            result = self._dispatch(msg["method"], msg.get("params") or {})
        except RpcError as e:
            return None if is_notification else self._error(msg_id, e.code, e.message)
        if is_notification:
            return None  # notifications never get a response
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _error(self, msg_id, code, message):
        return json.dumps({"jsonrpc": "2.0", "id": msg_id,
                           "error": {"code": code, "message": message}})

    def _dispatch(self, method, params):
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False},
                                 "resources": {}, "prompts": {}},
                "serverInfo": {"name": "mini-mcp", "version": "0.1.0"},
            }
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method == "ping":
            return {}
        if not self.initialized:
            raise RpcError(INVALID_REQUEST, "Server not initialized")
        if method == "tools/list":
            return {"tools": [{"name": n, "description": t["description"],
                               "inputSchema": t["inputSchema"]}
                              for n, t in self.tools.items()]}
        if method == "tools/call":
            return self._call_tool(params)
        if method == "resources/list":
            return {"resources": [{"uri": u, "name": n, "mimeType": "text/plain"}
                                  for u, (n, _) in self.resources.items()]}
        if method == "resources/read":
            uri = params.get("uri")
            if uri not in self.resources:
                raise RpcError(-32002, "Resource not found")
            return {"contents": [{"uri": uri, "mimeType": "text/plain",
                                  "text": self.resources[uri][1]}]}
        if method == "prompts/list":
            return {"prompts": [{"name": n, **p} for n, p in self.prompts.items()]}
        if method == "prompts/get":
            return self._get_prompt(params)
        raise RpcError(METHOD_NOT_FOUND, "Method not found: " + str(method))

    def _call_tool(self, params):
        tool = self.tools.get(params.get("name"))
        if tool is None:
            raise RpcError(INVALID_PARAMS, "Unknown tool: " + str(params.get("name")))
        args = params.get("arguments") or {}
        for key in tool["inputSchema"]["required"]:
            if key not in args:
                raise RpcError(INVALID_PARAMS, "Missing argument: " + key)
        try:
            text = tool["fn"](args)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as e:  # tool failures are results, not protocol errors
            return {"content": [{"type": "text", "text": "Tool failed: %s" % e}],
                    "isError": True}

    def _get_prompt(self, params):
        name = params.get("name")
        if name not in self.prompts:
            raise RpcError(INVALID_PARAMS, "Unknown prompt: " + str(name))
        text = (params.get("arguments") or {}).get("text")
        if text is None:
            raise RpcError(INVALID_PARAMS, "Missing argument: text")
        return {"description": self.prompts[name]["description"],
                "messages": [{"role": "user",
                              "content": {"type": "text",
                                          "text": "Summarize this:\n" + text}}]}


class MiniMcpClient:
    def __init__(self, server):
        self.server = server
        self._next_id = 1

    def request(self, method, params=None):
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._next_id += 1
        reply = json.loads(self.server.handle(json.dumps(msg)))
        if "error" in reply:
            raise RpcError(reply["error"]["code"], reply["error"]["message"])
        return reply["result"]

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.server.handle(json.dumps(msg))

    def connect(self):
        """The handshake: initialize request, then initialized notification."""
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mini-client", "version": "0.1.0"},
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call_tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def list_resources(self):
        return self.request("resources/list")["resources"]

    def read_resource(self, uri):
        return self.request("resources/read", {"uri": uri})["contents"]

    def list_prompts(self):
        return self.request("prompts/list")["prompts"]

    def get_prompt(self, name, arguments):
        return self.request("prompts/get", {"name": name, "arguments": arguments})


if __name__ == "__main__":
    client = MiniMcpClient(MiniMcpServer())
    info = client.connect()
    print("server:", info["serverInfo"], "capabilities:", list(info["capabilities"]))
    print("tools:", [t["name"] for t in client.list_tools()])
    print("add:", client.call_tool("add_numbers", {"a": 2, "b": 40})["content"][0]["text"])
    print("resource:", client.read_resource("notes://welcome")[0]["text"])
    print("prompt:", client.get_prompt("summarize", {"text": "MCP is a protocol."})["messages"][0])
