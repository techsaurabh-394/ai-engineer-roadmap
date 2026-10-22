import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from mcp_mini_jsonrpc import MiniMcpClient, MiniMcpServer, RpcError


@pytest.fixture
def client():
    c = MiniMcpClient(MiniMcpServer())
    c.connect()
    return c


def test_initialize_shape():
    server = MiniMcpServer()
    reply = json.loads(server.handle(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})))
    assert reply["jsonrpc"] == "2.0" and reply["id"] == 1
    r = reply["result"]
    assert "protocolVersion" in r and "tools" in r["capabilities"]
    assert r["serverInfo"]["name"] == "mini-mcp"


def test_requires_initialized_notification():
    server = MiniMcpServer()
    reply = json.loads(server.handle(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})))
    assert reply["error"]["code"] == -32600


def test_notification_gets_no_response():
    assert MiniMcpServer().handle(json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"})) is None


def test_tools(client):
    names = [t["name"] for t in client.list_tools()]
    assert names == ["add_numbers", "word_count"]
    res = client.call_tool("add_numbers", {"a": 2, "b": 40})
    assert res["content"][0]["text"] == "42" and res["isError"] is False
    assert client.call_tool("word_count", {"text": "learn MCP step by step"})["content"][0]["text"] == "5"


def test_tool_errors(client):
    with pytest.raises(RpcError) as e:
        client.call_tool("nope", {})
    assert e.value.code == -32602
    with pytest.raises(RpcError):
        client.call_tool("add_numbers", {"a": 1})
    res = client.call_tool("add_numbers", {"a": 1, "b": "x"})  # tool-level failure
    assert res["isError"] is True


def test_resources(client):
    assert client.list_resources()[0]["uri"] == "notes://welcome"
    assert client.read_resource("notes://welcome")[0]["text"].startswith("Hello")
    with pytest.raises(RpcError) as e:
        client.read_resource("notes://missing")
    assert e.value.code == -32002


def test_prompts(client):
    assert client.list_prompts()[0]["name"] == "summarize"
    got = client.get_prompt("summarize", {"text": "abc"})
    assert got["messages"][0]["role"] == "user"
    assert "abc" in got["messages"][0]["content"]["text"]


def test_bad_json_and_unknown_method(client):
    assert json.loads(client.server.handle("{nope"))["error"]["code"] == -32700
    with pytest.raises(RpcError) as e:
        client.request("does/not/exist")
    assert e.value.code == -32601
