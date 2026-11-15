import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mcp_session import LIFECYCLE, MISTAKES, init_messages, replay  # noqa: E402


def test_lifecycle_replies_match_the_protocol_shapes():
    log = replay([json.dumps(m) for _, _, m in LIFECYCLE])
    assert log[0]["out"]["result"]["serverInfo"]["name"] == "mini-mcp"
    assert log[1]["out"] is None                      # notification: no reply
    assert [t["name"] for t in log[2]["out"]["result"]["tools"]] == ["add_numbers", "word_count"]
    assert log[3]["out"]["result"]["content"][0]["text"] == "42"
    assert log[4]["out"]["result"]["contents"][0]["text"] == "Hello from a resource."
    assert log[5]["out"]["result"]["messages"][0]["content"]["text"].startswith("Summarize this:")


def test_every_reply_echoes_its_request_id():
    log = replay([json.dumps(m) for _, _, m in LIFECYCLE])
    for (_l, _w, m), e in zip(LIFECYCLE, log):
        if "id" in m:
            assert e["out"]["id"] == m["id"] and e["out"]["jsonrpc"] == "2.0"


def test_each_mistake_gives_its_documented_error():
    codes = {"before_init": -32600, "unknown_method": -32601, "unknown_tool": -32602, "missing_arg": -32602, "bad_json": -32700}
    for key, code in codes.items():
        m = MISTAKES[key]
        raws = ([json.dumps(x) for x in init_messages()] if m["init"] else []) + [m.get("raw") or json.dumps(m["msg"])]
        assert replay(raws)[-1]["out"]["error"]["code"] == code, key


def test_a_failing_tool_is_a_result_not_a_protocol_error():
    m = MISTAKES["tool_fails"]
    raws = [json.dumps(x) for x in init_messages()] + [json.dumps(m["msg"])]
    out = replay(raws)[-1]["out"]
    assert "error" not in out and out["result"]["isError"] is True
