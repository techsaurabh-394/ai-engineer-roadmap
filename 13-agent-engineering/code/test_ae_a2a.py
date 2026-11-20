import pytest

import ae_a2a as m


def test_card_validation():
    assert m.validate_card(m.CARD) == []
    bad = m.validate_card({"name": "x", "url": "http://x", "skills": []})
    assert "no skills" in bad and "url must be https" in bad and "missing description" in bad


def test_transitions():
    assert m.can_move("submitted", "working")
    assert not m.can_move("completed", "working")
    assert not m.can_move("submitted", "completed")


def test_task_rejects_illegal_move():
    t = m.Task("t", "hi")
    with pytest.raises(ValueError):
        t.move("completed")
    with pytest.raises(ValueError):
        t.move("nonsense")


def test_input_required_then_completed():
    a = m.RemoteAgent(m.CARD)
    r1 = m.send(a, 1, "Book me a flight")
    assert r1["result"]["status"]["state"] == "input-required"
    r2 = m.send(a, 2, "on 12 Dec", task_id="t1")
    assert r2["result"]["status"]["state"] == "completed"
    got = a.handle({"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": "t1"}})
    assert got["result"]["artifacts"][0]["name"] == "offer"


def test_cancel_finished_task_errors():
    a = m.RemoteAgent(m.CARD)
    m.send(a, 1, "flight on 12 Dec")
    r = a.handle({"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": "t1"}})
    assert r["error"]["code"] == -32002


def test_json_rpc_errors():
    a = m.RemoteAgent(m.CARD)
    assert a.handle({"id": 1, "method": "x"})["error"]["code"] == -32600
    assert a.handle({"jsonrpc": "2.0", "id": 1, "method": "x/y"})["error"]["code"] == -32601
    assert a.handle({"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "nope"}})["error"]["code"] == -32001
