"""Agent-to-agent (A2A) basics: an agent card, a task state machine and a JSON-RPC style call,
simulated in one process. Field and method names follow the general shape of the public
A2A protocol but versions differ: treat this as a teaching model and read the current spec.

Run:  python ae_a2a.py
"""
import json

STATES = {"submitted", "working", "input-required", "auth-required", "completed", "failed", "canceled", "rejected"}
TERMINAL = {"completed", "failed", "canceled", "rejected"}
ALLOWED = {
    "submitted": {"working", "rejected", "canceled"},
    "working": {"input-required", "auth-required", "completed", "failed", "canceled"},
    "input-required": {"working", "canceled", "failed"},
    "auth-required": {"working", "canceled", "failed"},
}


def validate_card(card):
    """Return a list of problems (empty means OK)."""
    problems = []
    for key in ("name", "description", "url", "version", "skills"):
        if key not in card:
            problems.append(f"missing {key}")
    if card.get("skills") == []:
        problems.append("no skills")
    for s in card.get("skills", []):
        for key in ("id", "name", "description"):
            if key not in s:
                problems.append(f"skill missing {key}")
    if not str(card.get("url", "")).startswith("https://"):
        problems.append("url must be https")
    return problems


def can_move(old, new):
    return new in ALLOWED.get(old, set())


class Task:
    def __init__(self, task_id, text):
        self.id = task_id
        self.state = "submitted"
        self.history = [("user", text)]
        self.artifacts = []

    def move(self, new):
        if new not in STATES:
            raise ValueError(f"unknown state {new}")
        if not can_move(self.state, new):
            raise ValueError(f"illegal move {self.state} -> {new}")
        self.state = new


class RemoteAgent:
    """A pretend remote agent that sells flights. It asks for a date if the message has none."""

    def __init__(self, card):
        self.card = card
        self.tasks = {}
        self.counter = 0

    def handle(self, request):
        """Handle one JSON-RPC style request dict, return a response dict."""
        method, rid, params = request.get("method"), request.get("id"), request.get("params", {})
        if request.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "invalid request"}}
        if method == "message/send":
            text = params["message"]["parts"][0]["text"]
            tid = params.get("taskId")
            if tid is None:
                self.counter += 1
                tid = f"t{self.counter}"
                self.tasks[tid] = Task(tid, text)
                self.tasks[tid].move("working")
            else:
                self.tasks[tid].history.append(("user", text))
                self.tasks[tid].move("working")
            t = self.tasks[tid]
            if "on 12 dec" not in text.lower() and not any("12 dec" in h[1].lower() for h in t.history):
                t.move("input-required")
                t.history.append(("agent", "Which date?"))
            else:
                t.artifacts.append({"name": "offer", "text": "LHR to JFK, 12 Dec, 420 GBP"})
                t.move("completed")
            return {"jsonrpc": "2.0", "id": rid, "result": {"id": t.id, "status": {"state": t.state}}}
        if method == "tasks/get":
            t = self.tasks.get(params["id"])
            if t is None:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "task not found"}}
            return {"jsonrpc": "2.0", "id": rid, "result": {"id": t.id, "status": {"state": t.state}, "artifacts": t.artifacts}}
        if method == "tasks/cancel":
            t = self.tasks.get(params["id"])
            if t is None:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "task not found"}}
            try:
                t.move("canceled")
            except ValueError as e:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32002, "message": str(e)}}
            return {"jsonrpc": "2.0", "id": rid, "result": {"id": t.id, "status": {"state": t.state}}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}}


def send(agent, rid, text, task_id=None):
    params = {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}}
    if task_id:
        params["taskId"] = task_id
    return agent.handle({"jsonrpc": "2.0", "id": rid, "method": "message/send", "params": params})


CARD = {"name": "FlightDesk", "description": "Finds and prices flights.", "url": "https://flights.example.com/a2a",
        "version": "1.0.0", "skills": [{"id": "search", "name": "Flight search", "description": "Find fares."}],
        "capabilities": {"streaming": False}}

if __name__ == "__main__":
    print("card problems:", validate_card(CARD))
    print("bad card     :", validate_card({"name": "x", "url": "http://x", "skills": []}))
    a = RemoteAgent(CARD)
    r1 = send(a, 1, "Book me a flight London to New York")
    print("1:", json.dumps(r1["result"]))
    r2 = send(a, 2, "on 12 Dec", task_id="t1")
    print("2:", json.dumps(r2["result"]))
    print("3:", json.dumps(a.handle({"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": "t1"}})["result"]))
    print("4 cancel finished task:", a.handle({"jsonrpc": "2.0", "id": 4, "method": "tasks/cancel", "params": {"id": "t1"}})["error"])
    print("5 unknown method:", a.handle({"jsonrpc": "2.0", "id": 5, "method": "x/y"})["error"])
    print("legal moves from working:", sorted(ALLOWED["working"]))
