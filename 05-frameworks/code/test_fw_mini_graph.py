import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from fw_mini_graph import END, MiniGraph, demo


def test_linear_flow():
    g = MiniGraph()
    g.add_node("a", lambda s: {"x": 1}).add_node("b", lambda s: {"y": s["x"] + 1})
    g.set_entry("a").add_edge("a", "b")
    out = g.run({})
    assert out["y"] == 2 and out["_trace"] == ["a", "b"]


def test_conditional_loop():
    out = demo()
    assert out["draft"].split() == ["word"] * 3
    assert out["_trace"].count("write") == 3


def test_conditional_branch():
    g = MiniGraph()
    g.add_node("start", lambda s: {}).add_node("big", lambda s: {"r": "big"}).add_node("small", lambda s: {"r": "small"})
    g.set_entry("start")
    g.add_conditional_edge("start", lambda s: "big" if s["n"] > 5 else "small")
    assert g.run({"n": 9})["r"] == "big"
    assert g.run({"n": 1})["r"] == "small"


def test_infinite_loop_guard():
    g = MiniGraph()
    g.add_node("a", lambda s: {}).set_entry("a").add_edge("a", "a")
    with pytest.raises(RuntimeError):
        g.run({}, max_steps=5)


def test_unknown_node_and_no_entry():
    g = MiniGraph()
    with pytest.raises(ValueError):
        g.run({})
    g.add_node("a", lambda s: {}).set_entry("a").add_edge("a", "ghost")
    with pytest.raises(KeyError):
        g.run({})


def test_input_not_mutated():
    g = MiniGraph()
    g.add_node("a", lambda s: {"k": 1}).set_entry("a").add_edge("a", END)
    inp = {}
    g.run(inp)
    assert inp == {}
