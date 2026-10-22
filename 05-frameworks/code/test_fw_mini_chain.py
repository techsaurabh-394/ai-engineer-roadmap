import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fw_mini_chain import Step, demo, parser, prompt_template


def test_demo():
    assert demo() == "APPLE"


def test_pipe_order():
    chain = Step(lambda x: x + 1) | (lambda x: x * 10)
    assert chain.invoke(1) == 20


def test_plain_function_on_left():
    chain = (lambda x: x + "a") | Step(lambda x: x + "b")
    assert chain.invoke("") == "ab"


def test_batch():
    assert (Step(str.upper)).batch(["a", "b"]) == ["A", "B"]


def test_prompt_and_parser():
    assert prompt_template("hi {n}").invoke({"n": "x"}) == "hi x"
    assert parser("ANSWER: foo ") == "foo"


def test_name_tracks_steps():
    assert (Step(len, "a") | Step(str, "b")).name == "a|b"
