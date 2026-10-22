"""A tiny pipe-style chain (the idea behind LangChain's LCEL `a | b | c`). Stdlib only."""


class Step:
    """Wraps a function so steps can be joined with the | operator."""

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "step")

    def invoke(self, value):
        return self.fn(value)

    def __or__(self, other):
        other = other if isinstance(other, Step) else Step(other)
        return Step(lambda v: other.invoke(self.invoke(v)), f"{self.name}|{other.name}")

    def __ror__(self, other):  # allows plain_function | Step
        return Step(other) | self

    def batch(self, values):
        return [self.invoke(v) for v in values]


def prompt_template(template):
    return Step(lambda d: template.format(**d), "prompt")


def fake_model(text):
    """Offline stand-in for an LLM: echoes the last word, upper-cased."""
    return "ANSWER: " + text.split()[-1].upper()


def parser(text):
    return text.removeprefix("ANSWER: ").strip()


def demo():
    chain = prompt_template("Name a fruit starting with {letter} please apple") | Step(fake_model, "model") | Step(parser, "parser")
    return chain.invoke({"letter": "a"})


if __name__ == "__main__":
    print(demo())
