"""A tiny state-machine graph runner (the idea behind LangGraph). Stdlib only.

Nodes are functions: state(dict) -> dict of updates.
Edges are fixed (A -> B) or conditional (A -> router(state) -> name).
"""

END = "__end__"


class MiniGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional = {}
        self.entry = None

    def add_node(self, name, fn):
        if name in self.nodes:
            raise ValueError(f"duplicate node: {name}")
        self.nodes[name] = fn
        return self

    def set_entry(self, name):
        self.entry = name
        return self

    def add_edge(self, src, dst):
        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src, router):
        """router(state) returns the name of the next node (or END)."""
        self.conditional[src] = router
        return self

    def run(self, state, max_steps=50):
        if self.entry is None:
            raise ValueError("no entry node")
        state = dict(state)
        current = self.entry
        trace = []
        for _ in range(max_steps):
            if current == END:
                state["_trace"] = trace
                return state
            if current not in self.nodes:
                raise KeyError(f"unknown node: {current}")
            trace.append(current)
            state.update(self.nodes[current](state) or {})
            if current in self.conditional:
                current = self.conditional[current](state)
            else:
                current = self.edges.get(current, END)
        raise RuntimeError("max_steps exceeded (possible infinite loop)")


def demo():
    """Draft -> review loop: revise until the draft is long enough."""
    g = MiniGraph()
    g.add_node("write", lambda s: {"draft": s.get("draft", "") + "word "})
    g.add_node("review", lambda s: {"ok": len(s["draft"].split()) >= 3})
    g.set_entry("write").add_edge("write", "review")
    g.add_conditional_edge("review", lambda s: END if s["ok"] else "write")
    return g.run({})


if __name__ == "__main__":
    print(demo())
