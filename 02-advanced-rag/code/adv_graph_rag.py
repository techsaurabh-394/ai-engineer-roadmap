"""A tiny knowledge-graph RAG: a triple store plus multi-hop retrieval.

A triple is (subject, relation, object), e.g. ("Alice", "works_at", "Acme").
Vector search finds passages that LOOK like the question. A graph lets you
follow links: "Who manages the team that owns the billing service?" needs
billing -> owned_by -> team -> managed_by -> person. That is two hops.

Run:  python adv_graph_rag.py
"""
from collections import defaultdict, deque

TRIPLES = [
    ("billing-service", "owned_by", "Payments Team"),
    ("Payments Team", "managed_by", "Priya"),
    ("Priya", "reports_to", "Marco"),
    ("search-service", "owned_by", "Discovery Team"),
    ("Discovery Team", "managed_by", "Lena"),
    ("billing-service", "depends_on", "postgres-main"),
    ("postgres-main", "hosted_in", "eu-west-1"),
]


class TripleStore:
    def __init__(self, triples=()):
        self.out = defaultdict(list)  # subject -> [(relation, object)]
        self.inc = defaultdict(list)  # object  -> [(relation, subject)]
        self.triples = []
        for t in triples:
            self.add(*t)

    def add(self, subj, rel, obj):
        if (subj, rel, obj) in self.triples:
            return
        self.triples.append((subj, rel, obj))
        self.out[subj].append((rel, obj))
        self.inc[obj].append((rel, subj))

    def entities(self):
        names = set()
        for s, _, o in self.triples:
            names.add(s)
            names.add(o)
        return names

    def query(self, subj=None, rel=None, obj=None):
        """Pattern match. None means 'any'."""
        return [
            t for t in self.triples
            if (subj is None or t[0] == subj) and (rel is None or t[1] == rel) and (obj is None or t[2] == obj)
        ]

    def neighbors(self, entity):
        """Edges in both directions, as triples."""
        edges = [(entity, r, o) for r, o in self.out[entity]]
        edges += [(s, r, entity) for r, s in self.inc[entity]]
        return edges


def find_entities(question, store):
    """Naive entity linking: any known entity name that appears in the question (case-insensitive)."""
    q = question.lower()
    return sorted(e for e in store.entities() if e.lower() in q)


def multi_hop(store, start, max_hops=2):
    """Breadth-first walk from `start`. Returns every triple reached within max_hops, in visit order."""
    seen_nodes = {start}
    seen_edges = []
    frontier = deque([(start, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth == max_hops:
            continue
        for triple in store.neighbors(node):
            if triple not in seen_edges:
                seen_edges.append(triple)
            nxt = triple[2] if triple[0] == node else triple[0]
            if nxt not in seen_nodes:
                seen_nodes.add(nxt)
                frontier.append((nxt, depth + 1))
    return seen_edges


def follow_path(store, start, relations):
    """Follow an explicit chain of relations, e.g. ['owned_by', 'managed_by']."""
    current = [start]
    for rel in relations:
        current = [o for node in current for (_, _, o) in store.query(subj=node, rel=rel)]
    return current


def graph_context(question, store, max_hops=2):
    """What we would paste into the LLM prompt: facts as short sentences."""
    facts = []
    for entity in find_entities(question, store):
        for s, r, o in multi_hop(store, entity, max_hops):
            line = f"{s} {r.replace('_', ' ')} {o}."
            if line not in facts:
                facts.append(line)
    return facts


if __name__ == "__main__":
    kg = TripleStore(TRIPLES)
    q = "Who manages the team that owns billing-service?"
    print("Entities :", find_entities(q, kg))
    print("Path     :", follow_path(kg, "billing-service", ["owned_by", "managed_by"]))
    print("Context  :")
    for fact in graph_context(q, kg):
        print("  -", fact)
