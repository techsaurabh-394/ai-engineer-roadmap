import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag_hnsw_ivf import (  # noqa: E402
    GraphIndex,
    IVFIndex,
    brute_force_search,
    make_clustered_data,
    recall_at_1,
)

DATA = make_clustered_data()
RNG = random.Random(7)
QUERIES = [[RNG.uniform(-2, 12), RNG.uniform(-2, 12)] for _ in range(30)]


def test_brute_force_exact():
    vecs = [[0, 0], [5, 5], [1, 1]]
    ids, comp = brute_force_search([0.9, 0.9], vecs, k=2)
    assert ids == [2, 0]
    assert comp == 3


def test_ivf_lists_cover_all_vectors():
    ivf = IVFIndex(DATA, nlist=4)
    all_ids = sorted(i for ids in ivf.lists.values() for i in ids)
    assert all_ids == list(range(len(DATA)))


def test_ivf_nprobe_trades_work_for_recall():
    ivf = IVFIndex(DATA, nlist=4)
    r1, c1 = recall_at_1(lambda q: ivf.search(q, nprobe=1), DATA, QUERIES)
    r4, c4 = recall_at_1(lambda q: ivf.search(q, nprobe=4), DATA, QUERIES)
    assert c1 < c4
    assert r4 == 1.0  # probing every list is exact
    assert r1 >= 0.8
    assert c1 < len(DATA) / 2


def test_graph_search_is_cheaper_than_brute_force():
    g = GraphIndex(DATA)
    r, c = recall_at_1(lambda q: g.search(q, ef=16), DATA, QUERIES)
    assert r >= 0.9
    assert c < len(DATA)


def test_graph_links_are_two_way():
    g = GraphIndex(DATA[:40])
    for i, nbs in enumerate(g.neighbors):
        for j in nbs:
            assert i in g.neighbors[j]
