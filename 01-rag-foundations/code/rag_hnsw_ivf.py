"""Toy approximate nearest neighbour (ANN) indexes, stdlib only.

* brute_force_search  - compare the query with every vector (exact, slow at scale)
* IVFIndex            - k-means clusters; search only the `nprobe` closest clusters
* GraphIndex          - a flat "navigable small world" graph searched with a beam,
                        the core idea behind HNSW (real HNSW adds layers on top)

Every search returns how many distance computations it made, so you can see
the speed/accuracy trade-off.

Run:  python rag_hnsw_ivf.py
"""
import heapq
import math
import random


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def brute_force_search(query, vectors, k=1):
    """Exact search. Returns (ids, distance_computations)."""
    scored = sorted(range(len(vectors)), key=lambda i: dist(query, vectors[i]))
    return scored[:k], len(vectors)


def kmeans(vectors, n_clusters, iters=10, seed=0):
    """Plain k-means. Returns (centroids, assignment list)."""
    rng = random.Random(seed)
    centroids = [list(v) for v in rng.sample(vectors, n_clusters)]
    assign = [0] * len(vectors)
    for _ in range(iters):
        for i, v in enumerate(vectors):
            assign[i] = min(range(n_clusters), key=lambda c: dist(v, centroids[c]))
        for c in range(n_clusters):
            members = [vectors[i] for i in range(len(vectors)) if assign[i] == c]
            if members:
                centroids[c] = [sum(col) / len(members) for col in zip(*members)]
    return centroids, assign


class IVFIndex:
    """Inverted File index: vectors are grouped into clusters ("lists")."""

    def __init__(self, vectors, nlist=4, seed=0):
        self.vectors = vectors
        self.centroids, assign = kmeans(vectors, nlist, seed=seed)
        self.lists = {c: [] for c in range(nlist)}
        for i, c in enumerate(assign):
            self.lists[c].append(i)

    def search(self, query, k=1, nprobe=1):
        """Look only inside the `nprobe` nearest clusters. Returns (ids, distance_computations)."""
        order = sorted(range(len(self.centroids)), key=lambda c: dist(query, self.centroids[c]))
        computations = len(self.centroids)
        candidates = [i for c in order[:nprobe] for i in self.lists[c]]
        computations += len(candidates)
        candidates.sort(key=lambda i: dist(query, self.vectors[i]))
        return candidates[:k], computations


class GraphIndex:
    """Each vector is a node linked to its `m` nearest neighbours plus a few random long links.

    Search: start at an entry node, keep a beam of the `ef` best nodes seen, and
    walk to neighbours that are closer to the query. Stop when nothing improves.
    """

    def __init__(self, vectors, m=5, random_links=2, seed=0):
        rng = random.Random(seed)
        self.vectors = vectors
        n = len(vectors)
        self.neighbors = []
        for i in range(n):
            near = sorted((j for j in range(n) if j != i), key=lambda j: dist(vectors[i], vectors[j]))[:m]
            far = rng.sample(range(n), min(random_links, n))
            self.neighbors.append(sorted(set(near) | {j for j in far if j != i}))
        # Make links two-way, like HNSW does.
        for i in range(n):
            for j in self.neighbors[i]:
                if i not in self.neighbors[j]:
                    self.neighbors[j].append(i)
        self.entry = 0

    def search(self, query, k=1, ef=8):
        """Beam search over the graph. Returns (ids, distance_computations)."""
        d0 = dist(query, self.vectors[self.entry])
        computations = 1
        visited = {self.entry}
        candidates = [(d0, self.entry)]   # min-heap: closest unexplored first
        best = [(-d0, self.entry)]        # max-heap (negated): the ef best found so far
        while candidates:
            d, node = heapq.heappop(candidates)
            if d > -best[0][0] and len(best) >= ef:
                break  # closest unexplored node is worse than our worst kept node
            for nb in self.neighbors[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                dn = dist(query, self.vectors[nb])
                computations += 1
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(best, (-dn, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        result = sorted((-nd, i) for nd, i in best)
        return [i for _, i in result[:k]], computations


def make_clustered_data(n_per_cluster=50, centers=((0, 0), (10, 0), (0, 10), (10, 10)), seed=0):
    rng = random.Random(seed)
    return [[cx + rng.gauss(0, 1), cy + rng.gauss(0, 1)] for cx, cy in centers for _ in range(n_per_cluster)]


def recall_at_1(index_search, vectors, queries):
    hits = 0
    total_comp = 0
    for q in queries:
        truth, _ = brute_force_search(q, vectors, 1)
        got, comp = index_search(q)
        hits += int(got[:1] == truth)
        total_comp += comp
    return hits / len(queries), total_comp / len(queries)


if __name__ == "__main__":
    data = make_clustered_data()
    rng = random.Random(42)
    queries = [[rng.uniform(-2, 12), rng.uniform(-2, 12)] for _ in range(50)]
    print(f"{len(data)} vectors, {len(queries)} queries")
    r, c = recall_at_1(lambda q: brute_force_search(q, data), data, queries)
    print(f"brute force     recall@1={r:.2f}  avg distance computations={c:.0f}")
    ivf = IVFIndex(data, nlist=4)
    for nprobe in (1, 2, 4):
        r, c = recall_at_1(lambda q: ivf.search(q, nprobe=nprobe), data, queries)
        print(f"IVF nprobe={nprobe}     recall@1={r:.2f}  avg distance computations={c:.0f}")
    graph = GraphIndex(data)
    for ef in (2, 8, 32):
        r, c = recall_at_1(lambda q: graph.search(q, ef=ef), data, queries)
        print(f"graph ef={ef:<3}     recall@1={r:.2f}  avg distance computations={c:.0f}")
