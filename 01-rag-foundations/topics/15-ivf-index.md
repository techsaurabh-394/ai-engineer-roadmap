# IVF Index
> **Level:** Advanced · **Reading time:** 7 min · **Phase:** 1 — RAG Foundations

**In one line:** An IVF (Inverted File) index groups vectors into clusters, then at query time searches only the few clusters closest to the query instead of every vector.

## Explain it like I'm new
Imagine a huge library with no catalogue. To find a book about volcanoes you would check every shelf. Now imagine the books are sorted into sections: Geology, Cooking, History. You walk to Geology (and maybe Geography, just in case) and search only there. You skip 95% of the library.

IVF does this with vectors:
- **Clustering** groups similar vectors together. IVF uses **k-means**, an algorithm that finds `nlist` centre points (**centroids**) and assigns every vector to its nearest centroid.
- Each cluster keeps a list of the vector IDs inside it. This list is the **inverted list**, borrowed from search-engine terminology.
- At query time, compare the query with the centroids only, pick the `nprobe` closest clusters, and do exact search inside them.

The risk: the true nearest neighbour might sit just across a cluster border, in a cluster you did not probe. Probing more clusters (bigger `nprobe`) lowers that risk and costs more time.

## How it works
**Training and adding**
1. Run k-means on a sample of vectors to get `nlist` centroids. This is the **training** step.
2. Assign every vector to its nearest centroid and append its ID to that cluster's list.

**Searching**
1. Compute the distance from the query to all `nlist` centroids.
2. Pick the `nprobe` nearest clusters.
3. Compute exact distances to every vector inside those clusters.
4. Return the best k.

```text
          centroid A           centroid B
         .  . A  .            .   B .  .
          .  .  .               .  .  .
                         q  <- query is near A and B
          centroid C           centroid D
         .  . C  .            .  D  .
nprobe=1 -> search only A          (fast, may miss a neighbour in B)
nprobe=2 -> search A and B         (slower, safer)
nprobe=4 -> search everything      (= brute force)
```

Worked example from the demo: 200 vectors in 4 clusters, `nlist = 4`. With `nprobe = 1`, a query costs 4 centroid comparisons + about 50 vectors ≈ 54 distance computations, versus 200 for brute force. On the demo's random queries, recall@1 is 0.98 at `nprobe = 1` and 1.00 at `nprobe = 2`. (Toy data; real recall depends on your data and settings.)

A common heuristic is to make `nlist` grow with the square root of the collection size (a few times `√N` for N vectors), then tune `nprobe` while measuring recall.

## Code example
From [`code/rag_hnsw_ivf.py`](../code/rag_hnsw_ivf.py):

```python
class IVFIndex:
    def __init__(self, vectors, nlist=4, seed=0):
        self.vectors = vectors
        self.centroids, assign = kmeans(vectors, nlist, seed=seed)   # training
        self.lists = {c: [] for c in range(nlist)}
        for i, c in enumerate(assign):
            self.lists[c].append(i)                                   # inverted lists

    def search(self, query, k=1, nprobe=1):
        order = sorted(range(len(self.centroids)),
                       key=lambda c: dist(query, self.centroids[c]))
        candidates = [i for c in order[:nprobe] for i in self.lists[c]]
        candidates.sort(key=lambda i: dist(query, self.vectors[i]))
        return candidates[:k]
```

Run `python rag_hnsw_ivf.py` to print recall and work for `nprobe = 1, 2, 4`.

**In real projects** (`pip install faiss-cpu numpy`):

```python
import faiss
import numpy as np

d, nlist = 128, 100
xb = np.random.rand(100_000, d).astype("float32")
quantizer = faiss.IndexFlatL2(d)                  # holds the centroids
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(xb)                                   # k-means
index.add(xb)
index.nprobe = 8                                  # clusters to search
D, I = index.search(xb[:5], 4)
```

`IndexIVFPQ` adds **product quantization** (compressing each vector into a few bytes) for very large collections.

## When to use it / when not to
- Good for large collections where memory matters, especially combined with quantization (IVF-PQ).
- Fast to build compared with HNSW at large scale, but needs a training sample that represents your data.
- If the data distribution changes a lot after training, clusters get unbalanced; retrain.
- For small or medium collections that fit in RAM with low-latency needs, [HNSW](14-hnsw.md) is often simpler.

| `nprobe` | Speed | Recall |
|---|---|---|
| 1 | Fastest | Lowest |
| ~1–10% of `nlist` | Common balance | Usually high |
| = `nlist` | Same as brute force | Exact |

## Common mistakes
- Training on too few vectors (k-means needs many points per centroid) or on unrepresentative data.
- Forgetting to call `train()` before `add()` in FAISS.
- Leaving `nprobe = 1` (FAISS default) and wondering why recall is poor.
- Choosing `nlist` far too large for the data, giving many tiny or empty clusters.

## Interview questions
**1. How does an IVF index speed up search?**
<details><summary>Answer</summary>
It clusters vectors with k-means and, at query time, only searches the nprobe clusters whose centroids are closest to the query, skipping most of the data.
</details>

**2. What do nlist and nprobe control?**
<details><summary>Answer</summary>
nlist is the number of clusters (set at build time). nprobe is how many clusters are searched per query; higher nprobe means better recall and slower queries.
</details>

**3. Why can IVF miss the true nearest neighbour?**
<details><summary>Answer</summary>
The neighbour may sit in a cluster near a border that was not among the nprobe probed clusters. Raising nprobe reduces this.
</details>

## Go deeper
- FAISS docs — https://faiss.ai/
- Johnson et al., *Billion-scale similarity search with GPUs* — https://arxiv.org/abs/1702.08734

**Prev:** [HNSW](14-hnsw.md) · **Next:** [Hybrid Search](16-hybrid-search.md) · **Back to phase:** [Phase 1](../README.md)
