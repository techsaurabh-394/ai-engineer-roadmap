# Euclidean Distance
> **Level:** Beginner · **Reading time:** 5 min · **Phase:** 1 — RAG Foundations

**In one line:** Euclidean distance is the straight-line distance between two points, so smaller means more similar and 0 means identical.

## Explain it like I'm new
Put two pins on a paper map and stretch a ruler between them. The ruler reading is the Euclidean distance. It is the "as the crow flies" distance you learned with Pythagoras: for a right triangle with sides 3 and 4, the long side is 5.

An [embedding](02-embeddings.md) is a point in a space with many dimensions. The same ruler idea works in 384 or 1,536 dimensions; you just add more squared terms.

Unlike [cosine similarity](11-cosine-similarity.md), Euclidean distance cares about **length** too. Two vectors that point the same way but have different lengths are "far apart" by Euclidean distance, yet "identical" by cosine. It is also called the **L2 distance** (after the L2 norm). Vector databases and FAISS often call it `L2`.

## How it works
1. Subtract the vectors element by element.
2. Square each difference.
3. Add them up.
4. Take the square root: `d(a, b) = √((a₁−b₁)² + (a₂−b₂)² + …)`.

```text
y
4 |        * b (3,4)
3 |       /
2 |      /  d = 5
1 |     /
0 *----+----+----+-- x
  a (0,0)
```

Worked examples:
- `a = [0, 0]`, `b = [3, 4]`: `√(9 + 16) = √25 = 5`.
- `a = [1, 2, 3]`, `b = [2, 4, 6]`: differences `[−1, −2, −3]`, squares `[1, 4, 9]`, sum 14, distance `√14 ≈ 3.742`. Cosine said these are identical in direction (1.0). Euclidean says they are 3.7 apart, because `b` is twice as long.

**Squared distance:** many libraries skip the square root. Ranking is the same (the square root keeps order), and it is cheaper.

**The useful identity.** For unit-length vectors `u` and `v`:
`‖u − v‖² = 2 − 2·cos(u, v)`.
So on normalized embeddings, ranking by smallest Euclidean distance gives *exactly the same order* as ranking by highest cosine similarity. [`code/rag_vector_math.py`](../code/rag_vector_math.py) checks this in its tests.

## Code example
From [`code/rag_vector_math.py`](../code/rag_vector_math.py):

```python
import math

def euclidean_distance(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def normalize(a):
    n = math.sqrt(sum(x * x for x in a))
    return [x / n for x in a]

print(euclidean_distance([0, 0], [3, 4]))              # 5.0
print(euclidean_distance([1, 2, 3], [2, 4, 6]))        # 3.741...
u, v = normalize([1, 2, 3]), normalize([2, 4, 6])
print(euclidean_distance(u, v))                        # 0.0: same point after normalizing
```

**In real projects** (FAISS, `pip install faiss-cpu numpy`):

```python
import faiss
import numpy as np

xb = np.random.rand(10_000, 128).astype("float32")     # stored vectors
xq = np.random.rand(5, 128).astype("float32")          # queries
index = faiss.IndexFlatL2(128)                         # exact search, squared L2
index.add(xb)
distances, ids = index.search(xq, 4)                   # 4 nearest per query
```

## When to use it / when not to
- Use it when the model card recommends L2, or when absolute position matters (for example, some image or clustering tasks).
- On normalized embeddings, it ranks like cosine, so pick whichever your database computes fastest.
- On unnormalized vectors, length differences dominate; prefer cosine for text unless told otherwise.
- [IVF](15-ivf-index.md) indexes use k-means, which is based on Euclidean distance.

| Metric | Higher or lower is better? | Sensitive to length? |
|---|---|---|
| Cosine similarity | Higher | No |
| Dot product | Higher | Yes |
| Euclidean (L2) | Lower | Yes |

## Common mistakes
- Sorting distances in descending order as if they were similarities.
- Comparing raw L2 scores across different models or collections.
- Forgetting that FAISS `IndexFlatL2` returns *squared* distances.
- Using L2 on unnormalized text embeddings and getting results biased by vector length.

## Interview questions
**1. How do Euclidean distance and cosine similarity differ?**
<details><summary>Answer</summary>
Euclidean measures straight-line distance and depends on vector length; cosine measures only the angle. Two vectors in the same direction but with different lengths have cosine 1 but a non-zero Euclidean distance.
</details>

**2. When do they produce the same ranking?**
<details><summary>Answer</summary>
When all vectors are normalized to length 1, because then squared Euclidean distance equals 2 − 2·cosine, a decreasing function of cosine.
</details>

**3. Why do libraries often use squared L2?**
<details><summary>Answer</summary>
The square root does not change the order of results and costs extra compute, so skipping it is faster with the same ranking.
</details>

## Go deeper
- FAISS docs (IndexFlatL2, metric types) — https://faiss.ai/
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS) — https://arxiv.org/abs/1702.08734

**Prev:** [Cosine Similarity](11-cosine-similarity.md) · **Next:** [Dot Product](13-dot-product.md) · **Back to phase:** [Phase 1](../README.md)
