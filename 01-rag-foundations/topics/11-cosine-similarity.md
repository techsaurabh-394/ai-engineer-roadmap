# Cosine Similarity
> **Level:** Beginner · **Reading time:** 6 min · **Phase:** 1 — RAG Foundations

**In one line:** Cosine similarity measures how closely two vectors point in the same direction, from −1 (opposite) through 0 (unrelated) to 1 (same direction), ignoring their length.

## Explain it like I'm new
Picture two people pointing at the sky. If they point at the same star, their arms are aligned, even if one person has longer arms. Cosine similarity only cares about the **direction** of the arms, not their length.

For text [embeddings](02-embeddings.md), direction carries the meaning. A short sentence and a long paragraph about the same topic often point in a similar direction even though their vectors may have different lengths. Cosine similarity says "these are about the same thing" without being fooled by length.

The **cosine** is a function from school geometry. The cosine of the angle between two arrows is 1 when the angle is 0°, 0 when the angle is 90° (a right angle), and −1 at 180°.

- **Vector length** (also called the **norm** or magnitude): `‖a‖ = √(a₁² + a₂² + …)`, Pythagoras extended to many dimensions.
- **Dot product**: `a · b = a₁b₁ + a₂b₂ + …` (see [Dot Product](13-dot-product.md)).

## How it works
1. Compute the dot product of the two vectors.
2. Compute the length of each vector.
3. Divide: `cos(a, b) = (a · b) / (‖a‖ × ‖b‖)`.

```text
        b = [2, 4]
       /
      /   angle 0°  -> cos = 1.0   (a = [1, 2] lies on the same line)
     /
    o-------- c = [2, -1] is at 90° to a -> cos = 0.0
```

Worked example with `a = [1, 2, 3]` and `b = [2, 4, 6]`:
- `a · b = 1·2 + 2·4 + 3·6 = 28`
- `‖a‖ = √(1 + 4 + 9) = √14 ≈ 3.742`; `‖b‖ = √(4 + 16 + 36) = √56 ≈ 7.483`
- `cos = 28 / (3.742 × 7.483) = 28 / 28 = 1.0` → same direction (b is just a doubled a).

With `c = [3, 0, −1]`: `a · c = 3 + 0 − 3 = 0`, so `cos(a, c) = 0` → perpendicular, "unrelated".

**Shortcut:** if both vectors are **normalized** (length 1), the denominator is 1, so cosine similarity equals the dot product. Many embedding pipelines normalize for this reason.

**Cosine distance** is often defined as `1 − cosine similarity`, so 0 means identical. Check which one your database reports.

## Code example
From [`code/rag_vector_math.py`](../code/rag_vector_math.py):

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(sum(x * x for x in a))

def cosine_similarity(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)

print(cosine_similarity([1, 2, 3], [2, 4, 6]))    # 1.0
print(cosine_similarity([1, 2, 3], [3, 0, -1]))   # 0.0
print(cosine_similarity([1, 0], [-1, 0]))         # -1.0
```

Run: `cd 01-rag-foundations/code && python rag_vector_math.py`

**In real projects** with NumPy (`pip install numpy`):

```python
import numpy as np

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# Many docs at once: normalize rows, then one matrix product
docs = np.random.rand(1000, 384); q = np.random.rand(384)
docs /= np.linalg.norm(docs, axis=1, keepdims=True); q /= np.linalg.norm(q)
scores = docs @ q            # 1000 cosine similarities
top5 = np.argsort(-scores)[:5]
```

## When to use it / when not to
- The usual default for text embeddings. Many embedding models are trained with cosine-style objectives; check the model card.
- Use it when vector length is noise (for example, it changes with text length) and only direction matters.
- If vectors are already normalized, use the cheaper dot product; rankings are identical.
- If the model was trained so that length carries information, use the metric the model card recommends instead.

## Common mistakes
- Treating the score as a probability or a universal threshold. "0.8" means different things for different models.
- Dividing by zero on an empty or all-zero vector.
- Mixing up cosine *similarity* (higher is better) and cosine *distance* (lower is better).
- Normalizing documents but not the query (or the reverse) and then using dot product.

## Interview questions
**1. Why is cosine similarity popular for text embeddings?**
<details><summary>Answer</summary>
It compares direction only, so differences in vector length (often unrelated to meaning) do not affect the score, and many embedding models are trained with cosine-based objectives.
</details>

**2. How do cosine similarity and dot product relate?**
<details><summary>Answer</summary>
Cosine is the dot product divided by both lengths. For unit-length vectors the lengths are 1, so the two are equal and rank results identically.
</details>

**3. What does a cosine similarity of 0 mean?**
<details><summary>Answer</summary>
The vectors are perpendicular: no shared direction. For embeddings that usually means unrelated content, though in practice scores for unrelated text are often small positive numbers rather than exactly 0.
</details>

## Go deeper
- Reimers & Gurevych, *Sentence-BERT* (cosine similarity for sentence embeddings) — https://arxiv.org/abs/1908.10084
- FAISS docs (metric types) — https://faiss.ai/

**Prev:** [Vector Databases](10-vector-databases.md) · **Next:** [Euclidean Distance](12-euclidean-distance.md) · **Back to phase:** [Phase 1](../README.md)
