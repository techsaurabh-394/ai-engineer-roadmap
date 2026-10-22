# Vector Databases
> **Level:** Intermediate · **Reading time:** 7 min · **Phase:** 1 — RAG Foundations

**In one line:** A vector database stores embeddings with their text and metadata, and finds the vectors closest to a query vector quickly, even among millions.

## Explain it like I'm new
A normal database is like a phone book: you look up an exact name. A vector database is like asking a music app "play songs that *sound like* this one". There is no exact match. You want the nearest neighbours by similarity.

The core operation is **nearest neighbour search**: given a query vector, return the k stored vectors that are most similar (the **top-k**). You could compare the query with every stored vector. That is called **brute-force** or **flat** search. It is exact, and fine for thousands of vectors. For millions, it gets slow.

So vector databases build an **index**: a data structure that finds *almost* the best matches while checking only a small part of the data. This is **Approximate Nearest Neighbour (ANN)** search. You trade a little accuracy (**recall**: the share of the true top-k you actually get) for a big speed-up. Two common index types are [HNSW](14-hnsw.md) and [IVF](15-ivf-index.md).

A vector database also handles the boring but vital parts: storing metadata, [filtering](20-metadata-filtering.md), updates and deletes, persistence to disk, and scaling.

## How it works
1. **Create a collection** with a fixed vector size (e.g. 384) and a distance metric ([cosine](11-cosine-similarity.md), [dot product](13-dot-product.md) or [Euclidean](12-euclidean-distance.md)).
2. **Upsert** records: `id`, `vector`, `text` (or a pointer to it), `metadata`. ("Upsert" = insert or update.)
3. The database adds each vector to its **ANN index**.
4. **Query** with a vector, `k`, and an optional metadata filter.
5. It returns IDs, scores, text and metadata, best first.

```text
              +-----------------------------------------+
query vec --> |  ANN index (HNSW / IVF)  --> candidates |
filter    --> |  metadata index          --> allowed    | --> top-k (id, score, text, meta)
              |  storage: vectors, text, metadata       |
              +-----------------------------------------+
```

Rough storage maths: 1 million chunks × 768 dimensions × 4 bytes (float32) ≈ 3 GB for vectors alone, before index overhead. This is why dimension size and **quantization** (storing numbers in fewer bits) matter at scale.

## Code example
The smallest possible "vector database", stdlib only, with brute-force search and filters:

```python
import math

class TinyVectorDB:
    def __init__(self, dim):
        self.dim, self.rows = dim, {}

    def upsert(self, id, vector, text, meta=None):
        assert len(vector) == self.dim
        self.rows[id] = (vector, text, meta or {})

    def delete(self, id):
        self.rows.pop(id, None)

    def query(self, vector, k=3, where=None):
        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb or 1)
        rows = [(id, r) for id, r in self.rows.items()
                if not where or all(r[2].get(f) == v for f, v in where.items())]
        scored = sorted(((cos(vector, r[0]), id, r[1]) for id, r in rows), reverse=True)
        return scored[:k]

db = TinyVectorDB(dim=3)
db.upsert("a", [1, 0, 0], "about tokens", {"team": "security"})
db.upsert("b", [0, 1, 0], "about docker", {"team": "platform"})
print(db.query([0.9, 0.1, 0], k=1))                        # [(0.99.., 'a', 'about tokens')]
print(db.query([0.9, 0.1, 0], k=1, where={"team": "platform"}))
```

See a toy ANN index (IVF and graph search) in [`code/rag_hnsw_ivf.py`](../code/rag_hnsw_ivf.py).

**In real projects** (Qdrant in local in-memory mode, `pip install qdrant-client`):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = QdrantClient(":memory:")
client.create_collection("docs", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
client.upsert("docs", points=[PointStruct(id=1, vector=[1, 0, 0], payload={"team": "security"})])
hits = client.query_points("docs", query=[0.9, 0.1, 0], limit=1).points
```

## When to use it / when not to

| Option | Good for | Notes |
|---|---|---|
| In-memory list / NumPy | Prototypes, < ~10k vectors | Exact, simple |
| FAISS (library) | Fast local search, research | Index only; you manage metadata and storage |
| Chroma, Qdrant, Weaviate, Milvus | Apps needing filters, persistence, APIs | Dedicated vector databases |
| pgvector (Postgres), Elasticsearch/OpenSearch | You already run them | Vectors next to existing data, hybrid search |

- Do not add a new database before you need one. Many teams start with pgvector or a library.
- Pick based on filters, hybrid search support, operations burden and scale, not hype.

## Common mistakes
- Choosing a metric that does not match how the embedding model was trained (check the model card).
- Changing the embedding model without rebuilding the collection.
- Not storing a document ID, so updates create duplicates.
- Default ANN settings with no recall check against brute-force results.

## Interview questions
**1. What does a vector database do that a plain array of vectors does not?**
<details><summary>Answer</summary>
It provides ANN indexes for fast search at scale, metadata storage and filtering, persistence, updates/deletes, and often replication, hybrid search and access control.
</details>

**2. What is the trade-off in approximate nearest neighbour search?**
<details><summary>Answer</summary>
Speed and memory versus recall. ANN checks only part of the data, so it can miss some true nearest neighbours. Index parameters tune this balance.
</details>

**3. When is brute-force search fine?**
<details><summary>Answer</summary>
For small collections (thousands to tens of thousands of vectors) or offline jobs. It is exact and simple, and its latency is often acceptable at that size.
</details>

## Go deeper
- FAISS docs — https://faiss.ai/
- Qdrant docs — https://qdrant.tech/documentation/
- pgvector — https://github.com/pgvector/pgvector

**Prev:** [Metadata](09-metadata.md) · **Next:** [Cosine Similarity](11-cosine-similarity.md) · **Back to phase:** [Phase 1](../README.md)
