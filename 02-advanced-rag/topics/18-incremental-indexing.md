# Incremental Indexing

> **Level:** Intermediate · **Reading time:** 6 min · **Phase:** 2 — Advanced RAG

**In one line:** Incremental indexing updates your RAG index by re-embedding only the chunks that are new or changed, and deleting chunks that disappeared, instead of rebuilding everything.

## Explain it like I'm new

If one page of a 500-page book changes, you don't reprint the whole book. You swap that page. Incremental indexing does the same for your vector index.

**Embedding** every chunk costs time and money (API calls or GPU time). A full re-index of 1 million chunks every night is wasteful when only 200 changed. Worse, if you only *add* new versions without removing old ones, your index fills with **stale chunks**: outdated text that still gets retrieved and leads to wrong answers.

The usual trick is a **content hash**: a short fingerprint computed from the text (for example SHA-256). Same text → same hash. Change one character → completely different hash. By comparing hashes from the last run with the current ones, you know exactly what to add, skip and delete.

## How it works

1. Chunk each document deterministically (same input → same chunks).
2. Compute `hash = sha256(chunk_text)` for every chunk.
3. Compare with the stored record of `doc_id → [hashes]`:
   - hash already stored → **skip**, no embedding call
   - new hash → **embed and insert**
   - old hash no longer present → **delete** (unless another document still uses it)
4. Save the new `doc_id → [hashes]` record.
5. For deleted documents, remove all their chunks.

```text
Old version              New version              Action
"Intro paragraph."  ──▶  "Intro paragraph."       skip    (same hash)
"Install with pip." ──▶  "Install with uv."       embed new + delete old
"Run the server."   ──▶  "Run the server."        skip
                                                  embed calls: 1 instead of 3
```

## Code example

From [`adv_incremental_index.py`](../code/adv_incremental_index.py). `CountingEmbedder` counts calls, so the test can prove work was skipped.

```python
from adv_incremental_index import CountingEmbedder, IncrementalIndex

emb = CountingEmbedder()
idx = IncrementalIndex(emb)
v1 = "Intro paragraph.\n\nInstall with pip.\n\nRun the server."
v2 = "Intro paragraph.\n\nInstall with uv.\n\nRun the server."

print(idx.upsert("readme", v1), emb.calls)  # {'added': 3, 'unchanged': 0, 'removed': 0} 3
print(idx.upsert("readme", v1), emb.calls)  # {'added': 0, 'unchanged': 3, 'removed': 0} 3
print(idx.upsert("readme", v2), emb.calls)  # {'added': 1, 'unchanged': 2, 'removed': 1} 4
```

The core of `upsert`:

```python
for piece in chunk(text):
    h = content_hash(piece)
    new_hashes.append(h)
    if h not in self.vectors:          # only embed what we have never seen
        self.vectors[h] = self.embedder.embed(piece)
        self.texts[h] = piece
        added += 1
old = set(self.doc_chunks.get(doc_id, []))
self.doc_chunks[doc_id] = new_hashes
removed = self._garbage_collect(old - set(new_hashes))  # drop stale chunks nobody uses
```

**In real projects** LangChain has an indexing API with a "record manager" that tracks hashes and supports cleanup modes for deleting stale documents, and LlamaIndex has ingestion pipelines with document stores that skip unchanged documents. Check [LangChain](https://python.langchain.com/) and [LlamaIndex](https://docs.llamaindex.ai/) for current usage.

## When to use it / when not to

- **Use it** whenever documents change over time: wikis, tickets, code repos, product docs.
- **Use it** when embedding is expensive or rate-limited.
- **Do a full rebuild instead** when you change the embedding model, the chunker settings or the chunk metadata format. Old hashes then describe incompatible vectors.
- **Combine** with change detection at the source (modified timestamps, webhooks, CDC) so you don't even re-read unchanged files.

| Strategy | Cost per run | Stale chunks |
|---|---|---|
| Append only | Low | Accumulate (bad) |
| Full rebuild | High | None |
| Hash-based incremental | Low | None, if deletes are handled |

## Common mistakes

- Non-deterministic chunking (e.g. depends on dictionary order or timestamps), so every run looks like a full change.
- Forgetting deletes. Old chunks keep answering questions with outdated facts.
- Hashing only the text when metadata matters. If `permissions` change, the chunk must be updated too; include relevant metadata in the hash.
- Keeping the hash record in memory only. Persist it next to the index.

## Interview questions

**1. How do you avoid re-embedding the whole corpus on every update?**
<details><summary>Answer</summary>
Hash each chunk's content (and important metadata), keep a record of hashes per document, embed only hashes not yet stored, and delete hashes that disappeared.
</details>

**2. When must you do a full re-index anyway?**
<details><summary>Answer</summary>
When the embedding model changes, when chunking parameters change, or when the vector schema/metadata format changes. Vectors from different models or chunkers are not compatible.
</details>

**3. Why can stale chunks be dangerous?**
<details><summary>Answer</summary>
They can be retrieved and cited as current truth, producing wrong answers (old prices, revoked policies) and, if permissions changed, they may leak content to users who should no longer see it.
</details>

## Go deeper

- [Python hashlib documentation](https://docs.python.org/3/library/hashlib.html)
- [LangChain documentation](https://python.langchain.com/)

**Prev:** [API RAG](17-api-rag.md) · **Next:** [Streaming Retrieval](19-streaming-retrieval.md) · **Back to phase:** [Phase 2](../README.md)
