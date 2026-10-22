# Phase 2 — Advanced RAG

**Goal:** take the basic RAG pipeline from Phase 1 and make it production-grade: better ranking, smarter queries, richer data sources, fresh indexes and real evaluation.

**Who it's for:** you can already chunk documents, embed them, retrieve the top-k chunks and prompt an LLM with them. Now your answers are "okay but not great" and you want to know *which lever to pull*.

**Prerequisites:**
- [Phase 1 — RAG Foundations](../01-rag-foundations/README.md) (chunking, embeddings, vector search, prompting with context)
- Basic Python (functions, dicts, list comprehensions)
- About 10–14 hours for the whole phase

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 1 | Reranking | Intermediate | [01-reranking.md](topics/01-reranking.md) |
| 2 | Cross Encoder | Intermediate | [02-cross-encoder.md](topics/02-cross-encoder.md) |
| 3 | Bi Encoder | Intermediate | [03-bi-encoder.md](topics/03-bi-encoder.md) |
| 4 | Context Compression | Intermediate | [04-context-compression.md](topics/04-context-compression.md) |
| 5 | Query Expansion | Intermediate | [05-query-expansion.md](topics/05-query-expansion.md) |
| 6 | Multi Query Retrieval | Intermediate | [06-multi-query-retrieval.md](topics/06-multi-query-retrieval.md) |
| 7 | HyDE | Advanced | [07-hyde.md](topics/07-hyde.md) |
| 8 | Step Back Prompting | Advanced | [08-step-back-prompting.md](topics/08-step-back-prompting.md) |
| 9 | Self Query Retriever | Intermediate | [09-self-query-retriever.md](topics/09-self-query-retriever.md) |
| 10 | Multi Vector Retriever | Advanced | [10-multi-vector-retriever.md](topics/10-multi-vector-retriever.md) |
| 11 | Parent Document Retriever | Intermediate | [11-parent-document-retriever.md](topics/11-parent-document-retriever.md) |
| 12 | Ensemble Retrieval | Intermediate | [12-ensemble-retrieval.md](topics/12-ensemble-retrieval.md) |
| 13 | Knowledge Graph RAG | Advanced | [13-knowledge-graph-rag.md](topics/13-knowledge-graph-rag.md) |
| 14 | GraphRAG | Advanced | [14-graphrag.md](topics/14-graphrag.md) |
| 15 | Multi Modal RAG | Advanced | [15-multi-modal-rag.md](topics/15-multi-modal-rag.md) |
| 16 | SQL RAG | Advanced | [16-sql-rag.md](topics/16-sql-rag.md) |
| 17 | API RAG | Advanced | [17-api-rag.md](topics/17-api-rag.md) |
| 18 | Incremental Indexing | Intermediate | [18-incremental-indexing.md](topics/18-incremental-indexing.md) |
| 19 | Streaming Retrieval | Advanced | [19-streaming-retrieval.md](topics/19-streaming-retrieval.md) |
| 20 | RAG Evaluation (RAGAS / DeepEval) | Advanced | [20-rag-evaluation.md](topics/20-rag-evaluation.md) |

## Learning order

Read in number order. The topics are grouped so each block builds on the last:

1. **Ranking (1–4):** the two-stage retrieve → rerank pattern, why bi-encoders and cross-encoders exist, and trimming context.
2. **Query transformation (5–8):** fixing the *question* before searching — expansion, multi-query with RRF, HyDE, step-back.
3. **Smarter retrievers (9–12):** metadata filters, multiple vectors per document, small-to-big retrieval, hybrid search.
4. **Beyond plain text (13–17):** knowledge graphs, GraphRAG, images and tables, databases, live APIs.
5. **Operating it (18–20):** keeping the index fresh, streaming to users, and measuring quality.

Tip: if you only have one afternoon, read 1, 6, 12, 18 and 20. That covers the changes that most often move real-world quality.

## How to run the code

Everything in [`code/`](code/) is offline, standard-library Python. No API keys, no pip installs beyond `pytest`.

```bash
# from the repo root
python -m pytest -q 02-advanced-rag/code

# run individual demos
cd 02-advanced-rag/code
python eval_harness.py          # hit@3 and MRR on the Phase 1 mini RAG
python adv_rerank.py            # bi-encoder retrieval, then cross-encoder-style rerank
python adv_query_transform.py   # query expansion, multi-query + RRF, step-back, HyDE
python adv_graph_rag.py         # triple store + multi-hop retrieval
python adv_incremental_index.py # hash-based incremental indexing
python adv_rag_metrics.py       # toy context precision/recall, faithfulness, relevancy
```

| File | Topics it supports |
|---|---|
| [`eval_harness.py`](code/eval_harness.py) | 20 (uses `mini_rag` from [Phase 1 code](../01-rag-foundations/code/mini_rag.py)) |
| [`adv_rerank.py`](code/adv_rerank.py) | 1, 2, 3 |
| [`adv_query_transform.py`](code/adv_query_transform.py) | 5, 6, 7, 8, 12 |
| [`adv_graph_rag.py`](code/adv_graph_rag.py) | 13, 14 |
| [`adv_incremental_index.py`](code/adv_incremental_index.py) | 18, 19 |
| [`adv_rag_metrics.py`](code/adv_rag_metrics.py) | 20 |

Try it:
1. In `adv_rerank.py`, add a new document that repeats "cancel" ten times. Does the bi-encoder fall for it? Does the reranker?
2. In `adv_query_transform.py`, remove `"restart"` from `SYNONYMS`. Which document drops out?
3. In `adv_graph_rag.py`, add `("Marco", "reports_to", "CEO")` and call `follow_path` with three relations.
4. In `adv_rag_metrics.py`, write an answer with one extra invented sentence and watch faithfulness fall.

## Phase checkpoint quiz

<details><summary><b>1. Why do production systems retrieve with a bi-encoder and then rerank with a cross-encoder, instead of using one of them?</b></summary>

Bi-encoders pre-compute document vectors, so they can search millions of chunks fast but less precisely. Cross-encoders read query and document together, which is more accurate but needs one model run per pair. Retrieve ~50 cheaply for recall, then rerank to ~5 for precision.
</details>

<details><summary><b>2. What is Reciprocal Rank Fusion and why is it popular?</b></summary>

RRF scores each document as Σ 1/(k + rank) across result lists (k is usually 60). It uses only ranks, so it merges lists from different queries or retrievers without normalising incompatible scores.
</details>

<details><summary><b>3. How does HyDE differ from multi-query retrieval?</b></summary>

HyDE asks the LLM to write a hypothetical *answer* and searches with it, to match the style of real documents. Multi-query asks for several alternative *questions*, runs each, and fuses the results. The HyDE passage is never shown to the user.
</details>

<details><summary><b>4. What problem does the parent-document retriever solve?</b></summary>

The chunk-size trade-off. Small child chunks embed precisely; returning their larger parent section gives the LLM enough context to answer completely.
</details>

<details><summary><b>5. Why combine BM25 with vector search?</b></summary>

BM25 matches exact terms (IDs, error codes, names); vectors match meaning and paraphrases. They fail on different queries, so fusing them improves recall.
</details>

<details><summary><b>6. When would you choose GraphRAG over standard vector RAG?</b></summary>

For global, sensemaking questions over a whole corpus ("what are the main themes?"). GraphRAG pre-builds community summaries and answers by map-reduce over them. For specific fact lookups, vector RAG is cheaper and enough.
</details>

<details><summary><b>7. Name three safety controls for SQL RAG.</b></summary>

A read-only database user scoped to approved tables or views; validation that allows only a single SELECT and adds a LIMIT; timeouts plus row-level security and logging of every generated query.
</details>

<details><summary><b>8. How does incremental indexing avoid re-embedding everything, and what must it not forget?</b></summary>

It hashes each chunk and only embeds hashes not already stored. It must also delete chunks whose hashes disappeared, otherwise stale content keeps being retrieved. Changing the embedding model still requires a full rebuild.
</details>

<details><summary><b>9. Faithfulness is high but context recall is low. Where is the problem?</b></summary>

In retrieval. The model sticks to the context it gets, but the retriever misses needed information. Improve chunking, hybrid search, query transformation or k.
</details>

<details><summary><b>10. Name two pitfalls of LLM-as-judge evaluation.</b></summary>

Judges can prefer longer answers and be inconsistent between runs, and they can miss subtle factual errors. Pin the judge model and prompt, use clear rubrics, and spot-check against human labels.
</details>

More answers: [Interview prep — RAG](../08-interview-prep/02-rag.md) · [Interview prep — evaluation and production](../08-interview-prep/04-evaluation-and-production.md)

**Prev:** [Phase 1 — RAG Foundations](../01-rag-foundations/README.md) · **Next:** [Phase 3 — LLM Concepts](../03-llm-concepts/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
