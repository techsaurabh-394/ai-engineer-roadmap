# 02 · RAG (Retrieval-Augmented Generation)

### 1. Describe a RAG pipeline end to end.
Ingest: extract text, clean, chunk, embed, store with metadata. Query: rewrite the question if needed, retrieve candidate chunks (vector, keyword or hybrid), optionally rerank, build a prompt with the best chunks, generate an answer that cites sources, and log everything for evaluation.

### 2. How do you choose a chunking strategy?
Chunks should be small enough to be specific and large enough to carry meaning. Common starting points are a few hundred tokens with modest overlap, splitting on semantic boundaries (headings, paragraphs, sentences) rather than fixed character counts. Keep metadata (document, page, section) for citations and filtering. Tune against your own evaluation set, since the best size depends on the documents and questions.

### 3. Dense, sparse or hybrid retrieval?
Dense (embedding) search matches meaning and handles paraphrase; sparse (BM25/keyword) matches exact terms such as names, IDs and error codes. Hybrid search runs both and merges rankings, often with reciprocal rank fusion, and is a strong default because each covers the other's weaknesses.

### 4. What does a reranker add?
A reranker (often a cross-encoder) scores each query-chunk pair jointly, which is more accurate than comparing precomputed vectors but slower. Typical pattern: retrieve 20 to 100 candidates cheaply, rerank, keep the top few for the prompt. It usually improves precision at the cost of latency.

### 5. How do you pick top-k?
Retrieve enough to cover the answer, but not so many that noise and cost rise. Measure recall at several k on an evaluation set, then choose the smallest k where recall plateaus; reranking lets you retrieve wide and pass only a few chunks to the model.

### 6. What is query rewriting and when is it useful?
Reformulating the user's question before retrieval: resolving pronouns from chat history, expanding abbreviations, splitting a multi-part question into sub-queries, or generating a hypothetical answer to embed. It helps with conversational follow-ups and vague queries; test that it doesn't drift from the user's intent.

### 7. How do you evaluate a RAG system?
Evaluate retrieval and generation separately. Retrieval: hit rate at k, MRR, nDCG, recall against labelled relevant chunks. Generation: faithfulness (is every claim supported by the sources), answer relevance, and citation correctness. Build a golden set of real questions with expected sources, run it on every change, and add production failures to it.

### 8. List common RAG failure modes.
The answer isn't in the index (ingestion gaps); the right chunk exists but isn't retrieved (chunking, embeddings, query mismatch); it's retrieved but ranked too low; the context is contradictory or stale; the model ignores or misuses sources; tables and scanned PDFs extracted badly. Diagnose which stage failed before changing anything.

### 9. How do you handle tables, PDFs and other messy sources?
Use extraction that preserves structure (layout-aware parsers, OCR for scans), keep tables as structured text or Markdown rather than flattened lines, attach page and section metadata, and test extraction quality on a sample before indexing everything. Bad extraction can't be fixed downstream.

### 10. How do you keep an index fresh?
Track document versions or hashes, re-embed only changed documents, delete chunks of removed documents, and store timestamps in metadata so queries can prefer recent content. Make ingestion idempotent so re-runs are safe.

### 11. How do you enforce access control in RAG?
Filter at retrieval time using permissions stored as metadata (user, group, tenant), so unauthorised chunks are never retrieved or placed in the prompt. Filtering after generation is too late. Test with users who must not see certain documents.

### 12. When is RAG the wrong tool?
When the task needs reasoning over the whole document set (aggregation, counting, comparisons across many files), when the knowledge is small enough to fit in the prompt, when the question needs structured data (use SQL or an API tool), or when the problem is behaviour or style (consider fine-tuning).

### 13. How do you choose a vector database?
Match your scale and ops capacity: pgvector keeps vectors in Postgres alongside your data (simple, good for moderate scale); dedicated stores such as Qdrant, Weaviate or Pinecone offer richer filtering and scaling features. Check filtering support, hybrid search, update patterns and cost. Many approximate indexes use HNSW, trading a little recall for large speedups.
