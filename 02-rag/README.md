# 02 · RAG (Retrieval-Augmented Generation)

**Goal:** make an AI answer from *your* documents and show where each claim came from. Time: 2 hours.

## Big idea
A model doesn't know your private or recent documents. RAG fixes that in two phases:

1. **Ingest (once per document):** split into chunks, turn each chunk into a vector, store it.
2. **Answer (every question):** turn the question into a vector, find the closest chunks, give *only those* to the model, and ask it to answer with citations.

![RAG pipeline](../assets/rag-pipeline.svg)

## Run it
```bash
cd 02-rag/code
python mini_rag.py "Why use multi-stage builds?"
python mini_rag.py "Where should refresh tokens be stored?"
pytest -q
```
`mini_rag.py` is the whole pipeline in about 100 readable lines: `chunk()` → `Index` (hashed TF-IDF vectors) → `search()` → `answer()`. With `ANTHROPIC_API_KEY` set, Claude writes the final answer from the retrieved chunks; without one, it prints the best chunk.

## Try it
1. Add your own document to `DOCS` and ask a question about it.
2. Change `size` in `chunk()` to 100, then 1000. What happens to the retrieved passage?
3. Ask something that isn't in the documents. Does it say it can't find it?
4. Break it on purpose: use a question whose words don't appear in the text (a paraphrase). This is why real systems use dense embeddings or hybrid search.

## Level up
The full app with PDF upload, a web UI, FastAPI and an evaluation script lives in the [rag-docs-chat](https://github.com/techsaurabh-394/rag-docs-chat) repo.

## Common mistakes
- Chunks too big (vague matches) or too small (no context).
- No metadata, so you can't cite pages.
- Skipping evaluation, so you can't tell if a change helped (step 5).
- Filtering permissions *after* generation instead of before retrieval.

## Checkpoint
- Why retrieve first instead of pasting every document into the prompt?
- What is the difference between dense, sparse and hybrid retrieval?
- What does a reranker add?
- Name three ways a RAG system fails.

(Answers: [interview prep, RAG](../07-interview-prep/02-rag.md))

**Next:** [03 · Agents](../03-agents/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
