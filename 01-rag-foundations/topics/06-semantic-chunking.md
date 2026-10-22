# Semantic Chunking
> **Level:** Intermediate · **Reading time:** 6 min · **Phase:** 1 — RAG Foundations

**In one line:** Semantic chunking starts a new chunk wherever the meaning of the text shifts, by comparing the embeddings of neighbouring sentences.

## Explain it like I'm new
Imagine listening to a podcast and writing chapter markers. You do not add a marker every five minutes. You add one when the hosts change topic, from "travel tips" to "camera gear". Semantic chunking places chunk borders the same way: where the topic changes, not where a character counter says so.

**Semantic** means "about meaning". To measure meaning the method uses [embeddings](02-embeddings.md): each sentence becomes a vector, and the similarity between neighbouring sentences tells you whether they talk about the same thing. When the similarity suddenly drops, that is a **breakpoint**, and a new chunk begins.

The result is chunks that each cover one idea. They match questions more cleanly and contain less unrelated text. The cost is that you must embed every sentence during ingestion, and chunk sizes become uneven.

## How it works
1. Split the document into sentences.
2. (Optional) Group each sentence with one neighbour on each side, so single short sentences do not swing the score.
3. Embed every sentence (or group).
4. Compute the similarity between each sentence and the next.
5. Mark a breakpoint where similarity falls below a threshold. Common choices: a fixed value, or a percentile of all the gaps (e.g. "the biggest 5% of drops").
6. Join the sentences between breakpoints into chunks. Optionally enforce a maximum size with a recursive split.

```text
sentence:     S1    S2    S3    S4    S5    S6
similarity:     0.81  0.77  0.12  0.85  0.79
                              ^ drop below threshold 0.5 -> breakpoint
chunks:       [ S1  S2  S3 ] [ S4  S5  S6 ]
```

## Code example
The demo in [`code/rag_chunkers.py`](../code/rag_chunkers.py) uses **Jaccard word overlap** (shared words ÷ all distinct words) instead of real embeddings, so it runs offline. Swap in a real embedding similarity in production.

```python
import re

def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

def jaccard(a, b):
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

def semantic_chunks(text, threshold=0.1, similarity=jaccard):
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks, current = [], [sentences[0]]
    for prev, sent in zip(sentences, sentences[1:]):
        if similarity(prev, sent) < threshold:      # topic changed
            chunks.append(" ".join(current)); current = [sent]
        else:
            current.append(sent)
    chunks.append(" ".join(current))
    return chunks

print(semantic_chunks("Cats purr softly. Cats sleep a lot. Kubernetes schedules pods."))
# ['Cats purr softly. Cats sleep a lot.', 'Kubernetes schedules pods.']
```

**In real projects** LangChain ships an experimental version (`pip install langchain-experimental langchain-huggingface sentence-transformers`):

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                           breakpoint_threshold_type="percentile")
chunks = splitter.split_text(long_text)
```

LlamaIndex has a similar `SemanticSplitterNodeParser`.

## When to use it / when not to
- Useful for long, loosely structured text where topics shift without headings: transcripts, meeting notes, essays.
- Less useful for well-structured docs. Headings already mark topics, and [recursive chunking](05-recursive-chunking.md) is cheaper.
- Measure it. Semantic chunking is not always better than a tuned recursive splitter.

| | Recursive | Semantic |
|---|---|---|
| Ingest cost | Very low | One embedding per sentence |
| Chunk sizes | Even | Uneven |
| Borders | Formatting | Topic shifts |

## Common mistakes
- No maximum size, so one long on-topic section becomes a giant chunk.
- A threshold tuned on one document and reused on very different ones.
- Bad sentence splitting (abbreviations like "e.g.", code, lists) producing junk "sentences".
- Expecting it to fix poor text extraction from PDFs.

## Interview questions
**1. How does semantic chunking decide where to cut?**
<details><summary>Answer</summary>
It embeds sentences, measures similarity between neighbours, and cuts where similarity drops below a threshold (fixed or percentile-based), meaning the topic changed.
</details>

**2. What are its downsides?**
<details><summary>Answer</summary>
Higher ingestion cost (embedding every sentence), uneven chunk sizes, sensitivity to the threshold and to sentence splitting, and no guarantee of better retrieval.
</details>

**3. Why group a sentence with its neighbours before embedding?**
<details><summary>Answer</summary>
Single short sentences give noisy embeddings. A small window smooths the signal so breakpoints reflect real topic changes, not one odd sentence.
</details>

## Go deeper
- LangChain (SemanticChunker in langchain-experimental) — https://python.langchain.com/
- LlamaIndex (SemanticSplitterNodeParser) — https://docs.llamaindex.ai/
- Reimers & Gurevych, *Sentence-BERT* (sentence embeddings) — https://arxiv.org/abs/1908.10084

**Prev:** [Recursive Chunking](05-recursive-chunking.md) · **Next:** [Parent Child Chunking](07-parent-child-chunking.md) · **Back to phase:** [Phase 1](../README.md)
