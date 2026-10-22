# Context Compression

> **Level:** Intermediate · **Reading time:** 6 min · **Phase:** 2 — Advanced RAG

**In one line:** Context compression trims retrieved chunks down to only the parts that help answer the question, before they go into the prompt.

## Explain it like I'm new

You ask a friend, "What time does the pharmacy close on Sunday?" They hand you the pharmacy's entire 4-page brochure. The answer is in there, but you have to dig. A better friend highlights one line: "Sunday: 10:00–16:00."

Retrieved chunks are like the brochure. A 500-word chunk may contain one useful sentence. Sending all of it costs **tokens** (the units models read and bill by), adds latency, and gives the model more irrelevant text to get confused by. Research has also shown that models use information in the middle of long contexts less reliably than at the start or end.

Context compression is the highlighting step. It sits between retrieval and generation. Common methods:

- **Filtering:** drop whole chunks that score low for relevance.
- **Extraction:** keep only the sentences in a chunk that relate to the question.
- **Summarising / abstractive compression:** ask a model to rewrite the chunk shorter, focused on the question.
- **Token pruning:** methods like LLMLingua drop low-information tokens using a small language model.

## How it works

1. Retrieve top *k* chunks as usual.
2. Split each chunk into sentences.
3. Score each sentence against the question (embedding similarity, keyword overlap, or an LLM yes/no).
4. Keep sentences above a threshold, preserving original order. Drop chunks with nothing left.
5. Build the prompt from the compressed text, keeping the source ID for citations.

```text
chunk (5 sentences, 120 tokens)            compressed (1 sentence, 18 tokens)
┌─────────────────────────────┐            ┌────────────────────────────┐
│ Our pharmacy opened in 1998.│            │ Sunday: 10:00-16:00.       │
│ We stock 3,000 products.    │  ──score─▶ └────────────────────────────┘
│ Sunday: 10:00-16:00.        │            source: pharmacy.pdf p.2
│ Parking is free.            │
│ Ask about flu shots.        │
└─────────────────────────────┘
```

Worked token math: 5 chunks × 400 tokens = 2,000 context tokens. If compression keeps 20%, you send 400 tokens. That is 1,600 fewer input tokens on every request.

## Code example

A stdlib extractive compressor:

```python
import re

STOP = {"what", "time", "does", "the", "on", "a", "is", "of", "to", "in"}

def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}

def compress(question, chunk, min_overlap=1):
    q = words(question)
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    kept = [s for s in sentences if len(q & words(s)) >= min_overlap]
    return " ".join(kept)

chunk = "Our pharmacy opened in 1998. We stock 3,000 products. Sunday hours are 10:00-16:00. Parking is free."
print(compress("What time does the pharmacy close on Sunday?", chunk))
# -> "Our pharmacy opened in 1998. Sunday hours are 10:00-16:00."
```

Keyword overlap keeps the "pharmacy opened" sentence too, which shows why real systems use embedding similarity or an LLM judge for the score.

**In real projects** LangChain and LlamaIndex both provide compression or "node postprocessor" components that wrap a retriever and filter or extract from its output. Check the current docs for the exact class names, as they move between versions: [LangChain](https://python.langchain.com/), [LlamaIndex](https://docs.llamaindex.ai/).

## When to use it / when not to

- **Use it** when chunks are long and answers are short (FAQs, policies, manuals).
- **Use it** when token cost or latency matters, or you need to fit many sources into a small context window.
- **Avoid it** when the answer needs the surrounding context (step-by-step instructions, code, tables). Cutting sentences can break meaning.
- **Avoid LLM-based compression** on the hot path if you cannot afford an extra model call per chunk.

| Method | Cost | Risk |
|---|---|---|
| Chunk filtering | Low | Drops a chunk that was needed |
| Sentence extraction | Low–medium | Loses pronoun references ("it", "this") |
| LLM summarising | High | Can introduce errors or hallucinations |

## Common mistakes

- Losing the source ID during compression, so citations break.
- Compressing so hard that faithfulness drops. Measure answer quality, not just token savings.
- Re-ordering sentences, which can change meaning.
- Letting the compressor model "answer" instead of extract. Prompt it to copy text, not rewrite facts.

## Interview questions

**1. Why compress if modern models have huge context windows?**
<details><summary>Answer</summary>
Cost and latency scale with input tokens, and models can use information in long contexts less reliably, especially in the middle. Less irrelevant text usually means fewer distractions and cheaper, faster requests.
</details>

**2. Extractive vs abstractive compression?**
<details><summary>Answer</summary>
Extractive keeps original sentences or tokens, so it cannot invent facts and keeps citations exact. Abstractive rewrites into a summary, which can be shorter and smoother but may introduce errors.
</details>

**3. How do you know compression didn't hurt?**
<details><summary>Answer</summary>
Run the same golden set with and without it and compare faithfulness, answer correctness and context recall, along with token counts and latency.
</details>

## Go deeper

- [Lost in the Middle (Liu et al.)](https://arxiv.org/abs/2307.03172)
- [LLMLingua (Jiang et al.)](https://arxiv.org/abs/2310.05736)

**Prev:** [Bi Encoder](03-bi-encoder.md) · **Next:** [Query Expansion](05-query-expansion.md) · **Back to phase:** [Phase 2](../README.md)
