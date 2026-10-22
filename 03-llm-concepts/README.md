# Phase 3 · LLM Concepts

**Goal:** understand what happens inside a large language model (LLM), from attention to the way each next token is picked, so you can predict its behaviour, choose the right settings, and debug bad outputs.

**Who it's for:** beginners who have built a simple RAG app (Phases 1-2) and now want to know how the model itself works. No machine-learning background needed.

**Prerequisites:**
- Basic Python (lists, loops, functions).
- High-school math: multiplication, averages, the idea of a probability.
- Helpful but optional: [Phase 1 · RAG Foundations](../01-rag-foundations/README.md) for embeddings and vectors.

**Time:** about 6-8 hours, including the code.

## Big idea
A language model reads **tokens** (pieces of words) and predicts the next token, over and over. Inside, a **transformer** lets every token look at the others (**attention**) to work out what it means. At the end, the model outputs a score for every token in its vocabulary; a **decoding strategy** (greedy, temperature, top-k, top-p, beam search) turns those scores into one chosen token. Everything else in this phase is detail around that loop.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | Transformer Architecture | Beginner | [01-transformer-architecture.md](topics/01-transformer-architecture.md) |
| 02 | Self Attention | Intermediate | [02-self-attention.md](topics/02-self-attention.md) |
| 03 | Multi Head Attention | Intermediate | [03-multi-head-attention.md](topics/03-multi-head-attention.md) |
| 04 | Positional Encoding | Intermediate | [04-positional-encoding.md](topics/04-positional-encoding.md) |
| 05 | Decoder Only Models | Intermediate | [05-decoder-only-models.md](topics/05-decoder-only-models.md) |
| 06 | Encoder Decoder Models | Intermediate | [06-encoder-decoder-models.md](topics/06-encoder-decoder-models.md) |
| 07 | KV Cache | Advanced | [07-kv-cache.md](topics/07-kv-cache.md) |
| 08 | Context Window | Beginner | [08-context-window.md](topics/08-context-window.md) |
| 09 | Temperature | Beginner | [09-temperature.md](topics/09-temperature.md) |
| 10 | Top-k | Beginner | [10-top-k.md](topics/10-top-k.md) |
| 11 | Top-p | Intermediate | [11-top-p.md](topics/11-top-p.md) |
| 12 | Beam Search | Intermediate | [12-beam-search.md](topics/12-beam-search.md) |
| 13 | Hallucination | Intermediate | [13-hallucination.md](topics/13-hallucination.md) |
| 14 | Prompt Engineering | Beginner | [14-prompt-engineering.md](topics/14-prompt-engineering.md) |

## Learning order
1. **How the model is built (01-06).** Start with the [Transformer Architecture](topics/01-transformer-architecture.md) overview, then zoom into attention (02-03), word order (04), and the two main model shapes (05-06).
2. **How it runs (07-08).** The [KV Cache](topics/07-kv-cache.md) explains why generation is fast; the [Context Window](topics/08-context-window.md) explains limits and cost.
3. **How it picks words (09-12).** Temperature, top-k, top-p, and beam search. Run the code while reading.
4. **How to use it well (13-14).** Why models make things up, and how to write prompts that work.

If you are short on time, read 01, 08, 09, 11, 13, 14 first. Come back to the rest later.

## Run the code
Everything is offline: standard-library Python, no API key, no downloads.

```bash
# from the repo root
python 03-llm-concepts/code/sampling_demo.py    # temperature + top-p on real-looking logits
python 03-llm-concepts/code/llm_attention.py    # attention, causal mask, multi-head split, positional encoding
python 03-llm-concepts/code/llm_decoding.py     # greedy vs beam vs top-k vs top-p, KV-cache cost

python -m pytest -q 03-llm-concepts/code        # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`sampling_demo.py`](code/sampling_demo.py) | softmax with temperature, top-p filter | 09, 11 |
| [`llm_attention.py`](code/llm_attention.py) | scaled dot-product attention, causal mask, multi-head split/merge, sinusoidal positions | 01-06 |
| [`llm_decoding.py`](code/llm_decoding.py) | greedy, top-k, top-p, temperature, beam search on a toy table, KV-cache counting | 05, 07, 09-12 |

## Try it
1. In `sampling_demo.py`, change `LOGITS` so "banana" has the highest score. What does the model now "believe"?
2. In `llm_decoding.py`, run `beam_search(1)`, `beam_search(2)`, `beam_search(3)`. When does beam beat greedy?
3. Change `top_p_filter(..., 0.9)` to `0.5`. Which tokens survive?
4. In `llm_attention.py`, remove the causal mask and compare the first row of the weights.
5. Compute the KV-cache memory for a model with 40 layers, 8 KV heads, head size 128, 2-byte numbers, and 32,000 tokens.

## Prompting, RAG or fine-tuning?
| Need | Use |
|---|---|
| Better instructions or format | Prompting (start here, it's cheapest) |
| Knowledge the model lacks (your documents, recent facts), with citations | RAG ([Phase 1](../01-rag-foundations/README.md), [Phase 2](../02-advanced-rag/README.md)) |
| A consistent style or behaviour, or a small model good at a narrow task | Fine-tuning |

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. In one sentence, what does self-attention compute?**
<details><summary>Answer</summary>
For each token, a weighted average of all tokens' value vectors, where the weights come from softmax(query · key / sqrt(d_k)).
</details>

**2. Why do transformers need positional encoding?**
<details><summary>Answer</summary>
Attention by itself ignores order: shuffling the input just shuffles the output. Position signals (sinusoidal, learned, or rotary) let the model tell word orders apart.
</details>

**3. What is the causal mask, and which models use it?**
<details><summary>Answer</summary>
It stops each position from attending to later positions. Decoder-only models use it everywhere; encoder-decoder models use it in the decoder's self-attention.
</details>

**4. What does the KV cache store, and what is its main cost?**
<details><summary>Answer</summary>
The key and value vectors of all previous tokens, per layer and head, so each new token only computes its own. The cost is GPU memory that grows linearly with context length and batch size.
</details>

**5. Your 8k-token model gets a 7,500-token prompt and max_tokens = 1,000. What happens?**
<details><summary>Answer</summary>
Input plus output (8,500) exceeds the window. The API errors or the output is cut short. Trim or summarize the input, retrieve less, or lower max_tokens.
</details>

**6. Temperature 0.2 vs 1.5: which for extracting invoice totals, and why?**
<details><summary>Answer</summary>
0.2 (or 0). Extraction needs the single most likely, exact answer; randomness only adds errors.
</details>

**7. How does top-p differ from top-k?**
<details><summary>Answer</summary>
Top-k keeps a fixed number of tokens. Top-p keeps the smallest set whose probabilities sum to p, so the set grows when the model is unsure and shrinks when it is confident.
</details>

**8. Why can beam search find a better sequence than greedy decoding?**
<details><summary>Answer</summary>
It keeps several partial sequences, so a token that looks worse now but leads to a much more probable continuation is not thrown away. In this phase's toy table, beam finds "a dog ran" (0.18) while greedy finds "the cat sat" (0.12).
</details>

**9. Name two causes of hallucination and two ways to reduce it.**
<details><summary>Answer</summary>
Causes: training optimizes for plausible text, not truth; missing or rare knowledge; no built-in lookup. Reductions: ground with RAG or tools, require and verify citations, allow "I don't know", evaluate on a labelled set.
</details>

**10. What is prompt injection, and why can't one clever prompt fully stop it?**
<details><summary>Answer</summary>
Untrusted text (web pages, emails, files) that contains instructions the model then follows. The model reads instructions and data as the same kind of tokens, so wording alone cannot guarantee separation. Defences must also live in code: least-privilege tools, validation, and human approval.
</details>

More practice: [interview prep · LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md).

**Prev:** [Phase 2 · Advanced RAG](../02-advanced-rag/README.md) · **Next:** [Phase 4 · Agentic AI](../04-agentic-ai/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
