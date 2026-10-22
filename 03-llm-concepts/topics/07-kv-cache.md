# KV Cache
> **Level:** Advanced · **Reading time:** 7 min · **Phase:** 3 · LLM Concepts

**In one line:** The KV cache stores the key and value vectors of tokens already processed, so each new token only computes its own instead of recomputing the whole prefix.

## Explain it like I'm new
Imagine writing a long email and, before typing each new word, re-reading the entire email from the start. That is what a naive model does: to pick token 101, it reprocesses tokens 1-100 again. Then for token 102, it re-reads 1-101.

A **KV cache** is like keeping notes. After reading a word once, you write down what you need about it. For the next word, you just read your notes plus the new word.

Why keys and values? In [self-attention](02-self-attention.md), the new token's **query** must be compared to the **keys** of all earlier tokens and then mix their **values**. Thanks to the [causal mask](05-decoder-only-models.md), earlier tokens never change when new ones are added, so their K and V stay the same. We can safely compute them once and keep them.

## How it works
1. **Prefill.** Run the whole prompt through the model in one parallel pass. Save K and V for every token, in every layer and every head.
2. **Decode step.** For the new token only: compute its Q, K, V. Append K and V to the cache.
3. Attend: the new Q against **all cached K**, mix **all cached V**.
4. Produce the next token, repeat from step 2.

**Counting work.** Generating `n` tokens:
- Without cache: step `t` computes K,V for `t` tokens, total `1 + 2 + … + n = n(n+1)/2`.
- With cache: one new K,V per step, total `n`.

| tokens n | without cache | with cache |
|---|---|---|
| 10 | 55 | 10 |
| 100 | 5,050 | 100 |
| 1,000 | 500,500 | 1,000 |

```text
step 1: [K1 V1]
step 2: [K1 V1][K2 V2]              <- only K2,V2 are new
step 3: [K1 V1][K2 V2][K3 V3]        <- only K3,V3 are new
         \___________ reused from cache ___________/
```

**The cost: memory.** Cache size per token is roughly:

```text
2 (K and V) × layers × kv_heads × head_dim × bytes_per_number
```

Example with made-up but realistic sizes: 32 layers, 8 KV heads, head_dim 128, 16-bit numbers (2 bytes): `2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB per token`. At 8,000 tokens that is about 1 GB for **one** conversation. That is why long contexts and many users are memory-bound, and why [GQA/MQA](03-multi-head-attention.md) exist.

**Related ideas:**
- **Prompt caching** (offered by API providers): reuses the cache for a repeated prompt prefix across requests, lowering cost and latency.
- **PagedAttention** (vLLM): stores the cache in blocks like OS memory pages, so many requests share GPU memory well.

## Code example
From [`llm_decoding.py`](../code/llm_decoding.py):

```python
def count_kv_computations(num_tokens, use_cache):
    computed = 0
    for t in range(1, num_tokens + 1):
        computed += 1 if use_cache else t   # cache: only the new token; no cache: whole prefix
    return computed

for n in (10, 100, 1000):
    print(n, count_kv_computations(n, False), count_kv_computations(n, True))
# 10 55 10
# 100 5050 100
# 1000 500500 1000
```

**In real projects** the Hugging Face `generate()` method uses the cache by default (`use_cache=True` in the model config). Serving engines such as vLLM manage it for you.

## When to use it / when not to
- **Always** for autoregressive generation. Every serious inference engine uses it.
- It does **not** speed up the prefill step much; it speeds up each decode step.
- Watch memory: long prompts × many concurrent users can exhaust GPU memory.
- Put stable content (system prompt, documents) **first** in prompts so provider prompt caching can reuse it.

## Common mistakes
- Thinking the cache stores outputs or tokens. It stores per-layer key and value tensors.
- Forgetting that the cache grows linearly with context length and batch size.
- Changing an early part of the prompt and expecting cache reuse. Any change invalidates everything after it.
- Confusing the KV cache (inside one generation) with a semantic/response cache (reusing whole answers).

## Interview questions
**1. Why can we cache K and V but not Q?**
<details><summary>Answer</summary>
With causal attention, earlier tokens' K and V never change when new tokens arrive, and future queries need them. A query is only used once, by its own token, at its own step, so there is nothing to reuse.
</details>

**2. What is the main downside of the KV cache?**
<details><summary>Answer</summary>
Memory. It grows linearly with sequence length, layers, KV heads, and batch size, and often limits how many requests or how long a context a GPU can serve.
</details>

**3. Name two techniques that reduce KV-cache memory.**
<details><summary>Answer</summary>
Grouped-query or multi-query attention (fewer K/V heads), quantizing the cache to fewer bits, paged memory management (PagedAttention), or sliding-window attention that drops old tokens.
</details>

## Go deeper
- [Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)

**Prev:** [Encoder Decoder Models](06-encoder-decoder-models.md) · **Next:** [Context Window](08-context-window.md) · **Back to phase:** [Phase 3](../README.md)
