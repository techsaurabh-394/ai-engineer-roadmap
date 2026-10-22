# Multi Head Attention
> **Level:** Intermediate · **Reading time:** 7 min · **Phase:** 3 · LLM Concepts

**In one line:** Multi-head attention runs several smaller attention operations side by side, so different heads can track different kinds of relationships at once.

## Explain it like I'm new
Imagine reviewing a contract with a team. One person checks names and dates, another checks money amounts, a third checks who promised what to whom. Each reads the same document but looks for different links. At the end, you combine their notes.

Multi-head attention works the same way. Instead of one big attention with vectors of width 512, the model splits the width into, say, 8 **heads** of width 64. Each head has its own Q, K, V weight matrices, so it can learn its own idea of "relevant". Research has found heads that, for example, attend to the previous token or link pronouns to nouns, but there is no fixed rule about what a head learns.

The cost is roughly the same as one big head, because each head works on a smaller slice.

## How it works
1. Start with token vectors `X` of shape `n × d_model`.
2. Project into Q, K, V with learned matrices (each `d_model × d_model`).
3. **Split** each of Q, K, V into `h` heads: shape `n × (d_model / h)` per head.
4. Run [scaled dot-product attention](02-self-attention.md) in each head **independently**.
5. **Concatenate** the head outputs back to `n × d_model`.
6. Multiply by an output matrix `W_o` to mix information across heads.

```text
X (3 tokens x 4 features)
        | split width into 2 heads
   +----+----+
 head 1    head 2        each: 3 x 2
 attention attention     (own Q,K,V)
   +----+----+
        | concat -> 3 x 4
        | x W_o
      output (3 x 4)
```

**Numbers from the original paper:** `d_model = 512`, `h = 8`, so each head has `d_k = d_v = 64`.

**Variants you will see in modern LLMs:**
- **Multi-Query Attention (MQA):** all heads share one K and one V. Smaller [KV cache](07-kv-cache.md), faster decoding.
- **Grouped-Query Attention (GQA):** groups of query heads share K and V. A middle ground used by many open models.

## Code example
From [`llm_attention.py`](../code/llm_attention.py). Q = K = V = X here (no learned projections) so the split is easy to see.

```python
from llm_attention import split_heads, merge_heads, multi_head_attention

X = [[1.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0, 0.0]]

heads = split_heads(X, num_heads=2)
print(heads[0])   # [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  -> first 2 features of each token
print(heads[1])   # [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]  -> last 2 features
assert merge_heads(heads) == X

out, weights = multi_head_attention(X, num_heads=2)
print(len(weights))  # 2 separate attention tables, one per head
```

Notice the heads see different features and so produce different attention tables. That is the whole point.

**In real projects** (PyTorch):

```python
import torch
mha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 512)          # (batch, tokens, features)
out, attn_weights = mha(x, x, x)     # self-attention
print(out.shape)                     # torch.Size([2, 10, 512])
```

## When to use it / when not to
- Multi-head is the standard. You rarely choose "single head" in practice.
- More heads with the same width means smaller heads. Very small heads can lose expressive power.
- For inference memory, MQA or GQA reduce cache size with modest quality impact. This is a model-design choice, usually already made for you.

| Variant | K/V heads | KV cache size | Typical use |
|---|---|---|---|
| Multi-head (MHA) | h | largest | original transformer |
| Grouped-query (GQA) | g (1 < g < h) | medium | many current open LLMs |
| Multi-query (MQA) | 1 | smallest | fast decoding |

## Common mistakes
- Choosing `d_model` not divisible by the number of heads. The split fails.
- Forgetting the output projection `W_o`. Without it, heads cannot share information.
- Assuming each head has a clear, human-readable job. Some do, many don't.
- Thinking more heads always means better quality. It is a trade-off.

## Interview questions
**1. Why use multiple heads instead of one big one?**
<details><summary>Answer</summary>
Each head has its own projections, so different heads can attend to different positions and relationships at the same time. One head would average them into a single pattern. The compute cost is about the same because each head is narrower.
</details>

**2. What do GQA and MQA change, and why?**
<details><summary>Answer</summary>
They let several query heads share the same key and value heads (all heads in MQA, groups in GQA). This shrinks the KV cache and memory bandwidth during decoding, which speeds up generation.
</details>

**3. If d_model is 4096 and there are 32 heads, what is the head dimension?**
<details><summary>Answer</summary>
4096 / 32 = 128.
</details>

## Go deeper
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150), multi-query attention
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)

**Prev:** [Self Attention](02-self-attention.md) · **Next:** [Positional Encoding](04-positional-encoding.md) · **Back to phase:** [Phase 3](../README.md)
