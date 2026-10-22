# Self Attention
> **Level:** Intermediate · **Reading time:** 8 min · **Phase:** 3 · LLM Concepts

**In one line:** Self-attention lets each token build its new meaning as a weighted average of all the tokens in the same sentence, with weights based on how relevant each one is.

## Explain it like I'm new
Think of a library search. You walk in with a **question** ("books about rivers"). Every book has a **label** on its spine. You compare your question to each label, and the better the match, the more of that book you read. What you walk out with is a blend of the books' **contents**, weighted by match.

Self-attention does exactly this for every token:
- **Query (Q):** what this token is looking for.
- **Key (K):** what each token offers, used for matching.
- **Value (V):** the information each token actually passes along.

"Self" means the queries, keys, and values all come from the same sequence. In real models, Q, K, and V are produced by multiplying each token's vector by three learned weight matrices (`W_q`, `W_k`, `W_v`). Training adjusts these matrices so useful words attend to each other, for example "it" attending to the noun it refers to.

## How it works
The formula from *Attention Is All You Need*:

```text
Attention(Q, K, V) = softmax( Q · Kᵀ / sqrt(d_k) ) · V
```

1. **Score.** Dot product of each query with each key gives an `n × n` table of raw scores. A higher dot product means "more similar".
2. **Scale.** Divide by `sqrt(d_k)` (the key size). Without this, scores grow with vector size, softmax becomes nearly one-hot, and gradients shrink.
3. **Mask (optional).** Add `-inf` where a token must not look, such as future tokens in a decoder (a **causal mask**). After softmax those weights become 0.
4. **Softmax** each row so the weights are positive and sum to 1.
5. **Mix.** Multiply the weights by V. Each output row is a weighted average of value vectors.

```text
         keys:  k1    k2    k3
query q1  ->  [0.51  0.19  0.31]   row sums to 1
query q2  ->  [0.19  0.51  0.31]
query q3  ->  [0.27  0.27  0.45]
output_i = sum_j  weight[i][j] * v_j
```

**Tiny worked example.** One query `q = [1, 0]`, two keys `k1 = [1, 0]`, `k2 = [0, 1]`, values `v1 = [10, 0]`, `v2 = [0, 10]`.
- Scores: `q·k1 = 1`, `q·k2 = 0`. Scale by `sqrt(2) ≈ 1.414`: `[0.707, 0]`.
- Softmax: `e^0.707 ≈ 2.03`, `e^0 = 1`, so weights `≈ [0.67, 0.33]`.
- Output: `0.67·[10,0] + 0.33·[0,10] = [6.7, 3.3]`. The query "leans" toward the value whose key matched.

The test `test_attention_hand_computed_example` in [`test_llm_attention.py`](../code/test_llm_attention.py) checks this exact number.

## Code example
```python
import math

def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    return [v / sum(e) for v in e]

def attention(Q, K, V):
    d_k = len(K[0])
    out = []
    for q in Q:
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d_k) for k in K]
        w = softmax(scores)
        out.append([sum(w[j] * V[j][c] for j in range(len(V))) for c in range(len(V[0]))])
    return out

print(attention([[1, 0]], [[1, 0], [0, 1]], [[10, 0], [0, 10]]))  # [[6.69..., 3.30...]]
```

The full version with masks lives in [`llm_attention.py`](../code/llm_attention.py): run `python 03-llm-concepts/code/llm_attention.py`.

**In real projects** (PyTorch 2.x has a fused, fast kernel):

```python
import torch
import torch.nn.functional as F

q = k = v = torch.randn(1, 3, 4)  # (batch, tokens, features)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## When to use it / when not to
- Self-attention is the default way to mix information across tokens in modern models.
- Cost is `O(n²)` in time and memory for `n` tokens. For very long inputs, people use tricks such as sliding-window attention, KV caching, and fused kernels (for example FlashAttention).
- For tiny fixed-size inputs, a plain MLP may be enough.

| Term | Shape for n tokens, width d |
|---|---|
| Q, K, V | n × d_k (or d_v) |
| Score table | n × n |
| Output | n × d_v |

## Common mistakes
- Forgetting to scale by `sqrt(d_k)`. Training becomes unstable.
- Applying softmax over the wrong axis. Each **query row** must sum to 1.
- Masking by setting scores to 0 instead of `-inf`. A score of 0 still gets weight after softmax.
- Mixing up attention weights with "explanations". Weights hint at information flow but are not proof of reasoning.

## Interview questions
**1. Why divide by sqrt(d_k)?**
<details><summary>Answer</summary>
Dot products of random vectors grow in size with dimension. Large scores push softmax into regions where one weight is near 1 and the gradient is near 0. Scaling keeps the scores in a range where softmax still learns.
</details>

**2. What is the time complexity of self-attention, and why?**
<details><summary>Answer</summary>
O(n² · d): every one of n queries is compared with every one of n keys, each comparison costing d operations. Memory for the score table is O(n²).
</details>

**3. What is the difference between self-attention and cross-attention?**
<details><summary>Answer</summary>
In self-attention Q, K, and V all come from the same sequence. In cross-attention the queries come from one sequence (for example the decoder) and the keys and values from another (the encoder output).
</details>

## Go deeper
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473), the earlier attention idea

**Prev:** [Transformer Architecture](01-transformer-architecture.md) · **Next:** [Multi Head Attention](03-multi-head-attention.md) · **Back to phase:** [Phase 3](../README.md)
