# Encoder Decoder Models
> **Level:** Intermediate · **Reading time:** 8 min · **Phase:** 3 · LLM Concepts

**In one line:** An encoder-decoder model first reads the whole input with an encoder, then writes the output with a decoder that looks back at the encoded input through cross-attention.

**Before you start:** [Self Attention](02-self-attention.md) and [Decoder Only Models](05-decoder-only-models.md).

**Words to know:**
- **Encoder:** the half of the transformer that reads the whole input and turns it into rich vectors.
- **Decoder:** the half that writes the output one token at a time.
- **Cross-attention:** attention where queries come from the decoder and keys and values come from the encoder.
- **Bidirectional:** each token can look both left and right.
- **Sequence-to-sequence (seq2seq):** a task that maps an input sequence to a different output sequence, such as translation.

## Explain it like I'm new
Think of a human interpreter at a meeting. First they listen to the full sentence in French (the **encoder** step: understand everything, in both directions). Then they speak the sentence in English word by word (the **decoder** step), glancing back at their notes of the French sentence as they go. Those glances are **cross-attention**.

This was the original transformer design from *Attention Is All You Need*, built for machine translation. Popular examples are **T5** (which frames every task as "text in, text out") and **BART**. Speech recognition models such as **Whisper** also use this shape: audio in, text out.

The big difference from [decoder-only models](05-decoder-only-models.md): the input gets **bidirectional** attention (every input token sees every other input token), and the input and output are separate sequences.

<!-- anim:06-encoder-decoder-models -->
## Watch it
![Encoder-decoder models: One part reads the input, another writes the output.](../../assets/anim/06-encoder-decoder-models.svg)

*It plays on a loop. Read the caption in the box, one step at a time. Want to go at your own pace? [Step through it with Previous and Next](https://techsaurabh-394.github.io/ai-engineer-roadmap/#06-encoder-decoder-models).*

## How it works
1. **Encoder.** The input tokens go through N layers of self-attention **without** a causal mask, plus feed-forward layers. Output: one context-rich vector per input token.
2. **Decoder.** Generates the output one token at a time. Each decoder layer has three parts:
   - **Masked self-attention** over output tokens produced so far (causal).
   - **Cross-attention:** queries from the decoder, keys and values from the encoder output.
   - **Feed-forward** network.
3. The decoder's final vector is turned into next-token probabilities, a token is chosen, and the loop repeats.
4. The encoder runs **once** per input; only the decoder loops.

```mermaid
flowchart LR
    subgraph Encoder
      I[input tokens] --> E[self-attention, no mask<br/>+ feed-forward] --> H[encoded input]
    end
    subgraph Decoder
      O[output so far] --> D1[masked self-attention]
      D1 --> D2[cross-attention]
      H --> D2
      D2 --> D3[feed-forward] --> N[next token]
    end
    N -- append --> O
```

### Worked example
The code example below uses 3 encoded input tokens and 1 output token so far:
1. The encoder has already turned the input into `[1,0]`, `[0,1]`, `[0.5,0.5]`.
2. The decoder's current state is `[0.9, 0.1]`, which acts as the query.
3. The query is compared to each encoded input token. Token 1 is most similar, so the weights are about `[0.43, 0.24, 0.32]` (they add up to 1, give or take rounding).
4. The result is the weighted blend of the inputs: `[0.59, 0.41]`. This blend flows into the decoder's feed-forward layer.
5. At the next output token the decoder asks again, with a new query, against the same encoder output. The encoder does not run again.

**Training** is like the decoder-only case but the loss is only on the output tokens: given input X, predict output Y token by token. This is also called **sequence-to-sequence (seq2seq)** learning.

## Code example
Cross-attention is the same attention function with Q from one sequence and K, V from another. Uses [`llm_attention.py`](../code/llm_attention.py).

```python
from llm_attention import scaled_dot_product_attention

encoder_out = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]   # 3 input tokens, already encoded
decoder_state = [[0.9, 0.1]]                          # 1 output token so far

context, weights = scaled_dot_product_attention(
    q=decoder_state, k=encoder_out, v=encoder_out
)
print([round(w, 2) for w in weights[0]])  # which input tokens the decoder "looks at"
print(context)                            # blended information passed to the next layer
```

**In real projects** (Hugging Face Transformers, T5):

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
ids = tok("translate English to German: The house is small.", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=20)[0], skip_special_tokens=True))
```

## When to use it / when not to
- **Good for** tasks with a clear input → output mapping: translation, summarization, speech-to-text, structured rewriting.
- Encoding the input once can be efficient when the input is long and the output is short.
- **Less common** for general chat. Large decoder-only models handle these tasks well with prompting, and the ecosystem (tools, serving, fine-tuning recipes) centres on them.

| | Encoder-decoder | Decoder-only |
|---|---|---|
| Input attention | bidirectional | causal |
| Input/output | separate sequences | one shared sequence |
| Extra layer | cross-attention | none |
| Typical examples | T5, BART, Whisper | GPT, Llama, Claude |

## Common mistakes
- Saying cross-attention and self-attention are different math. The math is the same; only the source of Q vs K/V differs.
- Thinking the encoder re-runs every output step. It runs once; its output is reused.
- Calling BERT an encoder-decoder. BERT is encoder-only.
- Assuming encoder-decoder is "outdated". It is still a strong choice for seq2seq tasks such as speech recognition.

## Recap
If you remember only three things:
- The encoder reads the whole input once, in both directions. The decoder then writes the output one token at a time.
- Cross-attention is ordinary attention where the queries come from the decoder and the keys and values come from the encoder.
- Only the decoder's own self-attention is causal. Most general chat LLMs are decoder-only, but encoder-decoder models still suit translation, summarization and speech-to-text.

## Interview questions
**1. What does cross-attention do?**
<details><summary>Answer</summary>
It lets each decoder position query the encoder's outputs. Queries come from the decoder; keys and values come from the encoder. This is how the output stays tied to the input.
</details>

**2. Which parts use a causal mask in an encoder-decoder model?**
<details><summary>Answer</summary>
Only the decoder's self-attention. The encoder's self-attention and the cross-attention are not causally masked (padding masks may still apply).
</details>

**3. Why have decoder-only models become more common for general LLMs?**
<details><summary>Answer</summary>
They are simpler (one stack, one sequence), scale well with next-token prediction on raw text, and can handle input→output tasks by putting the input in the prompt. The tooling around them is also more mature.
</details>

## Go deeper
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5, Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)

**Prev:** [Decoder Only Models](05-decoder-only-models.md) · **Next:** [KV Cache](07-kv-cache.md) · **Back to phase:** [Phase 3](../README.md)
