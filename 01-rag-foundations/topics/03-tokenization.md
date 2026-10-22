# Tokenization
> **Level:** Beginner · **Reading time:** 6 min · **Phase:** 1 — RAG Foundations

**In one line:** Tokenization is splitting text into small units called tokens (words or word pieces) that a model can turn into numbers.

## Explain it like I'm new
Think of LEGO. You cannot ship a built castle through a narrow tube, but you can ship the bricks and rebuild it. Models cannot read raw text. They read a sequence of brick IDs. A **tokenizer** breaks text into bricks (tokens) and maps each to an integer ID from a fixed **vocabulary**.

Tokens are often not whole words. Common words like "the" are one token. Rare words get split: "tokenization" might become "token" + "ization". This method is called **subword tokenization**. It keeps the vocabulary manageable (tens of thousands of entries) while still handling any word, including typos and new names.

Why should a RAG engineer care?
- **Limits are in tokens, not characters.** Embedding models and LLMs have maximum input sizes measured in tokens.
- **Cost is in tokens.** API pricing counts input and output tokens.
- **Chunk size is best set in tokens**, so chunks fit the embedding model.

A rough rule for English text with common tokenizers is about 4 characters per token, but code, numbers and other languages differ a lot. Always measure with the real tokenizer.

## How it works
Most modern models use **Byte-Pair Encoding (BPE)** or a close relative (WordPiece, Unigram).

1. Start with single characters (or bytes) as the vocabulary.
2. Count which pair of neighbouring symbols appears most often in a big training corpus.
3. Merge that pair into a new symbol and add it to the vocabulary.
4. Repeat until the vocabulary reaches the target size.
5. To tokenize new text, apply the learned merges in order.

```text
"lower lowest"   start:   l o w e r _ l o w e s t
merge (l,o)  ->  lo w e r _ lo w e s t
merge (lo,w) ->  low e r _ low e s t
merge (e,r)  ->  low er _ low e s t
tokens:          [low][er] [low][e][s][t]  -> IDs like [912, 263, 912, 68, 82, 83]
```

Different models use different tokenizers. The same sentence can be 10 tokens for one model and 14 for another.

## Code example
A tiny BPE trainer, stdlib only, to see merges happen:

```python
from collections import Counter

def train_bpe(words, num_merges):
    corpus = Counter(tuple(w) + ("_",) for w in words)     # "_" marks word end
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in corpus.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new = Counter()
        for word, freq in corpus.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1]); i += 2
                else:
                    out.append(word[i]); i += 1
            new[tuple(out)] += freq
        corpus = new
    return merges, corpus

merges, corpus = train_bpe("low low low lower lowest newest newest".split(), 5)
print(merges)          # [('l','o'), ('lo','w'), ...]
print(list(corpus))    # words as learned subword pieces
```

**In real projects** count tokens with the tokenizer your model uses. For OpenAI models (`pip install tiktoken`):

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("Refresh tokens belong in an httpOnly cookie.")
print(len(ids), ids[:5], enc.decode(ids))
```

For Hugging Face models, use `AutoTokenizer.from_pretrained(model_name)` from `transformers`.

## When to use it / when not to
- Measure tokens when you set chunk sizes, build prompts, or estimate cost.
- Use the tokenizer that matches the model. A token count from one tokenizer is only an estimate for another.
- For quick, rough budgeting a character estimate is fine. For hard limits, count real tokens.

## Common mistakes
- Sizing chunks in characters, then finding they exceed the embedding model's token limit and get cut off.
- Assuming one word equals one token.
- Forgetting that the system prompt, retrieved chunks *and* the answer all share the same context window.
- Using an English-based estimate for code, tables or non-English text.

## Interview questions
**1. Why do models use subword tokens instead of whole words?**
<details><summary>Answer</summary>
A whole-word vocabulary would be huge and still miss new words. Subwords keep the vocabulary small while any word can be built from pieces, so there are no unknown words.
</details>

**2. Why does tokenization matter in a RAG pipeline?**
<details><summary>Answer</summary>
Embedding and LLM limits, latency and cost are all in tokens. Chunk sizes and the number of retrieved chunks must fit the token budget, or text gets truncated.
</details>

**3. Can two models give different token counts for the same text?**
<details><summary>Answer</summary>
Yes. Each model family trains its own vocabulary and merges, so counts differ. Always count with the target model's tokenizer.
</details>

## Go deeper
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE) — https://arxiv.org/abs/1508.07909
- OpenAI tiktoken — https://github.com/openai/tiktoken

**Prev:** [Embeddings](02-embeddings.md) · **Next:** [Chunking](04-chunking.md) · **Back to phase:** [Phase 1](../README.md)
