# Sparse Retrieval
> **Level:** Intermediate · **Reading time:** 6 min · **Phase:** 1 — RAG Foundations

**In one line:** Sparse retrieval represents text as very long vectors that are mostly zeros — one slot per vocabulary word — and finds documents by matching the words they share with the query.

## Explain it like I'm new
Imagine a giant checklist with every word in the dictionary, 50,000 lines long. To describe a document, you tick the words it contains and write how often. A short paragraph ticks maybe 40 lines; the other 49,960 stay empty. That checklist is a **sparse vector**: huge, but almost all zeros.

**Sparse** means "mostly zero". Because most entries are zero, the computer only stores the non-zero ones, like `{"refresh": 2, "token": 3, "cookie": 1}`.

To search quickly, sparse systems use an **inverted index**: for each word, a list of the documents containing it. It is the index at the back of a book: "token → pages 4, 12, 31". To answer a query, look up each query word, collect the documents, and score them with a formula like TF-IDF or [BM25](17-bm25.md).

The opposite is **[dense retrieval](19-dense-retrieval.md)**: short vectors (hundreds of numbers) where almost every entry is non-zero and no single slot means a specific word.

There is also **learned sparse retrieval** (for example SPLADE): a neural model outputs a sparse vector over the vocabulary, giving weight to related words that were not in the text (it might add "automobile" for a text about "car"). It keeps the inverted-index speed while adding some meaning.

## How it works
1. **Tokenize** every document (lowercase, split into words, maybe remove stop words and apply stemming).
2. **Build an inverted index**: `word → [(doc_id, count), ...]`. Store document lengths too.
3. At query time, tokenize the query the same way.
4. **Look up** each query word's list. Only documents in those lists can score above zero.
5. **Score** them (TF-IDF, BM25) and return the top k.

```text
Documents                      Inverted index
d0: "refresh token cookie"     refresh -> d0
d1: "access token short"       token   -> d0, d1
d2: "docker image layer"       cookie  -> d0
                               access  -> d1   ...
Query "token cookie": look up token -> {d0, d1}, cookie -> {d0}
score: d0 matches 2 terms, d1 matches 1  ->  d0 first
```

**TF-IDF**, the older cousin of BM25: `weight = tf × log(N / df)`. The existing [`code/mini_rag.py`](../code/mini_rag.py) uses a TF-IDF variant with hashing (it maps words into 2,048 buckets) — a sparse retriever in a few dozen lines.

## Code example
An inverted index with TF-IDF scoring, stdlib only:

```python
import math, re
from collections import Counter, defaultdict

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

docs = ["refresh token cookie", "access token short lived", "docker image layer"]
index = defaultdict(list)                     # word -> [(doc_id, tf)]
for i, d in enumerate(docs):
    for word, tf in Counter(tokenize(d)).items():
        index[word].append((i, tf))

def search(query, k=3):
    n, scores = len(docs), Counter()
    for word in tokenize(query):
        postings = index.get(word, [])
        if not postings:
            continue
        idf = math.log(n / len(postings)) + 1
        for doc_id, tf in postings:           # only touches docs containing the word
            scores[doc_id] += tf * idf
    return scores.most_common(k)

print(search("token cookie"))                 # d0 first
```

For BM25 scoring see [`code/rag_bm25.py`](../code/rag_bm25.py).

**In real projects** (scikit-learn TF-IDF, `pip install scikit-learn`):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)                   # scipy sparse matrix
scores = linear_kernel(vec.transform(["token cookie"]), X).ravel()
```

For production keyword search: Elasticsearch, OpenSearch, or Postgres full-text search.

## When to use it / when not to
- Exact matches matter: IDs, SKUs, error codes, legal citations, names.
- You need a fast, cheap, explainable baseline with no model.
- Data has domain words embedding models have never seen.
- Not enough alone when users paraphrase or use synonyms. Pair with dense in [hybrid search](16-hybrid-search.md).

| | Sparse | Dense |
|---|---|---|
| Vector | Vocabulary-size, mostly zeros | Hundreds of dims, all non-zero |
| Index | Inverted index | ANN index (HNSW, IVF) |
| Matches | Shared words | Similar meaning |
| Explainable | Yes, which words matched | Hard |

## Common mistakes
- Tokenizing queries differently from documents.
- Aggressive stop-word lists that delete meaningful words ("not", "no", or "IT" as in IT department).
- Missing stemming or lemmatization, so "tokens" never matches "token".
- Assuming sparse is "outdated". It is still a strong baseline and is often combined with dense retrieval.

## Interview questions
**1. Why are sparse vectors called sparse?**
<details><summary>Answer</summary>
They have one dimension per vocabulary term, so almost all values are zero for any given text. Only non-zero entries are stored.
</details>

**2. What is an inverted index and why is it fast?**
<details><summary>Answer</summary>
A map from each term to the list of documents containing it. A query only touches the lists for its own terms, skipping documents that share no words.
</details>

**3. What is learned sparse retrieval?**
<details><summary>Answer</summary>
A neural model (e.g. SPLADE) predicts term weights over the vocabulary, including related terms not in the text. It keeps inverted-index efficiency while adding some semantic matching.
</details>

## Go deeper
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* — https://dl.acm.org/doi/10.1561/1500000019
- Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking* — https://arxiv.org/abs/2107.05720

**Prev:** [BM25](17-bm25.md) · **Next:** [Dense Retrieval](19-dense-retrieval.md) · **Back to phase:** [Phase 1](../README.md)
