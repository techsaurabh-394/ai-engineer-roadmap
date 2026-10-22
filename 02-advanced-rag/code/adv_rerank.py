"""Toy bi-encoder retrieval followed by cross-encoder-style reranking.

Bi-encoder: query and document are turned into vectors SEPARATELY, then compared
with cosine similarity. Fast, because document vectors can be pre-computed.

Cross-encoder: query and document are scored TOGETHER as a pair. Slower, because
nothing can be pre-computed, but the scorer can look at how the two texts interact
(coverage of the query, exact phrases). Real cross-encoders are transformer models;
here we use hand-written pair features so the idea runs offline.

Run:  python adv_rerank.py
"""
import math
import re
from collections import Counter

STOP = set(
    "a an and are as at be by can do does for from how i in is it my of on or the to what when where which who why with you your".split()
)

DOCS = {
    "refund-faq": "Refund refund refund. Refund policy. Refund rules.",
    "cancel-guide": (
        "To cancel your subscription open Settings, choose Billing and press Cancel. "
        "A refund for the unused month is issued within five days. "
        "Invoices, payment methods and receipts are also listed under Billing."
    ),
    "shipping": "Orders ship within two days. Tracking numbers arrive by email.",
    "pricing": "The Pro subscription costs ten dollars per month and can be paused.",
}


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


# ---------- bi-encoder ----------
def embed(text):
    """Encode ONE text into a sparse vector (term counts). Stand-in for a dense embedding model."""
    return Counter(tokens(text))


def cosine(a, b):
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class BiEncoderIndex:
    def __init__(self, docs):
        # Document vectors are computed once, at index time. This is why bi-encoders scale.
        self.docs = docs
        self.vectors = {doc_id: embed(text) for doc_id, text in docs.items()}

    def search(self, query, k=3):
        q = embed(query)
        scored = [(cosine(q, v), doc_id) for doc_id, v in self.vectors.items()]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(round(s, 4), d) for s, d in scored[:k] if s > 0]


# ---------- cross-encoder (toy) ----------
def cross_score(query, doc):
    """Score a (query, doc) PAIR. Looks at both texts at once.

    coverage: share of distinct query terms that appear in the doc (rewards answering the whole question)
    phrase:   share of query bigrams that appear in order in the doc
    """
    q = tokens(query)
    d = tokens(doc)
    if not q or not d:
        return 0.0
    q_set, d_set = set(q), set(d)
    coverage = len(q_set & d_set) / len(q_set)
    q_bigrams = set(zip(q, q[1:]))
    d_bigrams = set(zip(d, d[1:]))
    phrase = len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0
    return round(0.8 * coverage + 0.2 * phrase, 4)


def rerank(query, candidates, docs, top_n=2):
    """Re-score bi-encoder candidates with the pair scorer and keep the best top_n."""
    rescored = [(cross_score(query, docs[doc_id]), doc_id) for _, doc_id in candidates]
    rescored.sort(key=lambda x: (-x[0], x[1]))
    return rescored[:top_n]


def retrieve_then_rerank(query, docs=DOCS, k=3, top_n=2):
    index = BiEncoderIndex(docs)
    candidates = index.search(query, k=k)
    return candidates, rerank(query, candidates, docs, top_n=top_n)


if __name__ == "__main__":
    question = "How do I cancel my subscription and get a refund?"
    first, final = retrieve_then_rerank(question)
    print("Bi-encoder top-k :", first)
    print("After rerank     :", final)
