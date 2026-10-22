"""BM25 (sparse) search, a toy "dense" search, and hybrid fusion with RRF, stdlib only.

* BM25               - classic keyword ranking used by Elasticsearch, OpenSearch, Lucene
* toy_dense_scores   - stand-in for embedding search (character trigrams + cosine) so it runs offline
* reciprocal_rank_fusion - merge several ranked lists into one
* metadata filtering - drop documents that fail a filter BEFORE ranking

Run:  python rag_bm25.py "rotate refresh token"
"""
import math
import re
import sys
from collections import Counter

STOP = set("a an and are as at be by for from how in is it of on or the to what when where which why with".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    """Okapi BM25.

    score(q, d) = sum over query terms t of
        IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))
    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.tfs = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df

    def idf(self, term):
        d = self.df.get(term, 0)
        return math.log((self.n - d + 0.5) / (d + 0.5) + 1)

    def score(self, query, i):
        tf, length = self.tfs[i], self.lengths[i]
        s = 0.0
        for t in tokenize(query):
            f = tf.get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
            s += self.idf(t) * f * (self.k1 + 1) / denom
        return s

    def search(self, query, k=5, allowed=None):
        """Return [(score, doc_index)] best first. `allowed` = set of indexes that passed the filter."""
        ids = range(self.n) if allowed is None else sorted(allowed)
        scored = [(self.score(query, i), i) for i in ids]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:k]


def _trigrams(text):
    out = Counter()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        w = f"#{w}#"
        out.update(w[i:i + 3] for i in range(len(w) - 2))
    return out


def toy_dense_scores(query, texts, allowed=None):
    """Cosine similarity of character-trigram vectors.

    NOT a real embedding model. It only shows the *shape* of dense search: every
    document gets a similarity score, even with zero exact word overlap
    ("tokens" vs "token", "caching" vs "cache").
    """
    q = _trigrams(query)
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    ids = range(len(texts)) if allowed is None else sorted(allowed)
    out = []
    for i in ids:
        d = _trigrams(texts[i])
        dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
        out.append((sum(q[g] * d.get(g, 0) for g in q) / (qn * dn), i))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def reciprocal_rank_fusion(rankings, k=60):
    """Merge ranked lists of ids. score(id) = sum over lists of 1 / (k + rank), rank starts at 1."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def matches(meta, filters):
    """True if metadata satisfies every filter.

    filters: {"field": value}           -> equality
             {"field": [v1, v2]}        -> value in list
             {"field": ("gte", 2024)}   -> comparison: gte, lte, gt, lt
    """
    for field, cond in filters.items():
        if field not in meta:
            return False
        value = meta[field]
        if isinstance(cond, tuple):
            op, target = cond
            ok = {"gte": value >= target, "lte": value <= target,
                  "gt": value > target, "lt": value < target}.get(op)
            if ok is None:
                raise ValueError(f"unknown operator {op}")
            if not ok:
                return False
        elif isinstance(cond, list):
            if value not in cond:
                return False
        elif value != cond:
            return False
    return True


def hybrid_search(query, docs, k=3, filters=None, rrf_k=60):
    """docs: list of {"text": ..., "meta": {...}}. Filter first, then BM25 + dense, then RRF."""
    texts = [d["text"] for d in docs]
    allowed = None
    if filters:
        allowed = {i for i, d in enumerate(docs) if matches(d.get("meta", {}), filters)}
        if not allowed:
            return []
    sparse = [i for _, i in BM25(texts).search(query, k=len(texts), allowed=allowed)]
    dense = [i for s, i in toy_dense_scores(query, texts, allowed) if s > 0]
    return reciprocal_rank_fusion([sparse, dense], k=rrf_k)[:k]


DOCS = [
    {"text": "Refresh tokens belong in an httpOnly cookie and are rotated on every use.",
     "meta": {"source": "auth.md", "team": "security", "year": 2025}},
    {"text": "Access tokens should be short lived, for example 15 minutes.",
     "meta": {"source": "auth.md", "team": "security", "year": 2023}},
    {"text": "An ETag lets the server reply 304 Not Modified when nothing changed.",
     "meta": {"source": "http.md", "team": "platform", "year": 2024}},
    {"text": "Cache-Control max-age sets how many seconds a response stays fresh.",
     "meta": {"source": "http.md", "team": "platform", "year": 2025}},
    {"text": "Multi-stage Docker builds copy only the final artifact into a small image.",
     "meta": {"source": "docker.md", "team": "platform", "year": 2024}},
]

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "rotate refresh token"
    texts = [d["text"] for d in DOCS]
    print("BM25:  ", BM25(texts).search(q))
    print("dense: ", [(round(s, 3), i) for s, i in toy_dense_scores(q, texts)][:3])
    print("hybrid:", hybrid_search(q, DOCS))
    print("hybrid, year >= 2025:", hybrid_search(q, DOCS, filters={"year": ("gte", 2025)}))
