"""A RAG pipeline split into the stages the site's "RAG pipeline lab" shows (standard library only).

question -> query processing -> embed -> vector search -> rerank -> context -> answer

Every stage is a small function that returns the data you would inspect in a debugger, so the lab
and the tests can look at the middle of the pipeline, not only at the end.

What is real and what is simplified:
  * REAL computation: sentence-packing chunker, TF-IDF vectors over a vocabulary, cosine similarity,
    top-k search, a reranker, prompt assembly. The numbers the lab shows are exactly these.
  * SIMPLIFIED: the "embedding" is TF-IDF (it knows words, not meaning: "car" and "automobile" are
    unrelated), and the "answer" is the best-matching sentence, because a real answer needs an LLM.
    mini_rag.py shows where an LLM call goes.

Run:  python rag_pipeline.py "How long does a refresh token last?"
"""
import math
import re
import sys
from collections import Counter

DOCS = [
    {"name": "auth.txt", "text": (
        "Access tokens should be short lived, for example fifteen minutes. "
        "Refresh tokens belong in an httpOnly cookie and should be rotated on every use. "
        "Always verify the signature and the expiry claim of a JSON Web Token.")},
    {"name": "docker.txt", "text": (
        "Multi-stage builds compile an application in one stage and copy only the final artifact into a small runtime image. "
        "This keeps production images small and reduces the attack surface. "
        "Docker Compose describes several services in one YAML file and starts them together.")},
    {"name": "caching.txt", "text": (
        "An ETag identifies a version of a resource. When the resource has not changed the server replies with 304 Not Modified. "
        "The Cache-Control max-age directive sets how many seconds a response stays fresh.")},
    {"name": "rag.txt", "text": (
        "Retrieval augmented generation looks up relevant passages first and gives them to the model together with the question. "
        "Chunks that are too large dilute the match, and chunks that are too small lose context. "
        "A reranker reads the question and each candidate together and reorders the top results.")},
    {"name": "agents.txt", "text": (
        "An agent repeats a loop: the model picks a tool, the program runs it, and the result goes back to the model. "
        "Always cap the number of steps so a confused agent cannot loop forever.")},
]
STOP = set("a an and are as at be by for from how in is it of on or that the this to was what when where which who why with do does".split())


# ------------------------------------------------------------------ stage 0: ingestion
def stem(w):
    """Crude plural stripping so "tokens" matches "token". Real systems use a proper stemmer or subword tokens."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def chunk(text, size=160):
    """Split into sentences, then pack sentences into chunks of at most about `size` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def build_index(docs, size=160):
    """Chunk every document, build the vocabulary and IDF table, and embed every chunk."""
    items = [{"doc": d["name"], "text": c} for d in docs for c in chunk(d["text"], size)]
    for i, it in enumerate(items):
        it["id"] = i
        it["tokens"] = tokens(it["text"])
    df = Counter()
    for it in items:
        df.update(set(it["tokens"]))
    vocab = sorted(df)
    n = len(items)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in vocab}
    index = {"items": items, "vocab": vocab, "idf": idf}
    for it in items:
        it["vector"] = embed(it["tokens"], index)
    return index


# ------------------------------------------------------------------ stage 1: query processing + embedding
def process_query(query):
    """Lowercase, split into words, drop stop words. Returns (all words, kept terms)."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    return words, tokens(query)


def embed(toks, index):
    """TF-IDF vector over the vocabulary, scaled to length 1. Unknown words are ignored."""
    counts = Counter(toks)
    v = [(1 + math.log(counts[t])) * index["idf"][t] if counts.get(t) else 0.0 for t in index["vocab"]]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ------------------------------------------------------------------ stage 2: vector search
def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))        # both vectors have length 1, so the dot product is the cosine


def search(query_vector, index, k=3):
    """[(cosine, chunk id)] best first. Chunks with score 0 share no word with the question and are dropped."""
    scored = [(cosine(query_vector, it["vector"]), it["id"]) for it in index["items"]]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:k]


# ------------------------------------------------------------------ stage 3: rerank
def coverage(terms, chunk_tokens):
    """Share of the question's distinct terms that appear in the chunk (0 to 1)."""
    uniq = sorted(set(terms))
    return sum(1 for t in uniq if t in set(chunk_tokens)) / len(uniq) if uniq else 0.0


def rerank(terms, index, hits, weight=0.5):
    """Re-score the candidates: weight * coverage + (1 - weight) * cosine. A teaching stand-in for a cross-encoder."""
    out = []
    for cos, cid in hits:
        cov = coverage(terms, index["items"][cid]["tokens"])
        out.append((weight * cov + (1 - weight) * cos, cid, cov, cos))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


# ------------------------------------------------------------------ stage 4: context + stage 5: answer
def build_context(index, ids):
    return "\n".join(f"[{n}] ({index['items'][i]['doc']}) {index['items'][i]['text']}" for n, i in enumerate(ids, 1))


def build_prompt(question, context):
    return ("Answer ONLY from the numbered sources and cite them like [1]. "
            "If they do not contain the answer, say so.\n\nSources:\n" + context + "\n\nQuestion: " + question)


def extractive_answer(terms, index, ids):
    """Stand-in for the LLM: return the sentence from the context that shares the most question terms."""
    best, best_key = None, None
    for n, i in enumerate(ids, 1):
        for s in re.split(r"(?<=[.!?])\s+", index["items"][i]["text"]):
            overlap = len(set(terms) & set(tokens(s)))
            key = (overlap, -n)
            if overlap and (best_key is None or key > best_key):
                best, best_key = f"{s} [{n}]", key
    return best or "I could not find that in the documents."


def run(query, docs=None, size=160, k=3, top=2):
    """Run every stage and return all intermediate data."""
    index = build_index(docs or DOCS, size)
    words, terms = process_query(query)
    qv = embed(terms, index)
    hits = search(qv, index, k)
    ranked = rerank(terms, index, hits)
    ids = [cid for _, cid, _, _ in ranked[:top]]
    context = build_context(index, ids)
    return {"index": index, "words": words, "terms": terms, "query_vector": qv, "hits": hits, "ranked": ranked,
            "context_ids": ids, "context": context, "prompt": build_prompt(query, context),
            "answer": extractive_answer(terms, index, ids) if ids else "I could not find that in the documents."}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How long does a refresh token last?"
    r = run(q)
    print(f"question: {q}")
    print(f"terms kept: {r['terms']}")
    print(f"{len(r['index']['items'])} chunks, vocabulary of {len(r['index']['vocab'])} words")
    for rank, (cos, cid) in enumerate(r["hits"], 1):
        print(f"search #{rank}: chunk {cid} ({r['index']['items'][cid]['doc']}) cosine={cos:.4f}")
    for rank, (score, cid, cov, cos) in enumerate(r["ranked"], 1):
        print(f"rerank #{rank}: chunk {cid} score={score:.4f} (coverage {cov:.2f}, cosine {cos:.4f})")
    print("answer:", r["answer"])
