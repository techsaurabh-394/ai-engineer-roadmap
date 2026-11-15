"""A tiny multimodal RAG, stdlib only.

The corpus mixes text chunks and "images". An image is stored as an id, a
caption, and a toy feature vector (standing in for a CLIP-style embedding).
We retrieve by caption word overlap (BM25), by feature-vector cosine, and fuse
the two rankings with reciprocal rank fusion. Then we build a prompt that
refers to the chosen images by id.

Run:  python mm_rag.py
"""
import math
import re

# Feature axes (toy): [chart-ness, diagram-ness, photo-ness, table-ness]
CORPUS = [
    {"id": "t1", "kind": "text", "caption": "Revenue grew steadily in the third quarter across all regions.", "features": None},
    {"id": "t2", "kind": "text", "caption": "Disconnect the battery before any electrical work.", "features": None},
    {"id": "img1", "kind": "image", "caption": "Bar chart of quarterly revenue by region.", "features": [0.9, 0.1, 0.0, 0.2]},
    {"id": "img2", "kind": "image", "caption": "Exploded view of the pump housing.", "features": [0.1, 0.9, 0.1, 0.0]},
    {"id": "img3", "kind": "image", "caption": "Figure 4.", "features": [0.8, 0.2, 0.0, 0.3]},
    {"id": "img4", "kind": "image", "caption": "Team photo at the office offsite.", "features": [0.0, 0.0, 1.0, 0.0]},
]

QUERY_TEXT = "how did revenue change by quarter"
QUERY_FEATURES = [1.0, 0.1, 0.0, 0.2]   # what a text encoder might say "a revenue chart" looks like


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query, docs, k1=1.5, b=0.75):
    """Classic BM25 over caption/text. Returns one score per doc."""
    toks = [tokenize(d) for d in docs]
    n = len(docs)
    avg = sum(len(t) for t in toks) / n
    scores = []
    for t in toks:
        s = 0.0
        for term in set(tokenize(query)):
            df = sum(1 for d in toks if term in d)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_by_caption(query, items):
    scores = bm25_scores(query, [it["caption"] for it in items])
    pairs = [(it["id"], s) for it, s in zip(items, scores) if s > 0]
    return sorted(pairs, key=lambda p: (-p[1], p[0]))


def rank_by_vector(query_features, items):
    pairs = [(it["id"], cosine(query_features, it["features"])) for it in items if it["features"]]
    return sorted(pairs, key=lambda p: (-p[1], p[0]))


def rrf(rankings, k=60):
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    fused = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda p: (-p[1], p[0]))


def build_prompt(question, hits, items):
    by_id = {it["id"]: it for it in items}
    lines = ["Answer the question using only the sources below. Cite source ids in [brackets].", ""]
    for doc_id in hits:
        it = by_id[doc_id]
        if it["kind"] == "image":
            lines.append(f"[{doc_id}] (image attached) caption: {it['caption']}")
        else:
            lines.append(f"[{doc_id}] {it['caption']}")
    lines += ["", f"Question: {question}"]
    return "\n".join(lines)


def attached_images(hits, items):
    kinds = {it["id"]: it["kind"] for it in items}
    return [h for h in hits if kinds[h] == "image"]


def show(title, ranking):
    print(title)
    for doc_id, score in ranking:
        print(f"  {doc_id:<5} {score:.4f}")


def main():
    show("1. Caption search (BM25 on text and captions):", rank_by_caption(QUERY_TEXT, CORPUS))
    show("2. Feature-vector search (cosine, images only):", rank_by_vector(QUERY_FEATURES, CORPUS))
    fused = rrf([rank_by_caption(QUERY_TEXT, CORPUS), rank_by_vector(QUERY_FEATURES, CORPUS)])
    show("3. Reciprocal rank fusion (k=60):", fused)
    top = [d for d, _ in fused[:3]]
    print("\nTop 3 after fusion:", top, "| images to attach:", attached_images(top, CORPUS))
    print("\nPrompt:\n" + build_prompt(QUERY_TEXT, top, CORPUS))


if __name__ == "__main__":
    main()
