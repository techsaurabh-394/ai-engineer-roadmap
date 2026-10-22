"""A complete RAG pipeline in one readable file (~100 lines).

chunk -> embed (hashed TF-IDF) -> search -> build prompt -> answer with citations.
Runs offline. If ANTHROPIC_API_KEY is set, Claude writes the final answer; otherwise it prints the best passage.

Run:  python mini_rag.py "What is the multi-stage build for?"
"""
import hashlib
import math
import os
import re
import sys
from collections import Counter

DOCS = {
    "docker.txt": (
        "Multi-stage builds compile an application in one stage and copy only the final artifact into a small runtime image. "
        "This keeps production images small and reduces the attack surface. "
        "Docker Compose describes several services in one YAML file and starts them together."
    ),
    "caching.txt": (
        "An ETag identifies a version of a resource. When the resource has not changed the server replies with 304 Not Modified. "
        "The Cache-Control max-age directive sets how many seconds a response stays fresh."
    ),
    "auth.txt": (
        "Access tokens should be short lived. Refresh tokens belong in an httpOnly cookie and should be rotated on every use. "
        "Always verify the signature and the expiry claim of a JSON Web Token."
    ),
}
STOP = set("a an and are as at be by for from how in is it of on or that the this to was what when where which who why with do does".split())
DIM = 2048


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def chunk(text, size=300):
    """Split into sentences, then pack sentences into chunks of about `size` characters."""
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


def bucket(token):
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % DIM


class Index:
    def __init__(self, docs):
        self.items = [(name, c) for name, text in docs.items() for c in chunk(text)]
        df = Counter()
        self.vectors = []
        counts = [Counter(bucket(t) for t in tokens(c)) for _, c in self.items]
        for cnt in counts:
            df.update(cnt.keys())
        n = len(self.items)
        self.idf = {b: math.log((n + 1) / (d + 1)) + 1 for b, d in df.items()}
        self.vectors = [self._vec(cnt) for cnt in counts]

    def _vec(self, cnt):
        v = {b: (1 + math.log(c)) * self.idf.get(b, 1.0) for b, c in cnt.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {b: x / norm for b, x in v.items()}

    def search(self, query, k=2):
        q = self._vec(Counter(bucket(t) for t in tokens(query)))
        scored = [(sum(w * vec.get(b, 0.0) for b, w in q.items()), i) for i, vec in enumerate(self.vectors)]
        return [(s, *self.items[i]) for s, i in sorted(scored, reverse=True)[:k] if s > 0]


def answer(question, hits):
    if not hits:
        return "I could not find that in the documents."
    context = "\n".join(f"[{i}] ({name}) {text}" for i, (_, name, text) in enumerate(hits, 1))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic  # pip install anthropic

        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=400,
            system="Answer ONLY from the numbered sources and cite them like [1]. If they don't contain the answer, say so.",
            messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    _, name, text = hits[0]
    return f"[1] ({name}) {text}"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Why use multi-stage builds?"
    print(answer(q, Index(DOCS).search(q)))
