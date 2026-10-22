"""Incremental indexing: only re-embed chunks whose content actually changed.

Each chunk gets a content hash (SHA-256 of its text). On re-ingest we compare
hashes with what is already stored:
  same hash      -> skip (no embedding call)
  new hash       -> embed and store
  missing hash   -> delete (the text was removed or edited)

Run:  python adv_incremental_index.py
"""
import hashlib


def chunk(text):
    """Split on blank lines (paragraphs). Real systems use smarter chunkers."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CountingEmbedder:
    """Fake embedding model that counts calls, so we can prove work was skipped."""

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[:8]]


class IncrementalIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = {}  # chunk_hash -> vector
        self.texts = {}  # chunk_hash -> text
        self.doc_chunks = {}  # doc_id -> list of chunk hashes (in order)

    def upsert(self, doc_id, text):
        """Index a new or changed document. Returns counts of what happened."""
        new_hashes = []
        added = 0
        for piece in chunk(text):
            h = content_hash(piece)
            new_hashes.append(h)
            if h not in self.vectors:
                self.vectors[h] = self.embedder.embed(piece)
                self.texts[h] = piece
                added += 1
        old = set(self.doc_chunks.get(doc_id, []))
        self.doc_chunks[doc_id] = new_hashes
        removed = self._garbage_collect(old - set(new_hashes))
        unchanged = len(new_hashes) - added
        return {"added": added, "unchanged": unchanged, "removed": removed}

    def delete(self, doc_id):
        old = set(self.doc_chunks.pop(doc_id, []))
        return self._garbage_collect(old)

    def _garbage_collect(self, candidates):
        """Drop vectors no document points to any more. Shared chunks survive."""
        still_used = {h for hashes in self.doc_chunks.values() for h in hashes}
        removed = 0
        for h in candidates:
            if h not in still_used and h in self.vectors:
                del self.vectors[h]
                del self.texts[h]
                removed += 1
        return removed


if __name__ == "__main__":
    emb = CountingEmbedder()
    idx = IncrementalIndex(emb)
    v1 = "Intro paragraph.\n\nInstall with pip.\n\nRun the server."
    v2 = "Intro paragraph.\n\nInstall with uv.\n\nRun the server."
    print("first ingest :", idx.upsert("readme", v1), "embed calls =", emb.calls)
    print("no change    :", idx.upsert("readme", v1), "embed calls =", emb.calls)
    print("one edit     :", idx.upsert("readme", v2), "embed calls =", emb.calls)
