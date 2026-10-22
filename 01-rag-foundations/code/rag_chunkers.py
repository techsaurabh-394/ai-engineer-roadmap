"""Chunking strategies for RAG, stdlib only.

Five chunkers you will meet in real projects:

* fixed_size_chunks     - cut every N characters (optionally with overlap)
* recursive_chunks      - try big separators first (paragraphs), then smaller ones
* sliding_window_chunks - windows of N words that move forward by `stride` words
* semantic_chunks       - start a new chunk when the topic changes (toy word-overlap version)
* parent_child_chunks   - small "child" chunks for search, big "parent" chunks for context

Run:  python rag_chunkers.py
"""
import re

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def fixed_size_chunks(text, size=200, overlap=0):
    """Cut text into pieces of `size` characters. Neighbours share `overlap` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
    return chunks


def recursive_chunks(text, size=200, separators=DEFAULT_SEPARATORS):
    """Split on the biggest separator that works, recursing into pieces that are still too big.

    Small neighbouring pieces are merged back together so chunks are as full as possible
    without going over `size` characters.
    """
    if len(text) <= size:
        return [text] if text.strip() else []
    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        # Last resort: hard cut by characters.
        return [text[i:i + size] for i in range(0, len(text), size)]
    if sep not in text:
        return recursive_chunks(text, size, rest)

    parts = text.split(sep)
    # Put the separator back on every piece except the last, so no text is lost.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

    chunks, current = [], ""
    for piece in pieces:
        if len(piece) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(recursive_chunks(piece, size, rest))
        elif len(current) + len(piece) <= size:
            current += piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def sliding_window_chunks(text, window=50, stride=25):
    """Windows of `window` words. Each window starts `stride` words after the previous one."""
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    words = text.split()
    if not words:
        return []
    chunks = []
    for start in range(0, len(words), stride):
        chunks.append(" ".join(words[start:start + window]))
        if start + window >= len(words):
            break
    return chunks


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    """Share of words two texts have in common (0 = none, 1 = identical sets)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def semantic_chunks(text, threshold=0.1, similarity=jaccard):
    """Start a new chunk when a sentence looks unrelated to the previous one.

    Real systems compare sentence *embeddings*. Here `similarity` defaults to word
    overlap (Jaccard) so the idea runs offline.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks, current = [], [sentences[0]]
    for prev, sent in zip(sentences, sentences[1:]):
        if similarity(prev, sent) < threshold:
            chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    chunks.append(" ".join(current))
    return chunks


def parent_child_chunks(text, parent_size=400, child_size=100):
    """Return (parents, children).

    parents:  {parent_id: parent_text}
    children: [{"id": ..., "parent_id": ..., "text": ...}, ...]
    Search over children (precise), then hand the parent (more context) to the LLM.
    """
    if child_size > parent_size:
        raise ValueError("child_size must be <= parent_size")
    parents, children = {}, []
    for p_idx, parent in enumerate(recursive_chunks(text, parent_size)):
        pid = f"p{p_idx}"
        parents[pid] = parent
        for c_idx, child in enumerate(recursive_chunks(parent, child_size)):
            children.append({"id": f"{pid}-c{c_idx}", "parent_id": pid, "text": child})
    return parents, children


SAMPLE = (
    "Docker images are built in layers. Each instruction in a Dockerfile adds a layer.\n\n"
    "Multi-stage builds use several FROM lines. Only the final stage ends up in the shipped image, "
    "so the image stays small.\n\n"
    "Refresh tokens should live in an httpOnly cookie. Rotate refresh tokens on every use."
)

if __name__ == "__main__":
    print("fixed (60, overlap 10):")
    for c in fixed_size_chunks(SAMPLE, 60, 10):
        print("  |", repr(c))
    print("\nrecursive (120):")
    for c in recursive_chunks(SAMPLE, 120):
        print("  |", repr(c))
    print("\nsliding window (12 words, stride 6):")
    for c in sliding_window_chunks(SAMPLE, 12, 6):
        print("  |", c)
    print("\nsemantic (word-overlap threshold 0.1):")
    for c in semantic_chunks(SAMPLE, 0.1):
        print("  |", c)
    parents, children = parent_child_chunks(SAMPLE, 200, 80)
    print("\nparent-child:")
    for ch in children:
        print(f"  {ch['id']:8} -> parent {ch['parent_id']}: {ch['text']!r}")
