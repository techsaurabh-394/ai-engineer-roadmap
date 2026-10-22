"""Query transformations: expansion, multi-query + RRF, step-back and HyDE.

Everything runs offline. `FakeLLM` stands in for a real model: it returns canned,
deterministic text so you can see the *shape* of each technique.

Run:  python adv_query_transform.py
"""
import re
from collections import Counter

STOP = set("a an and are as at be by can do does for from how i in is it my of on or the to what when where which who why with".split())

CORPUS = {
    "d1": "Kubernetes restarts a container when its liveness probe fails.",
    "d2": "A pod that keeps crashing enters the CrashLoopBackOff state with growing back-off delays.",
    "d3": "Readiness probes decide whether a pod receives traffic from a Service.",
    "d4": "Container images are pulled from a registry such as Docker Hub.",
    "d5": "Kubernetes is a system for running and scheduling containers across many machines.",
}

SYNONYMS = {
    "crash": ["crashing", "crashloopbackoff", "restart"],
    "crashes": ["crashing", "crashloopbackoff", "restart"],
    "pod": ["container"],
    "k8s": ["kubernetes"],
}


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def keyword_search(query, corpus=CORPUS, k=3):
    """Score = number of query-term occurrences in the doc. Returns ranked doc ids."""
    q = set(tokens(query))
    scored = []
    for doc_id, text in corpus.items():
        counts = Counter(tokens(text))
        score = sum(counts[t] for t in q)
        if score:
            scored.append((score, doc_id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [d for _, d in scored[:k]]


# ---------- 1. query expansion ----------
def expand_query(query, synonyms=SYNONYMS):
    """Append synonyms for any known term. Original words stay first."""
    words = tokens(query)
    extra = []
    for w in words:
        for s in synonyms.get(w, []):
            if s not in words and s not in extra:
                extra.append(s)
    return " ".join(words + extra)


# ---------- fake LLM ----------
class FakeLLM:
    """Deterministic stand-in for a chat model. Records every prompt it sees."""

    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Write 3 different search queries"):
            return "why does my pod restart\nCrashLoopBackOff meaning\nliveness probe fails container restart"
        if prompt.startswith("Write a more general"):
            return "How does Kubernetes manage container health and restarts?"
        if prompt.startswith("Write a short passage"):
            return "A pod that keeps crashing goes into CrashLoopBackOff; Kubernetes restarts the container with back-off delays."
        return "I don't know."


# ---------- 2. multi-query + reciprocal rank fusion ----------
def multi_query_prompt(question, n=3):
    return f"Write {n} different search queries that would help answer this question, one per line.\nQuestion: {question}"


def rrf_merge(rankings, k=60):
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank)."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def multi_query_retrieve(question, llm, corpus=CORPUS, k=3):
    variants = [line.strip() for line in llm.complete(multi_query_prompt(question)).splitlines() if line.strip()]
    queries = [question] + variants
    rankings = [keyword_search(q, corpus, k=k) for q in queries]
    return queries, rrf_merge(rankings)[:k]


# ---------- 3. step-back prompting ----------
def step_back_prompt(question):
    return (
        "Write a more general, higher-level question whose answer would help answer the question below.\n"
        f"Question: {question}\nGeneral question:"
    )


def step_back_retrieve(question, llm, corpus=CORPUS, k=3):
    """Retrieve for BOTH the original and the step-back question, then fuse."""
    general = llm.complete(step_back_prompt(question)).strip()
    fused = rrf_merge([keyword_search(question, corpus, k), keyword_search(general, corpus, k)])
    return general, fused[:k]


# ---------- 4. HyDE (Hypothetical Document Embeddings) ----------
def hyde_prompt(question):
    return f"Write a short passage that answers the question. It may contain made-up details.\nQuestion: {question}\nPassage:"


def hyde_retrieve(question, llm, corpus=CORPUS, k=3):
    """Search with a FAKE answer instead of the question. The fake answer is never shown to the user."""
    hypothetical = llm.complete(hyde_prompt(question)).strip()
    return hypothetical, keyword_search(hypothetical, corpus, k)


if __name__ == "__main__":
    llm = FakeLLM()
    q = "Why does my pod crash over and over?"
    print("plain     :", keyword_search(q))
    print("expanded  :", expand_query(q), "->", keyword_search(expand_query(q)))
    print("multi     :", multi_query_retrieve(q, llm)[1])
    print("step-back :", step_back_retrieve(q, llm))
    print("HyDE      :", hyde_retrieve(q, llm)[1])
