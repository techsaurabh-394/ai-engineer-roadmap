"""Toy versions of RAG evaluation metrics, in the spirit of RAGAS.

Real tools (RAGAS, DeepEval) usually use an LLM to split text into claims and to
judge whether a claim is supported. Here we use sentence splitting and word
overlap so the logic is visible and runs offline. The FORMULAS are the point.

  context precision : are the relevant chunks ranked near the top?
  context recall    : do the retrieved chunks cover every fact in the reference answer?
  faithfulness      : is every claim in the generated answer supported by the chunks?
  answer relevancy  : does the answer address the question?

Run:  python adv_rag_metrics.py
"""
import re

STOP = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or that the this to was what when where which who why will with".split()
)


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def claims(text):
    """Split into sentence-level claims. An LLM does this in real tools."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def supported(claim, contexts, threshold=0.6):
    """A claim is supported if most of its content words appear in a single context chunk."""
    cw = words(claim)
    if not cw:
        return True
    return any(len(cw & words(c)) / len(cw) >= threshold for c in contexts)


def context_precision(relevance):
    """relevance: list of 0/1 for each retrieved chunk, in rank order.

    Average of precision@k taken at every rank k that holds a relevant chunk.
    [1, 0, 1] -> (1/1 + 2/3) / 2 = 0.833
    """
    hits = 0
    total = 0.0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def context_recall(reference_answer, contexts, threshold=0.6):
    """Share of reference-answer claims that the retrieved contexts support."""
    ref = claims(reference_answer)
    if not ref:
        return 0.0
    return sum(supported(c, contexts, threshold) for c in ref) / len(ref)


def faithfulness(answer, contexts, threshold=0.6):
    """Share of the generated answer's claims that the retrieved contexts support."""
    ans = claims(answer)
    if not ans:
        return 0.0
    return sum(supported(c, contexts, threshold) for c in ans) / len(ans)


def answer_relevancy(question, answer):
    """Toy: share of the question's content words that the answer addresses.

    RAGAS instead asks an LLM to generate questions from the answer and compares
    them to the real question with embeddings.
    """
    q = words(question)
    return len(q & words(answer)) / len(q) if q else 0.0


def evaluate(sample):
    return {
        "context_precision": round(context_precision(sample["relevance"]), 3),
        "context_recall": round(context_recall(sample["reference"], sample["contexts"]), 3),
        "faithfulness": round(faithfulness(sample["answer"], sample["contexts"]), 3),
        "answer_relevancy": round(answer_relevancy(sample["question"], sample["answer"]), 3),
    }


SAMPLE = {
    "question": "Where should refresh tokens be stored?",
    "contexts": [
        "Refresh tokens belong in an httpOnly cookie and should be rotated on every use.",
        "Docker Compose starts several services from one YAML file.",
        "Access tokens should be short lived.",
    ],
    "relevance": [1, 0, 1],
    "reference": "Refresh tokens belong in an httpOnly cookie. They should be rotated on every use.",
    "answer": "Refresh tokens should be stored in an httpOnly cookie. They expire after exactly 30 days.",
}

if __name__ == "__main__":
    for name, value in evaluate(SAMPLE).items():
        print(f"{name:18s} {value}")
