"""Distillation and synthetic data: soft targets, temperature, and a dedup+filter pipeline.

Run:  python fa_distill.py
"""
import math


def softmax(logits, temperature=1.0):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def kl_divergence(p, q):
    """KL(p || q) in nats."""
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def distill_loss(teacher_logits, student_logits, temperature=2.0):
    """KL between softened teacher and student, times T^2 (the usual scaling)."""
    p = softmax(teacher_logits, temperature)
    q = softmax(student_logits, temperature)
    return temperature**2 * kl_divergence(p, q)


def normalise(text):
    return " ".join(text.lower().split())


def clean_synthetic(rows, min_words=3, banned=("as an ai language model",)):
    """Drop too-short rows, boilerplate refusals and exact duplicates (after normalising).
    Returns (kept, report)."""
    seen, kept = set(), []
    report = {"short": 0, "banned": 0, "duplicate": 0}
    for r in rows:
        n = normalise(r)
        if len(n.split()) < min_words:
            report["short"] += 1
        elif any(b in n for b in banned):
            report["banned"] += 1
        elif n in seen:
            report["duplicate"] += 1
        else:
            seen.add(n)
            kept.append(r)
    return kept, report


if __name__ == "__main__":
    teacher = [4.0, 2.0, 1.0, 0.5]
    print("teacher T=1:", [round(x, 3) for x in softmax(teacher)])
    print("teacher T=3:", [round(x, 3) for x in softmax(teacher, 3)], "(softer: shows runner-up classes)")
    print("loss, student == teacher       :", round(distill_loss(teacher, teacher), 4))
    print("loss, student flat [1,1,1,1]   :", round(distill_loss(teacher, [1, 1, 1, 1]), 4))
    print("loss, student close [3.5,2,1,1]:", round(distill_loss(teacher, [3.5, 2, 1, 1]), 4))
    rows = ["Reset your password in settings.", "reset  your password in SETTINGS.", "ok",
            "As an AI language model, I cannot help.", "Invoices are emailed on the 1st."]
    kept, rep = clean_synthetic(rows)
    print("synthetic rows kept:", len(kept), "of", len(rows), rep)
