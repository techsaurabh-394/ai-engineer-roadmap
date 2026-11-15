"""CLIP-style contrastive learning on toy embeddings, stdlib only.

Two "encoders" are replaced by seeded random vectors. We L2-normalise them,
build the cosine similarity matrix, compute the symmetric InfoNCE loss with a
temperature, classify zero-shot with argmax, and show the loss falls when the
matched pairs are made more similar.

Run:  python mm_clip_contrastive.py
"""
import math
import random


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        raise ValueError("cannot normalise the zero vector")
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def similarity_matrix(images, texts):
    """sim[i][j] = cosine(image i, text j). Rows are images, columns are texts."""
    im = [l2_normalize(v) for v in images]
    tx = [l2_normalize(v) for v in texts]
    return [[dot(a, b) for b in tx] for a in im]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def cross_entropy_rows(logits):
    """Mean of -log softmax(row)[i] where the right answer for row i is column i."""
    return sum(-math.log(softmax(row)[i]) for i, row in enumerate(logits)) / len(logits)


def transpose(m):
    return [list(col) for col in zip(*m)]


def info_nce(sim, temperature=0.07):
    """Symmetric InfoNCE: average of image->text and text->image cross-entropy."""
    logits = [[s / temperature for s in row] for row in sim]
    return (cross_entropy_rows(logits) + cross_entropy_rows(transpose(logits))) / 2


def zero_shot(image, class_text_embeddings):
    """Index of the class prompt whose embedding is most similar to the image."""
    im = l2_normalize(image)
    scores = [dot(im, l2_normalize(t)) for t in class_text_embeddings]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


def make_pairs(n, dim, seed, noise):
    """Each pair shares a hidden 'concept' vector; both sides add their own noise.
    Smaller noise -> matched pairs more similar."""
    rng = random.Random(seed)
    concepts = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    images = [[c + rng.gauss(0, noise) for c in con] for con in concepts]
    texts = [[c + rng.gauss(0, noise) for c in con] for con in concepts]
    return images, texts


def fmt(m):
    return "\n".join("  " + " ".join(f"{x:6.2f}" for x in row) for row in m)


# A hand-made worked example: 3 images and their 3 captions in 3 dimensions.
TOY_IMAGES = [[0.9, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.3, 1.0]]
TOY_TEXTS = [[1.0, 0.2, 0.1], [0.2, 0.9, 0.1], [0.1, 0.1, 0.8]]
TOY_CLASSES = ["a photo of a cat", "a photo of a dog", "a photo of a car"]


def main():
    images, texts = TOY_IMAGES, TOY_TEXTS
    sim = similarity_matrix(images, texts)
    print("3x3 cosine similarity (rows = images, columns = texts):")
    print(fmt(sim))
    for t in (1.0, 0.5, 0.07):
        print(f"temperature {t}: symmetric InfoNCE loss = {info_nce(sim, t):.3f}")
    row0 = softmax([x / 0.5 for x in sim[0]])
    print("row 0 at T=0.5: logits", [round(x / 0.5, 2) for x in sim[0]],
          "-> probabilities", [round(p, 3) for p in row0],
          f"-> -ln(p0) = {-math.log(row0[0]):.3f}")
    print(f"chance loss for N=3 is ln(3) = {math.log(3):.3f}")

    print("\nMaking matched pairs more similar (less noise):")
    for noise in (2.0, 1.0, 0.5, 0.1):
        im, tx = make_pairs(3, 8, seed=3, noise=noise)
        print(f"  noise {noise}: loss = {info_nce(similarity_matrix(im, tx), 0.1):.3f}")

    print("\nZero-shot with 3 class prompts:")
    classes = TOY_CLASSES
    class_emb = texts
    for i, img in enumerate(images):
        pred, scores = zero_shot(img, class_emb)
        print(f"  image {i}: scores {[round(s, 2) for s in scores]} -> {classes[pred]}")

    big_i, big_t = make_pairs(32, 16, seed=1, noise=0.5)
    print(f"\nN=32 pairs: chance loss ln(32) = {math.log(32):.3f}, "
          f"loss = {info_nce(similarity_matrix(big_i, big_t), 0.1):.3f}")


if __name__ == "__main__":
    main()
