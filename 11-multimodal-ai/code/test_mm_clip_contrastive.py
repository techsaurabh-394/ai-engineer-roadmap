import math

import pytest

import mm_clip_contrastive as c


def test_normalize_unit_length():
    v = c.l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        c.l2_normalize([0.0, 0.0])


def test_similarity_matrix_shape_and_range():
    sim = c.similarity_matrix(c.TOY_IMAGES, c.TOY_TEXTS)
    assert len(sim) == 3 and all(len(r) == 3 for r in sim)
    assert all(-1.0 <= x <= 1.0 + 1e-9 for r in sim for x in r)
    for i in range(3):
        assert sim[i][i] == max(sim[i])  # matched pair is the best in its row


def test_identical_embeddings_give_cosine_one():
    sim = c.similarity_matrix([[1, 2, 3]], [[2, 4, 6]])
    assert sim[0][0] == pytest.approx(1.0)


def test_uniform_similarity_gives_chance_loss():
    sim = [[0.5] * 4 for _ in range(4)]
    assert c.info_nce(sim, 0.1) == pytest.approx(math.log(4))


def test_loss_is_symmetric_under_transpose():
    sim = c.similarity_matrix(c.TOY_IMAGES, c.TOY_TEXTS)
    assert c.info_nce(sim, 0.5) == pytest.approx(c.info_nce(c.transpose(sim), 0.5))


def test_loss_falls_when_matched_pairs_more_similar():
    losses = []
    for noise in (2.0, 1.0, 0.5, 0.1):
        im, tx = c.make_pairs(3, 8, seed=3, noise=noise)
        losses.append(c.info_nce(c.similarity_matrix(im, tx), 0.1))
    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < 0.01


def test_wrong_pairing_has_higher_loss():
    sim = c.similarity_matrix(c.TOY_IMAGES, c.TOY_TEXTS)
    shuffled = [sim[1], sim[2], sim[0]]
    assert c.info_nce(shuffled, 0.5) > c.info_nce(sim, 0.5)


def test_temperature_changes_loss_when_aligned():
    sim = c.similarity_matrix(c.TOY_IMAGES, c.TOY_TEXTS)
    assert c.info_nce(sim, 0.07) < c.info_nce(sim, 0.5) < c.info_nce(sim, 1.0)


def test_zero_shot_argmax():
    for i, img in enumerate(c.TOY_IMAGES):
        pred, scores = c.zero_shot(img, c.TOY_TEXTS)
        assert pred == i
        assert len(scores) == 3


def test_deterministic():
    assert c.make_pairs(3, 4, seed=9, noise=0.5) == c.make_pairs(3, 4, seed=9, noise=0.5)
