import pytest

import mm_patches as m


def test_196_patches_for_224_over_16():
    assert m.count_patches(224, 16) == 196
    assert m.count_tokens(224, 16) == 197
    assert m.count_tokens(224, 16, cls_token=False) == 196


def test_doubling_resolution_quadruples_patches():
    assert m.count_patches(448, 16) == 4 * m.count_patches(224, 16)


def test_attention_pairs_grow_faster_than_tokens():
    t1, t2 = m.count_tokens(224, 16), m.count_tokens(448, 16)
    assert m.attention_pairs(t2) / m.attention_pairs(t1) > 15


def test_split_and_flatten_preserve_pixels():
    img = [[r * 4 + c for c in range(4)] for r in range(4)]
    patches = m.split_into_patches(img, 2)
    assert len(patches) == 4
    assert patches[0] == [[0, 1], [4, 5]]
    assert patches[1] == [[2, 3], [6, 7]]
    assert patches[2] == [[8, 9], [12, 13]]
    assert m.flatten_patch(patches[3]) == [10, 11, 14, 15]
    flat = sorted(v for p in patches for v in m.flatten_patch(p))
    assert flat == list(range(16))


def test_rgb_vector_length():
    img = m.make_image(8, 8, channels=3, seed=1)
    patches = m.split_into_patches(img, 4)
    assert len(patches) == 4
    assert len(m.flatten_patch(patches[0])) == 4 * 4 * 3 == m.patch_vector_length(4, 3)


def test_indivisible_raises():
    with pytest.raises(ValueError):
        m.split_into_patches(m.make_image(6, 6), 4)
    with pytest.raises(ValueError):
        m.count_patches(225, 16)


def test_position_grid_row_major():
    assert m.position_grid(2, 3) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_project_shape():
    vec = [1, 2, 3]
    w = [[1, 0], [0, 1], [1, 1]]
    assert m.project(vec, w) == [4, 5]


def test_deterministic():
    assert m.make_image(4, 4, seed=3) == m.make_image(4, 4, seed=3)


def test_video_tokens():
    assert m.video_tokens(60, 1, 224) == (60, 60 * 197)
    assert m.video_tokens(60, 2, 224)[1] == 2 * m.video_tokens(60, 1, 224)[1]
    assert m.video_tokens(10, 1, 224, cls_token=False) == (10, 1960)
