import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_lora_math import (
    adapter_forward, finetune_memory_gb, full_params, lora_memory_gb, lora_params, low_rank_update,
    matmul, matvec, merge, rows_proportional,
)

W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
B = [[1, 0], [0, 1], [1, 1], [0, 2]]
A = [[1, 0, 2, 0], [0, 1, 0, 1]]


def test_matmul_hand_example():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_parameter_counts():
    assert full_params(4096) == 16_777_216
    assert lora_params(4096, 8) == 65_536
    assert full_params(4096) // lora_params(4096, 8) == 256


def test_update_shape_and_values():
    assert matmul(B, A) == [[1, 0, 2, 0], [0, 1, 0, 1], [1, 1, 2, 1], [0, 2, 0, 2]]
    update = low_rank_update(B, A, alpha=4, r=2)  # scale 2
    assert len(update) == 4 and len(update[0]) == 4
    assert update[2] == [2, 2, 4, 2]


def test_merge_values():
    merged = merge(W, B, A, 4, 2)
    assert merged == [[3, 0, 4, 0], [0, 3, 0, 2], [2, 2, 5, 2], [0, 4, 0, 5]]


def test_merge_equals_separate_adapter():
    x = [1, 2, 3, 4]
    merged_out = matvec(merge(W, B, A, 4, 2), x)
    sep_out = adapter_forward(W, B, A, 4, 2, x)
    assert all(abs(a - b) < 1e-9 for a, b in zip(merged_out, sep_out))
    assert merged_out == [15, 14, 29, 28]


def test_zero_B_leaves_model_unchanged():
    zero_B = [[0, 0]] * 4
    assert merge(W, zero_B, A, 4, 2) == W


def test_rank_one_rows_proportional():
    U = low_rank_update([[1], [2], [-1], [0]], [[3, 0, 1, 2]], 1, 1)
    assert rows_proportional(U)
    assert U[1] == [6, 0, 2, 4]


def test_higher_rank_not_proportional():
    assert not rows_proportional(W)
    assert not rows_proportional(matmul(B, A))  # rank 2


def test_memory_rule_of_thumb():
    assert finetune_memory_gb(7e9) == 84.0
    trainable = 4_194_304
    assert lora_memory_gb(7e9, trainable) < finetune_memory_gb(7e9) / 4
    assert lora_memory_gb(7e9, 0) == 14.0
