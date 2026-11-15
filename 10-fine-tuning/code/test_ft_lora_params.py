import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ft_lora_params as lp  # noqa: E402


def test_counts():
    assert lp.full_params(4096, 4096) == 16_777_216
    assert lp.lora_params(4096, 4096, 8) == 65_536
    assert lp.lora_params(4096, 11008, 8) == 8 * (4096 + 11008) == 120_832


def test_toy_transformer_totals():
    m = lp.ToyTransformer()
    assert m.total_params() == 6_738_149_376
    assert m.lora_trainable(["q", "v"], 8) == 4_194_304
    assert m.lora_trainable(["q", "k", "v", "o"], 16) == 16_777_216
    assert m.lora_trainable(list(m.module_shapes()), 16) == 39_976_960


def test_trainable_grows_linearly_with_rank():
    m = lp.ToyTransformer()
    assert m.lora_trainable(["q"], 16) == 2 * m.lora_trainable(["q"], 8)


def test_rank_function_known_cases():
    assert lp.rank([[1, 2], [2, 4]]) == 1
    assert lp.rank([[1, 0], [0, 1]]) == 2
    assert lp.rank([[0, 0], [0, 0]]) == 0
    assert lp.rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 2


def test_rank_of_BA_is_at_most_r():
    rng = random.Random(0)
    for r in (1, 2, 4, 8):
        BA = lp.matmul(lp.random_matrix(12, r, rng), lp.random_matrix(r, 12, rng))
        assert lp.rank(BA) == r  # Gaussian entries: exactly r, never more
    assert lp.rank(lp.random_matrix(12, 12, rng)) == 12


def test_merge_equals_separate_adapter():
    rng = random.Random(1)
    d, r, alpha = 16, 4, 8
    W, B, A = lp.random_matrix(d, d, rng), lp.random_matrix(d, r, rng), lp.random_matrix(r, d, rng)
    x = [rng.gauss(0, 1) for _ in range(d)]
    a = lp.matvec(lp.merged_weight(W, B, A, alpha, r), x)
    b = lp.adapter_output(W, B, A, alpha, r, x)
    assert max(abs(u - v) for u, v in zip(a, b)) < 1e-9


def test_zero_B_leaves_the_model_unchanged():
    rng = random.Random(2)
    W, A = lp.random_matrix(6, 6, rng), lp.random_matrix(2, 6, rng)
    B = [[0.0, 0.0] for _ in range(6)]
    assert lp.merged_weight(W, B, A, 16, 2) == W


def test_memory_arithmetic():
    m = lp.ToyTransformer()
    t = m.lora_trainable(["q", "v"], 8)
    assert round(lp.full_ft_gb(m.total_params()), 1) == 80.9
    assert round(lp.lora_gb(m.total_params(), t), 2) == 13.54
    assert round(lp.qlora_gb(m.total_params(), t), 2) == 3.53
    assert lp.qlora_gb(m.total_params(), t) < lp.lora_gb(m.total_params(), t) < lp.full_ft_gb(m.total_params())


def test_double_quantization_saves_bits():
    assert lp.nf4_bits_per_param() == 4.5
    assert abs(lp.nf4_bits_per_param(double_quant=True) - 4.127) < 1e-3
    # the paper quotes 0.5 bits -> about 0.127 bits of overhead per parameter
    assert abs((lp.nf4_bits_per_param(double_quant=True) - 4) - 0.127) < 1e-3


def test_batch_and_schedule():
    assert lp.effective_batch(4, 8, 1) == 32
    assert lp.steps_per_epoch(1000, 32) == 32
    assert lp.lr_at(0, 96, 2e-4, 10) == 2e-5
    assert abs(lp.lr_at(9, 96, 2e-4, 10) - 2e-4) < 1e-12
    assert lp.lr_at(95, 96, 2e-4, 10) < lp.lr_at(50, 96, 2e-4, 10) < 2e-4
    assert abs(lp.lr_at(95, 96, 2e-4, 10) - 2e-5) < 1e-6


def test_build_labels():
    assert lp.build_labels([5, 6, 7, 8, 9], [0, 0, 1, 1, 1]) == [-100, -100, 7, 8, 9]
    assert lp.build_labels([1, 2], [0, 0]) == [-100, -100]
    try:
        lp.build_labels([1], [1, 0])
        assert False
    except ValueError:
        pass


def test_loss_curves_are_deterministic_and_overfit_after_epoch_5():
    tr, va = lp.simulate_curves(8, seed=0)
    assert (tr, va) == lp.simulate_curves(8, seed=0)
    assert all(tr[i] > tr[i + 1] for i in range(7))          # training loss keeps falling
    assert lp.best_epoch(va) == 4                              # epoch 5 (0-based 4)
    assert lp.early_stop_epoch(va, patience=2) == 6            # fires at epoch 7
    gap = lp.overfit_gap(tr, va)
    assert gap[7] > gap[4] > 0
    assert lp.early_stop_epoch([3.0, 2.0, 1.0]) is None
