import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import llm_training as lt  # noqa: E402


def test_cross_entropy_quarter():
    assert math.isclose(lt.cross_entropy(0.25), -math.log(0.25))
    assert math.isclose(lt.cross_entropy(0.25), 1.386, abs_tol=1e-3)


def test_perplexity_of_quarter_is_four():
    assert math.isclose(lt.perplexity(lt.cross_entropy(0.25)), 4.0)


def test_gradient_descent_converges():
    path = lt.gradient_descent_1d(start=0.0, target=3.0, lr=0.1, steps=60)
    assert abs(path[-1] - 3.0) < 1e-3
    assert path[1] > path[0]


def test_softmax_sums_to_one():
    assert math.isclose(sum(lt.softmax([1.0, 2.0, -3.0, 0.5])), 1.0)


def test_bigram_loss_decreases():
    _, _, _, losses = lt.train_bigram(steps=100)
    assert losses[-1] < losses[0]
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))


def test_untrained_loss_is_ln_vocab():
    vocab, _, _, losses = lt.train_bigram(steps=1)
    assert math.isclose(losses[0], math.log(len(vocab)))


def test_bigram_learns_the_followers():
    vocab, idx, logits, _ = lt.train_bigram(steps=300)
    probs = lt.next_token_probs(vocab, idx, logits, "the")
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["cat"] + probs["dog"] > 0.6
    assert probs["cat"] > probs["sat"]


def test_loss_mask_excludes_prompt():
    mask = lt.loss_mask(["Q:", "hi", "A:"], ["hello", "<eos>"])
    assert mask == [0, 0, 0, 1, 1]
    assert lt.masked_loss([9.0, 9.0, 9.0, 1.0, 3.0], mask) == 2.0


def test_bradley_terry():
    assert lt.bradley_terry_prob(1.0, 1.0) == 0.5
    assert lt.bradley_terry_prob(3.0, 0.0) > 0.9
    assert lt.reward_model_loss(3.0, 0.0) < lt.reward_model_loss(0.0, 3.0)


def test_dpo_loss_lower_when_policy_prefers_chosen():
    good = lt.dpo_loss(-1.0, -3.0, -2.0, -2.0)
    bad = lt.dpo_loss(-3.0, -1.0, -2.0, -2.0)
    assert good < bad
    assert math.isclose(lt.dpo_loss(-2.0, -2.0, -2.0, -2.0), math.log(2))
