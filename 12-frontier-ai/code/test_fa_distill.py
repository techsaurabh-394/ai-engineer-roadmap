import pytest

import fa_distill as m


def test_softmax_sums_to_one_and_orders():
    p = m.softmax([2.0, 1.0, 0.1])
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_temperature_softens():
    sharp = m.softmax([4, 2, 1, 0.5], 1)
    soft = m.softmax([4, 2, 1, 0.5], 3)
    assert max(soft) < max(sharp)


def test_kl_zero_for_equal_and_positive_otherwise():
    p = [0.7, 0.2, 0.1]
    assert m.kl_divergence(p, p) == pytest.approx(0)
    assert m.kl_divergence(p, [1 / 3] * 3) > 0


def test_distill_loss_ordering():
    t = [4.0, 2.0, 1.0, 0.5]
    assert m.distill_loss(t, t) == pytest.approx(0, abs=1e-12)
    assert m.distill_loss(t, [3.5, 2, 1, 1]) < m.distill_loss(t, [1, 1, 1, 1])


def test_clean_synthetic_counts():
    rows = ["Reset your password in settings.", "reset  your password in SETTINGS.", "ok",
            "As an AI language model, I cannot help.", "Invoices are emailed on the 1st."]
    kept, rep = m.clean_synthetic(rows)
    assert kept == ["Reset your password in settings.", "Invoices are emailed on the 1st."]
    assert rep == {"short": 1, "banned": 1, "duplicate": 1}
