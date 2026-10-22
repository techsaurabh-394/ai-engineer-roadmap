import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_rerank import DOCS, BiEncoderIndex, cosine, cross_score, embed, rerank, retrieve_then_rerank  # noqa: E402

Q = "How do I cancel my subscription and get a refund?"


def test_cosine_basics():
    assert abs(cosine(embed("cat dog"), embed("cat dog")) - 1.0) < 1e-9
    assert cosine(embed("cat"), embed("dog")) == 0.0


def test_bi_encoder_is_fooled_by_repetition():
    top = BiEncoderIndex(DOCS).search(Q, k=3)
    assert top[0][1] == "refund-faq"
    assert "cancel-guide" in [d for _, d in top]


def test_cross_encoder_fixes_order():
    candidates, final = retrieve_then_rerank(Q, k=3, top_n=2)
    assert final[0][1] == "cancel-guide"
    assert len(final) == 2


def test_cross_score_prefers_full_coverage():
    assert cross_score(Q, DOCS["cancel-guide"]) > cross_score(Q, DOCS["refund-faq"])
    assert cross_score("", "anything") == 0.0


def test_rerank_only_reorders_candidates():
    final = rerank(Q, [(0.1, "shipping")], DOCS, top_n=5)
    assert [d for _, d in final] == ["shipping"]
