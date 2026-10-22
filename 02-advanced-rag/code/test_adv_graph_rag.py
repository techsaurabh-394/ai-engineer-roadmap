import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_graph_rag import TRIPLES, TripleStore, find_entities, follow_path, graph_context, multi_hop  # noqa: E402


def store():
    return TripleStore(TRIPLES)


def test_add_is_idempotent_and_query_works():
    kg = store()
    n = len(kg.triples)
    kg.add("billing-service", "owned_by", "Payments Team")
    assert len(kg.triples) == n
    assert kg.query(rel="managed_by", obj="Priya") == [("Payments Team", "managed_by", "Priya")]


def test_entity_linking():
    assert find_entities("Who owns billing-service?", store()) == ["billing-service"]


def test_two_hop_path():
    assert follow_path(store(), "billing-service", ["owned_by", "managed_by"]) == ["Priya"]
    assert follow_path(store(), "billing-service", ["owned_by", "managed_by", "reports_to"]) == ["Marco"]


def test_multi_hop_respects_depth():
    one = multi_hop(store(), "billing-service", max_hops=1)
    two = multi_hop(store(), "billing-service", max_hops=2)
    assert ("Payments Team", "managed_by", "Priya") not in one
    assert ("Payments Team", "managed_by", "Priya") in two
    assert ("Priya", "reports_to", "Marco") not in two


def test_graph_context_contains_answer_fact():
    facts = graph_context("Who manages the team that owns billing-service?", store())
    assert "Payments Team managed by Priya." in facts
    assert not any("Discovery Team" in f for f in facts)
