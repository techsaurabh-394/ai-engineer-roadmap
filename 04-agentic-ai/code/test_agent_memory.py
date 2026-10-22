import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent_memory as am  # noqa: E402


def test_short_term_window_drops_oldest():
    stm = am.ShortTermMemory(max_messages=2)
    for i in range(4):
        stm.add("user", str(i))
    assert [m["content"] for m in stm.messages()] == ["2", "3"]


def test_long_term_keyword_search_ranks_by_overlap():
    ltm = am.LongTermMemory()
    ltm.save("Paris is the capital of France")
    ltm.save("The deploy script lives in scripts deploy")
    ltm.save("Deploy on Fridays is forbidden")
    hits = ltm.search("how to deploy the script", k=2)
    assert hits[0] == "The deploy script lives in scripts deploy"
    assert len(hits) == 2
    assert ltm.search("bananas") == []


def test_long_term_survives_dump_and_load():
    ltm = am.LongTermMemory()
    ltm.save("user prefers dark mode")
    again = am.LongTermMemory.load(ltm.dump())
    assert again.search("dark mode") == ["user prefers dark mode"]


def test_episodic_recall_and_failures():
    epi = am.EpisodicMemory()
    epi.record("deploy staging", ["deploy"], "failure", "migrate first")
    epi.record("write report", ["draft"], "success")
    epi.record("deploy staging", ["migrate", "deploy"], "success")
    recent = epi.recall("deploy to staging")
    assert recent[0]["t"] == 3 and recent[0]["outcome"] == "success"
    assert [e["lesson"] for e in epi.failures()] == ["migrate first"]
    assert epi.recall("unrelated thing") == []


def test_semantic_facts_overwrite_and_case_insensitive():
    sem = am.SemanticMemory()
    sem.set("User", "city", "Pune")
    sem.set("user", "city", "Delhi")
    assert sem.get("USER", "City") == "Delhi"
    assert sem.about("user") == {"city": "Delhi"}
    assert sem.get("user", "age", "unknown") == "unknown"


def test_build_context_includes_all_memory_types():
    stm, ltm, sem = am.ShortTermMemory(2), am.LongTermMemory(), am.SemanticMemory()
    stm.add("user", "hi")
    ltm.save("invoice numbers start with INV")
    sem.set("user", "name", "Asha")
    ctx = am.build_context("what do invoice numbers look like", stm, ltm, sem)
    assert "Asha" in ctx and "INV" in ctx and "user: hi" in ctx
