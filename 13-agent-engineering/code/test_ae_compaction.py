import ae_compaction as m


def test_run_shape():
    r = m.make_run()
    assert len(r) == 30 and m.total(r) == 15 * 900 + 15 * 120
    assert m.facts(r) == {"order id is 8841", "refund limit is 50 dollars", "customer prefers email"}


def test_truncate_fits_budget_and_keeps_pinned():
    r = m.make_run()
    t = m.truncate_oldest(r, 6000)
    assert m.total(t) <= 6000
    assert t[0]["pinned"]


def test_truncate_loses_old_facts():
    t = m.truncate_oldest(m.make_run(), 6000)
    got = m.surviving_facts(t)
    assert "order id is 8841" not in got and "customer prefers email" in got


def test_clear_tool_results_shrinks_and_loses_tool_fact():
    r = m.make_run()
    c = m.clear_old_tool_results(r)
    assert m.total(c) < m.total(r) / 2
    got = m.surviving_facts(c)
    assert "refund limit is 50 dollars" not in got
    assert "order id is 8841" in got


def test_summary_keeps_only_what_it_kept():
    s = m.summarise_old(m.make_run(), summary_keeps={"order id is 8841"})
    assert m.surviving_facts(s) == {"order id is 8841", "customer prefers email"}
    assert any(x["role"] == "summary" for x in s)


def test_notes_rescue_tool_fact():
    n = m.clear_old_tool_results(m.notes_pass(m.make_run(), ["refund limit is 50 dollars"]))
    assert "refund limit is 50 dollars" in m.surviving_facts(n)
