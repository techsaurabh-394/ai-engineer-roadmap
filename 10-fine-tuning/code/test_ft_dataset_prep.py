import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ft_dataset_prep as p  # noqa: E402


def test_sample_has_30_records():
    assert len(p.SAMPLE) == 30


def test_validation_finds_the_two_bad_records():
    ok, bad = p.validate_all(p.SAMPLE)
    assert len(ok) == 28
    assert {i for i, _ in bad} == {"r27", "r28"}


def test_validate_rules():
    good = {"id": "x", "group": "g", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    assert p.validate_record(good) == []
    no_reply = {**good, "messages": [{"role": "user", "content": "hi"}]}
    assert "last message is not assistant" in p.validate_record(no_reply)
    two_users = {**good, "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                                      {"role": "assistant", "content": "c"}]}
    assert "roles do not alternate user/assistant" in p.validate_record(two_users)
    assert "missing group" in p.validate_record({**good, "group": ""})
    assert p.validate_record("nope") == ["record is not an object"]


def test_dedupe_exact_and_near():
    ok, _ = p.validate_all(p.SAMPLE)
    kept, dropped = p.dedupe(ok)
    assert [(a, b, c) for a, b, c, _ in dropped] == [("r04", "r03", "exact"), ("r06", "r05", "near")]
    assert len(kept) == 26


def test_jaccard():
    assert p.jaccard("a b c", "a b c") == 1.0
    assert p.jaccard("a b", "c d") == 0.0
    assert abs(p.jaccard("a b c", "a b d") - 0.5) < 1e-12


def test_scrub_pii():
    t, c = p.scrub_pii("mail jane.doe@example.com or call 020 7946 0123")
    assert t == "mail [EMAIL] or call [PHONE]"
    assert c == {"email": 1, "phone": 1}
    t2, c2 = p.scrub_pii("Order 1001 ships in 3 to 5 days, 30 percent off")
    assert c2 == {"email": 0, "phone": 0} and t2.startswith("Order 1001")


def test_pipeline_numbers():
    out = p.run_pipeline(p.SAMPLE)
    assert out["pii"] == {"email": 2, "phone": 2}
    assert (len(out["train"]), len(out["val"]), len(out["test"])) == (18, 4, 4)


def test_split_has_no_group_or_text_leakage_and_loses_nothing():
    out = p.run_pipeline(p.SAMPLE)
    tr, va, te = out["train"], out["val"], out["test"]
    assert p.group_overlap(tr, va) == [] and p.group_overlap(tr, te) == [] and p.group_overlap(va, te) == []
    assert p.text_overlap(tr, te) == []
    assert sorted(r["id"] for r in tr + va + te) == sorted(r["id"] for r in out["scrubbed"])


def test_split_is_deterministic_and_seed_sensitive():
    out = p.run_pipeline(p.SAMPLE)
    a = p.split_by_group(out["scrubbed"], seed=3)
    b = p.split_by_group(out["scrubbed"], seed=3)
    assert [[r["id"] for r in s] for s in a] == [[r["id"] for r in s] for s in b]
    c = p.split_by_group(out["scrubbed"], seed=4)
    assert [[r["id"] for r in s] for s in a] != [[r["id"] for r in s] for s in c]


def test_length_stats():
    out = p.run_pipeline(p.SAMPLE)
    st = p.length_stats(out["kept"])
    assert st["n"] == 26 and st["min"] == 12 and st["max"] == 23 and st["median"] == 17.0


def test_build_labels_masks_everything_but_the_reply():
    toks, mask = p.render_chat([{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello friend"}])
    assert toks == ["<|user|>", "hi", "there", "<|end|>", "<|assistant|>", "hello", "friend", "<|end|>"]
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1]
    labels = p.build_labels(toks, mask)
    assert labels == [-100] * 5 + ["hello", "friend", "<|end|>"]


def test_build_labels_with_ids_and_errors():
    assert p.build_labels([5, 6, 7], [0, 1, 1]) == [-100, 6, 7]
    vocab = {"a": 10, "b": 11, "c": 12}
    assert p.build_labels(["a", "b", "c"], [0, 1, 1], vocab) == [-100, 11, 12]
    try:
        p.build_labels(["a"], [1, 1])
        assert False
    except ValueError:
        pass


def test_masked_loss_ignores_prompt_positions():
    labels = [-100, -100, "x", "y"]
    assert p.masked_mean_loss([9.0, 9.0, 1.0, 3.0], labels) == 2.0
