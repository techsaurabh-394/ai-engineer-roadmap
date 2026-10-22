import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adv_incremental_index import CountingEmbedder, IncrementalIndex, chunk, content_hash  # noqa: E402

V1 = "Intro paragraph.\n\nInstall with pip.\n\nRun the server."
V2 = "Intro paragraph.\n\nInstall with uv.\n\nRun the server."


def test_hash_is_stable_and_sensitive():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    assert chunk(V1) == ["Intro paragraph.", "Install with pip.", "Run the server."]


def test_reingest_unchanged_costs_nothing():
    emb = CountingEmbedder()
    idx = IncrementalIndex(emb)
    assert idx.upsert("r", V1) == {"added": 3, "unchanged": 0, "removed": 0}
    assert idx.upsert("r", V1) == {"added": 0, "unchanged": 3, "removed": 0}
    assert emb.calls == 3


def test_only_changed_chunk_is_reembedded():
    emb = CountingEmbedder()
    idx = IncrementalIndex(emb)
    idx.upsert("r", V1)
    stats = idx.upsert("r", V2)
    assert stats == {"added": 1, "unchanged": 2, "removed": 1}
    assert emb.calls == 4
    assert "Install with pip." not in idx.texts.values()
    assert len(idx.vectors) == 3


def test_shared_chunks_survive_delete():
    idx = IncrementalIndex(CountingEmbedder())
    idx.upsert("a", "Shared.\n\nOnly in A.")
    idx.upsert("b", "Shared.\n\nOnly in B.")
    assert idx.delete("a") == 1
    assert "Shared." in idx.texts.values()
    assert "Only in A." not in idx.texts.values()
