"""Tests for QdrantStore (requires qdrant-client optional dependency).

Run with:
    pytest tests/test_qdrant_store.py -v
"""

from __future__ import annotations

import pytest
from conftest import make_test_memory

from emotional_memory.interfaces import MemoryStore
from emotional_memory.stores.qdrant import QdrantStore, _uuid_for

# Skip entire module if qdrant-client is not installed.
pytest.importorskip("qdrant_client")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestQdrantStoreInit:
    def test_in_memory_creates_store(self):
        store = QdrantStore()
        assert len(store) == 0

    def test_custom_collection_name(self):
        store = QdrantStore(collection_name="test_col")
        assert store._collection_name == "test_col"

    def test_protocol_conformance(self):
        store = QdrantStore()
        assert isinstance(store, MemoryStore)

    def test_collection_not_ready_before_first_save(self):
        store = QdrantStore()
        assert not store._collection_ready

    def test_url_and_path_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="at most one"):
            QdrantStore(url="http://localhost:6333", path=str(tmp_path / "qdrant"))


# ---------------------------------------------------------------------------
# UUID5 helper
# ---------------------------------------------------------------------------


class TestUuidFor:
    def test_deterministic(self):
        assert _uuid_for("abc") == _uuid_for("abc")

    def test_distinct_ids(self):
        assert _uuid_for("abc") != _uuid_for("xyz")

    def test_returns_string(self):
        result = _uuid_for("hello")
        assert isinstance(result, str)
        assert len(result) == 36  # standard UUID format


# ---------------------------------------------------------------------------
# Save / Get
# ---------------------------------------------------------------------------


class TestQdrantStoreSaveGet:
    def test_save_and_get(self):
        store = QdrantStore()
        m = make_test_memory("hello", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.id == m.id
        assert got.content == m.content

    def test_save_preserves_embedding(self):
        store = QdrantStore()
        m = make_test_memory("embedded", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.embedding == [1.0, 0.0, 0.0]

    def test_get_missing_returns_none_before_collection(self):
        store = QdrantStore()
        assert store.get("nonexistent-id") is None

    def test_get_missing_returns_none_after_collection_created(self):
        store = QdrantStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        assert store.get("nonexistent-id") is None

    def test_save_replaces_existing(self):
        store = QdrantStore()
        m = make_test_memory("original", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        updated = m.model_copy(update={"content": "replaced"})
        store.save(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "replaced"
        assert len(store) == 1

    def test_save_without_embedding_raises(self):
        store = QdrantStore()
        m = make_test_memory("no-embedding")
        with pytest.raises(ValueError, match="embedding"):
            store.save(m)

    def test_save_dim_mismatch_raises(self):
        store = QdrantStore()
        store.save(make_test_memory("first", embedding=[1.0, 0.0, 0.0]))
        m = make_test_memory("wrong-dim", embedding=[1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            store.save(m)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestQdrantStoreUpdate:
    def test_update_existing(self):
        store = QdrantStore()
        m = make_test_memory("original", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        updated = m.model_copy(update={"content": "updated"})
        store.update(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "updated"
        assert len(store) == 1

    def test_update_nonexistent_is_noop_no_collection(self):
        store = QdrantStore()
        m = make_test_memory("ghost", embedding=[1.0, 0.0, 0.0])
        store.update(m)
        assert len(store) == 0
        assert store.get(m.id) is None

    def test_update_nonexistent_is_noop_with_collection(self):
        store = QdrantStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        ghost = make_test_memory("ghost", embedding=[0.0, 1.0, 0.0])
        store.update(ghost)
        assert store.get(ghost.id) is None
        assert len(store) == 1


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestQdrantStoreDelete:
    def test_delete_existing(self):
        store = QdrantStore()
        m = make_test_memory("deletable", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        store.delete(m.id)
        assert store.get(m.id) is None
        assert len(store) == 0

    def test_delete_nonexistent_does_not_raise_no_collection(self):
        store = QdrantStore()
        store.delete("not-here")  # must not raise

    def test_delete_nonexistent_does_not_raise_with_collection(self):
        store = QdrantStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        store.delete("not-here")  # must not raise

    def test_delete_reduces_count(self):
        store = QdrantStore()
        a = make_test_memory("a", embedding=[1.0, 0.0, 0.0])
        b = make_test_memory("b", embedding=[0.0, 1.0, 0.0])
        store.save(a)
        store.save(b)
        assert len(store) == 2
        store.delete(a.id)
        assert len(store) == 1


# ---------------------------------------------------------------------------
# List all / Len
# ---------------------------------------------------------------------------


class TestQdrantStoreListAll:
    def test_empty_no_collection(self):
        assert QdrantStore().list_all() == []

    def test_returns_all_memories(self):
        store = QdrantStore()
        memories = [make_test_memory(f"m{i}", embedding=[float(i), 0.0, 0.0]) for i in range(5)]
        for m in memories:
            store.save(m)
        result = store.list_all()
        assert len(result) == 5
        assert {m.id for m in result} == {m.id for m in memories}


class TestQdrantStoreLen:
    def test_empty_is_zero(self):
        assert len(QdrantStore()) == 0

    def test_counts_after_saves(self):
        store = QdrantStore()
        store.save(make_test_memory("a", embedding=[1.0, 0.0, 0.0]))
        store.save(make_test_memory("b", embedding=[0.0, 1.0, 0.0]))
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Search by embedding
# ---------------------------------------------------------------------------


class TestQdrantStoreSearch:
    def test_returns_closest(self):
        store = QdrantStore()
        target = make_test_memory("target", embedding=[1.0, 0.0, 0.0])
        other = make_test_memory("other", embedding=[0.0, 1.0, 0.0])
        store.save(target)
        store.save(other)
        results = store.search_by_embedding([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == target.id

    def test_top_k_respected(self):
        store = QdrantStore()
        for i in range(5):
            store.save(make_test_memory(f"m{i}", embedding=[float(i + 1), 0.0, 0.0]))
        results = store.search_by_embedding([5.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_empty_store_returns_empty_no_collection(self):
        store = QdrantStore()
        assert store.search_by_embedding([1.0, 0.0, 0.0], top_k=5) == []

    def test_empty_store_returns_empty_with_collection(self):
        store = QdrantStore()
        m = make_test_memory("seed", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        store.delete(m.id)
        assert store.search_by_embedding([1.0, 0.0, 0.0], top_k=5) == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestQdrantStoreContextManager:
    def test_with_block_returns_store(self):
        with QdrantStore() as store:
            store.save(make_test_memory("inside-ctx", embedding=[1.0, 0.0, 0.0]))
            assert len(store) == 1

    def test_close_is_callable(self):
        store = QdrantStore()
        store.close()  # must not raise


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestQdrantStoreRepr:
    def test_repr_contains_collection_and_count(self):
        store = QdrantStore()
        r = repr(store)
        assert "QdrantStore" in r
        assert "emotional_memory" in r
        assert "count=" in r


# ---------------------------------------------------------------------------
# Persistence (file-backed)
# ---------------------------------------------------------------------------


class TestQdrantStorePersistence:
    def test_local_path_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "qdrant_data")
        store1 = QdrantStore(path=path)
        m = make_test_memory("persistent", embedding=[1.0, 0.0, 0.0])
        store1.save(m)
        assert len(store1) == 1
        store1.close()

        store2 = QdrantStore(path=path)
        got = store2.get(m.id)
        assert got is not None
        assert got.content == "persistent"
        assert store2._collection_ready
        assert store2._dim == 3
        store2.close()
