"""Tests for ChromaStore (requires chromadb optional dependency).

Run with:
    pytest tests/test_chroma_store.py -v
"""

from __future__ import annotations

import pytest
from conftest import make_test_memory

from emotional_memory.interfaces import MemoryStore
from emotional_memory.stores.chroma import ChromaStore

# Skip entire module if chromadb is not installed.
pytest.importorskip("chromadb")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestChromaStoreInit:
    def test_in_memory_creates_store(self):
        store = ChromaStore()
        assert len(store) == 0

    def test_custom_collection_name(self):
        store = ChromaStore(collection_name="test_col")
        assert store._collection_name == "test_col"

    def test_protocol_conformance(self):
        store = ChromaStore()
        assert isinstance(store, MemoryStore)

    def test_collection_not_ready_before_first_save(self):
        store = ChromaStore()
        assert store._collection is None

    def test_path_and_host_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="at most one"):
            ChromaStore(path=str(tmp_path / "chroma"), host="localhost")


# ---------------------------------------------------------------------------
# Save / Get
# ---------------------------------------------------------------------------


class TestChromaStoreSaveGet:
    def test_save_and_get(self):
        store = ChromaStore()
        m = make_test_memory("hello", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.id == m.id
        assert got.content == m.content

    def test_save_preserves_embedding(self):
        store = ChromaStore()
        m = make_test_memory("embedded", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.embedding == [1.0, 0.0, 0.0]

    def test_get_missing_returns_none_before_collection(self):
        store = ChromaStore()
        assert store.get("nonexistent-id") is None

    def test_get_missing_returns_none_after_collection_created(self):
        store = ChromaStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        assert store.get("nonexistent-id") is None

    def test_save_replaces_existing(self):
        store = ChromaStore()
        m = make_test_memory("original", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        updated = m.model_copy(update={"content": "replaced"})
        store.save(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "replaced"
        assert len(store) == 1

    def test_save_without_embedding_raises(self):
        store = ChromaStore()
        m = make_test_memory("no-embedding")
        with pytest.raises(ValueError, match="embedding"):
            store.save(m)

    def test_save_dim_mismatch_raises(self):
        store = ChromaStore()
        store.save(make_test_memory("first", embedding=[1.0, 0.0, 0.0]))
        m = make_test_memory("wrong-dim", embedding=[1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            store.save(m)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestChromaStoreUpdate:
    def test_update_existing(self):
        store = ChromaStore()
        m = make_test_memory("original", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        updated = m.model_copy(update={"content": "updated"})
        store.update(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "updated"
        assert len(store) == 1

    def test_update_nonexistent_is_noop_no_collection(self):
        store = ChromaStore()
        m = make_test_memory("ghost", embedding=[1.0, 0.0, 0.0])
        store.update(m)
        assert len(store) == 0
        assert store.get(m.id) is None

    def test_update_nonexistent_is_noop_with_collection(self):
        store = ChromaStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        ghost = make_test_memory("ghost", embedding=[0.0, 1.0, 0.0])
        store.update(ghost)
        assert store.get(ghost.id) is None
        assert len(store) == 1


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestChromaStoreDelete:
    def test_delete_existing(self):
        store = ChromaStore()
        m = make_test_memory("deletable", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        store.delete(m.id)
        assert store.get(m.id) is None
        assert len(store) == 0

    def test_delete_nonexistent_does_not_raise_no_collection(self):
        store = ChromaStore()
        store.delete("not-here")  # must not raise

    def test_delete_nonexistent_does_not_raise_with_collection(self):
        store = ChromaStore()
        store.save(make_test_memory("seed", embedding=[1.0, 0.0, 0.0]))
        store.delete("not-here")  # must not raise

    def test_delete_reduces_count(self):
        store = ChromaStore()
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


class TestChromaStoreListAll:
    def test_empty_no_collection(self):
        assert ChromaStore().list_all() == []

    def test_returns_all_memories(self):
        store = ChromaStore()
        memories = [make_test_memory(f"m{i}", embedding=[float(i), 0.0, 0.0]) for i in range(5)]
        for m in memories:
            store.save(m)
        result = store.list_all()
        assert len(result) == 5
        assert {m.id for m in result} == {m.id for m in memories}


class TestChromaStoreLen:
    def test_empty_is_zero(self):
        assert len(ChromaStore()) == 0

    def test_counts_after_saves(self):
        store = ChromaStore()
        store.save(make_test_memory("a", embedding=[1.0, 0.0, 0.0]))
        store.save(make_test_memory("b", embedding=[0.0, 1.0, 0.0]))
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Search by embedding
# ---------------------------------------------------------------------------


class TestChromaStoreSearch:
    def test_returns_closest(self):
        store = ChromaStore()
        target = make_test_memory("target", embedding=[1.0, 0.0, 0.0])
        other = make_test_memory("other", embedding=[0.0, 1.0, 0.0])
        store.save(target)
        store.save(other)
        results = store.search_by_embedding([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == target.id

    def test_top_k_respected(self):
        store = ChromaStore()
        for i in range(5):
            store.save(make_test_memory(f"m{i}", embedding=[float(i + 1), 0.0, 0.0]))
        results = store.search_by_embedding([5.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_empty_store_returns_empty_no_collection(self):
        store = ChromaStore()
        assert store.search_by_embedding([1.0, 0.0, 0.0], top_k=5) == []

    def test_empty_store_returns_empty_with_collection(self):
        store = ChromaStore()
        m = make_test_memory("seed", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        store.delete(m.id)
        assert store.search_by_embedding([1.0, 0.0, 0.0], top_k=5) == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestChromaStoreContextManager:
    def test_with_block_returns_store(self):
        with ChromaStore() as store:
            store.save(make_test_memory("inside-ctx", embedding=[1.0, 0.0, 0.0]))
            assert len(store) == 1

    def test_close_is_callable(self):
        store = ChromaStore()
        store.close()  # must not raise


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestChromaStoreRepr:
    def test_repr_contains_collection_and_count(self):
        store = ChromaStore()
        r = repr(store)
        assert "ChromaStore" in r
        assert "emotional_memory" in r
        assert "count=" in r


# ---------------------------------------------------------------------------
# Persistence (file-backed)
# ---------------------------------------------------------------------------


class TestChromaStorePersistence:
    def test_local_path_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "chroma_data")
        store1 = ChromaStore(path=path)
        m = make_test_memory("persistent", embedding=[1.0, 0.0, 0.0])
        store1.save(m)
        assert len(store1) == 1
        store1.close()

        store2 = ChromaStore(path=path)
        got = store2.get(m.id)
        assert got is not None
        assert got.content == "persistent"
        assert store2._collection is not None
        assert store2._dim == 3
        store2.close()
