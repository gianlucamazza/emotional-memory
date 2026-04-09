"""Tests for SQLiteStore (requires sqlite-vec optional dependency).

Run with:
    pytest tests/test_sqlite_store.py -v
"""

from __future__ import annotations

import sqlite3

import pytest
from conftest import make_test_memory

from emotional_memory.interfaces import MemoryStore
from emotional_memory.stores.sqlite import SQLiteStore, _pack_embedding, _unpack_embedding

# Skip entire module if sqlite-vec is not installed.
# Placed after imports: stores/sqlite.py does not import sqlite-vec at module level,
# so the import above succeeds even without the optional dependency.
pytest.importorskip("sqlite_vec")

# ---------------------------------------------------------------------------
# Byte-packing helpers
# ---------------------------------------------------------------------------


class TestPackUnpackEmbedding:
    def test_round_trip(self):
        vec = [0.1, 0.5, -0.3, 1.0]
        unpacked = _unpack_embedding(_pack_embedding(vec))
        assert len(unpacked) == len(vec)
        for a, b in zip(unpacked, vec, strict=True):
            assert abs(a - b) < 1e-5

    def test_empty_embedding(self):
        assert _unpack_embedding(_pack_embedding([])) == []


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestSQLiteStoreInit:
    def test_in_memory_creates_store(self):
        store = SQLiteStore(":memory:")
        assert len(store) == 0

    def test_file_backed_creates_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        assert db_path.exists()
        assert len(store) == 0
        store.close()

    def test_protocol_conformance(self):
        store = SQLiteStore(":memory:")
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# Save / Get
# ---------------------------------------------------------------------------


class TestSQLiteStoreSaveGet:
    def test_save_and_get(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("hello")
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.id == m.id
        assert got.content == m.content

    def test_save_with_embedding(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("embedded", embedding=[1.0, 0.0, 0.0])
        store.save(m)
        got = store.get(m.id)
        assert got is not None
        assert got.embedding is not None
        for a, b in zip(got.embedding, [1.0, 0.0, 0.0], strict=True):
            assert abs(a - b) < 1e-5

    def test_get_missing_returns_none(self):
        store = SQLiteStore(":memory:")
        assert store.get("does-not-exist") is None

    def test_save_replaces_existing(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("original")
        store.save(m)
        updated = m.model_copy(update={"content": "replaced"})
        store.save(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "replaced"
        assert len(store) == 1


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestSQLiteStoreUpdate:
    def test_update_existing(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("original")
        store.save(m)
        updated = m.model_copy(update={"content": "updated"})
        store.update(updated)
        got = store.get(m.id)
        assert got is not None
        assert got.content == "updated"

    def test_update_nonexistent_is_noop(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("ghost")
        store.update(m)  # no error, no new row
        assert len(store) == 0
        assert store.get(m.id) is None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestSQLiteStoreDelete:
    def test_delete_existing(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("bye")
        store.save(m)
        store.delete(m.id)
        assert store.get(m.id) is None
        assert len(store) == 0

    def test_delete_nonexistent_no_error(self):
        store = SQLiteStore(":memory:")
        store.delete("ghost-id")  # must not raise

    def test_delete_removes_from_vec_table(self):
        store = SQLiteStore(":memory:")
        m = make_test_memory("vec-entry", embedding=[1.0, 0.0])
        store.save(m)
        store.delete(m.id)
        results = store.search_by_embedding([1.0, 0.0], top_k=5)
        assert all(r.id != m.id for r in results)


# ---------------------------------------------------------------------------
# List all / Len
# ---------------------------------------------------------------------------


class TestSQLiteStoreListAll:
    def test_list_all_empty(self):
        store = SQLiteStore(":memory:")
        assert store.list_all() == []

    def test_list_all_returns_all(self):
        store = SQLiteStore(":memory:")
        m1 = make_test_memory("one")
        m2 = make_test_memory("two")
        m3 = make_test_memory("three")
        store.save(m1)
        store.save(m2)
        store.save(m3)
        ids = {m.id for m in store.list_all()}
        assert {m1.id, m2.id, m3.id} == ids


class TestSQLiteStoreLen:
    def test_len_empty(self):
        store = SQLiteStore(":memory:")
        assert len(store) == 0

    def test_len_after_saves(self):
        store = SQLiteStore(":memory:")
        store.save(make_test_memory("a"))
        store.save(make_test_memory("b"))
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestSQLiteStoreSearch:
    def test_search_returns_closest(self):
        store = SQLiteStore(":memory:")
        near = make_test_memory("near", embedding=[1.0, 0.0, 0.0])
        far = make_test_memory("far", embedding=[0.0, 0.0, 1.0])
        store.save(near)
        store.save(far)
        results = store.search_by_embedding([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == near.id

    def test_search_respects_top_k(self):
        store = SQLiteStore(":memory:")
        for i in range(5):
            store.save(make_test_memory(f"item{i}", embedding=[float(i), 0.0]))
        results = store.search_by_embedding([1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self):
        store = SQLiteStore(":memory:")
        results = store.search_by_embedding([1.0, 0.0], top_k=5)
        assert results == []

    def test_brute_force_fallback(self):
        """With no vec table, falls back to brute-force cosine scan."""
        store = SQLiteStore(":memory:")
        # Save a memory without an embedding — vec table stays uninitialized
        store.save(make_test_memory("no-vec"))
        assert not store._vec_ready
        # Search runs brute-force; finds nothing since no memory has an embedding
        results = store.search_by_embedding([1.0, 0.0], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Context manager / close
# ---------------------------------------------------------------------------


class TestSQLiteStoreContextManager:
    def test_context_manager(self):
        with SQLiteStore(":memory:") as store:
            m = make_test_memory("ctx")
            store.save(m)
            assert store.get(m.id) is not None

    def test_close_closes_connection(self):
        store = SQLiteStore(":memory:")
        store.close()
        with pytest.raises(sqlite3.ProgrammingError):
            store.get("any-id")


# ---------------------------------------------------------------------------
# Persistence across sessions
# ---------------------------------------------------------------------------


class TestSQLiteStoreReopen:
    def test_reopen_existing_db(self, tmp_path):
        db_path = tmp_path / "persist.db"
        m = make_test_memory("persisted", embedding=[0.5, 0.5])

        # Session 1: write
        with SQLiteStore(db_path) as s1:
            s1.save(m)

        # Session 2: read
        with SQLiteStore(db_path) as s2:
            assert s2._vec_ready
            got = s2.get(m.id)
            assert got is not None
            assert got.content == "persisted"
            results = s2.search_by_embedding([0.5, 0.5], top_k=1)
            assert len(results) == 1
            assert results[0].id == m.id


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSQLiteStoreEdgeCases:
    def test_save_without_then_with_embedding(self):
        store = SQLiteStore(":memory:")
        m_no_emb = make_test_memory("no-embedding")
        m_with_emb = make_test_memory("with-embedding", embedding=[1.0, 0.0])
        store.save(m_no_emb)
        assert not store._vec_ready
        store.save(m_with_emb)
        assert store._vec_ready
        assert store.get(m_no_emb.id) is not None
        assert store.get(m_with_emb.id) is not None

    def test_multiple_embeddings_same_dimension(self):
        store = SQLiteStore(":memory:")
        for i in range(4):
            store.save(make_test_memory(f"m{i}", embedding=[float(i), 0.0]))
        results = store.search_by_embedding([3.0, 0.0], top_k=2)
        assert len(results) == 2
