from conftest import make_test_memory

from emotional_memory.interfaces import MemoryStore
from emotional_memory.stores.in_memory import InMemoryStore


class TestInMemoryStore:
    def test_save_and_get(self):
        store = InMemoryStore()
        m = make_test_memory("hello")
        store.save(m)
        assert store.get(m.id) == m

    def test_get_missing_returns_none(self):
        store = InMemoryStore()
        assert store.get("nonexistent") is None

    def test_update(self):
        store = InMemoryStore()
        m = make_test_memory("original")
        store.save(m)
        updated = m.model_copy(update={"content": "updated"})
        store.update(updated)
        assert store.get(m.id).content == "updated"

    def test_delete(self):
        store = InMemoryStore()
        m = make_test_memory()
        store.save(m)
        store.delete(m.id)
        assert store.get(m.id) is None

    def test_delete_nonexistent_no_error(self):
        store = InMemoryStore()
        store.delete("ghost")  # should not raise

    def test_list_all_empty(self):
        store = InMemoryStore()
        assert store.list_all() == []

    def test_list_all_returns_all(self):
        store = InMemoryStore()
        m1 = make_test_memory("a")
        m2 = make_test_memory("b")
        store.save(m1)
        store.save(m2)
        ids = {m.id for m in store.list_all()}
        assert ids == {m1.id, m2.id}

    def test_len(self):
        store = InMemoryStore()
        assert len(store) == 0
        store.save(make_test_memory())
        assert len(store) == 1

    def test_search_by_embedding_empty(self):
        store = InMemoryStore()
        assert store.search_by_embedding([1.0, 0.0], top_k=3) == []

    def test_search_by_embedding_skips_no_embedding(self):
        store = InMemoryStore()
        store.save(make_test_memory())  # no embedding
        assert store.search_by_embedding([1.0, 0.0], top_k=3) == []

    def test_search_by_embedding_ranking(self):
        store = InMemoryStore()
        close = make_test_memory(embedding=[1.0, 0.0])
        far = make_test_memory(embedding=[0.0, 1.0])
        store.save(close)
        store.save(far)
        results = store.search_by_embedding([1.0, 0.0], top_k=2)
        assert results[0].id == close.id

    def test_search_by_embedding_top_k(self):
        store = InMemoryStore()
        for _ in range(5):
            store.save(make_test_memory(embedding=[1.0, 0.0]))
        results = store.search_by_embedding([1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_by_embedding_non_positive_top_k(self):
        """Non-positive top_k returns [] instead of raising out of numpy."""
        store = InMemoryStore()
        store.save(make_test_memory(embedding=[1.0, 0.0]))
        assert store.search_by_embedding([1.0, 0.0], top_k=0) == []
        assert store.search_by_embedding([1.0, 0.0], top_k=-1) == []

    def test_search_by_embedding_zero_query_norm(self):
        store = InMemoryStore()
        m = make_test_memory(embedding=[1.0, 0.0])
        store.save(m)
        results = store.search_by_embedding([0.0, 0.0], top_k=1)
        assert results == [m]

    def test_search_cache_reused_across_queries(self):
        store = InMemoryStore()
        close = make_test_memory(embedding=[1.0, 0.0])
        far = make_test_memory(embedding=[0.0, 1.0])
        store.save(close)
        store.save(far)
        first = store.search_by_embedding([1.0, 0.0], top_k=2)
        assert store._dirty is False
        assert store._matrix is not None
        matrix_id = id(store._matrix)
        second = store.search_by_embedding([0.0, 1.0], top_k=2)
        assert id(store._matrix) == matrix_id
        assert first[0].id == close.id
        assert second[0].id == far.id

    def test_search_cache_invalidated_on_update(self):
        store = InMemoryStore()
        m = make_test_memory(embedding=[1.0, 0.0])
        store.save(m)
        assert store.search_by_embedding([1.0, 0.0], top_k=1)[0].id == m.id
        updated = m.model_copy(update={"embedding": [0.0, 1.0]})
        store.update(updated)
        assert store._dirty is True
        results = store.search_by_embedding([0.0, 1.0], top_k=1)
        assert results[0].id == m.id
        assert store._dirty is False

    def test_search_cache_invalidated_on_delete(self):
        store = InMemoryStore()
        keep = make_test_memory(embedding=[1.0, 0.0])
        drop = make_test_memory(embedding=[0.0, 1.0])
        store.save(keep)
        store.save(drop)
        store.search_by_embedding([1.0, 0.0], top_k=2)
        store.delete(drop.id)
        assert store._dirty is True
        results = store.search_by_embedding([0.0, 1.0], top_k=2)
        assert [r.id for r in results] == [keep.id]

    def test_search_zero_norm_embedding_does_not_crash(self):
        store = InMemoryStore()
        store.save(make_test_memory(embedding=[0.0, 0.0]))
        store.save(make_test_memory(embedding=[1.0, 0.0]))
        results = store.search_by_embedding([1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_protocol_conformance(self):
        assert isinstance(InMemoryStore(), MemoryStore)
