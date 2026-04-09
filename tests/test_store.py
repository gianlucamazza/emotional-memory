from conftest import make_test_memory

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
