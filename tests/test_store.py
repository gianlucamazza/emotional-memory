from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.stimmung import StimmungField
from emotional_memory.stores.in_memory import InMemoryStore


def _memory(
    content: str = "test", valence: float = 0.0, embedding: list[float] | None = None
) -> Memory:
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=0.5),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.7,
    )
    return Memory.create(content=content, tag=tag, embedding=embedding)


class TestInMemoryStore:
    def test_save_and_get(self):
        store = InMemoryStore()
        m = _memory("hello")
        store.save(m)
        assert store.get(m.id) == m

    def test_get_missing_returns_none(self):
        store = InMemoryStore()
        assert store.get("nonexistent") is None

    def test_update(self):
        store = InMemoryStore()
        m = _memory("original")
        store.save(m)
        updated = m.model_copy(update={"content": "updated"})
        store.update(updated)
        assert store.get(m.id).content == "updated"

    def test_delete(self):
        store = InMemoryStore()
        m = _memory()
        store.save(m)
        store.delete(m.id)
        assert store.get(m.id) is None

    def test_delete_missing_is_noop(self):
        store = InMemoryStore()
        store.delete("does-not-exist")  # should not raise

    def test_list_all_empty(self):
        assert InMemoryStore().list_all() == []

    def test_list_all(self):
        store = InMemoryStore()
        m1, m2 = _memory("a"), _memory("b")
        store.save(m1)
        store.save(m2)
        ids = {m.id for m in store.list_all()}
        assert ids == {m1.id, m2.id}

    def test_len(self):
        store = InMemoryStore()
        assert len(store) == 0
        store.save(_memory())
        assert len(store) == 1

    def test_search_by_embedding_empty_store(self):
        store = InMemoryStore()
        result = store.search_by_embedding([1.0, 0.0], top_k=5)
        assert result == []

    def test_search_by_embedding_top_k_ordering(self):
        store = InMemoryStore()
        # Three memories with known embeddings
        m_best = _memory("best", embedding=[1.0, 0.0, 0.0])
        m_mid = _memory("mid", embedding=[0.7, 0.7, 0.0])
        m_worst = _memory("worst", embedding=[0.0, 0.0, 1.0])
        for m in (m_best, m_mid, m_worst):
            store.save(m)
        query = [1.0, 0.0, 0.0]
        results = store.search_by_embedding(query, top_k=2)
        assert len(results) == 2
        assert results[0].id == m_best.id  # most similar first

    def test_search_skips_memories_without_embedding(self):
        store = InMemoryStore()
        m_no_emb = _memory("no emb")  # embedding=None
        m_with_emb = _memory("with emb", embedding=[1.0, 0.0])
        store.save(m_no_emb)
        store.save(m_with_emb)
        results = store.search_by_embedding([1.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0].id == m_with_emb.id

    def test_search_top_k_limits_results(self):
        store = InMemoryStore()
        for i in range(10):
            store.save(_memory(f"m{i}", embedding=[float(i), 1.0]))
        results = store.search_by_embedding([1.0, 1.0], top_k=3)
        assert len(results) == 3
