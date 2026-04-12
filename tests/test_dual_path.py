"""Unit tests for dual-path encoding (LeDoux, 1996) — elaborate() and elaborate_pending()."""

from conftest import DeterministicEmbedder

from emotional_memory import CoreAffect, EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine


def _appraisal_engine(valence: float = 0.7, arousal: float = 0.6) -> StaticAppraisalEngine:
    return StaticAppraisalEngine(
        AppraisalVector(
            novelty=0.5,
            goal_relevance=0.8,
            coping_potential=0.7,
            norm_congruence=0.6,
            self_relevance=0.5,
        )
    )


def _engine(dual_path: bool = False, **config_kwargs) -> EmotionalMemory:
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=DeterministicEmbedder(),
        appraisal_engine=_appraisal_engine(),
        config=EmotionalMemoryConfig(dual_path_encoding=dual_path, **config_kwargs),
    )


# ---------------------------------------------------------------------------
# Fast-path encoding
# ---------------------------------------------------------------------------


class TestFastPathEncoding:
    def test_fast_path_sets_pending_appraisal(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Something happened quickly.")
        assert mem.tag.pending_appraisal is True

    def test_fast_path_has_no_appraisal_vector(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Something happened quickly.")
        assert mem.tag.appraisal is None

    def test_disabled_dual_path_performs_full_appraisal(self):
        em = _engine(dual_path=False)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Full pipeline encode.")
        assert mem.tag.pending_appraisal is False
        assert mem.tag.appraisal is not None

    def test_explicit_appraisal_bypasses_fast_path(self):
        """Explicit appraisal= argument is always used regardless of dual_path_encoding."""
        em = _engine(dual_path=True)
        explicit = AppraisalVector(
            novelty=0.9,
            goal_relevance=0.9,
            coping_potential=0.9,
            norm_congruence=0.9,
            self_relevance=0.9,
        )
        mem = em.encode("Explicit appraisal provided.", appraisal=explicit)
        assert mem.tag.pending_appraisal is False
        assert mem.tag.appraisal == explicit

    def test_fast_path_memory_is_retrievable(self):
        """Fast-path memories can be retrieved before elaboration."""
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.5, arousal=0.6))
        em.encode("A retrievable fast-path memory.")
        results = em.retrieve("fast path memory", top_k=1)
        assert len(results) == 1
        assert results[0].tag.pending_appraisal is True


# ---------------------------------------------------------------------------
# elaborate()
# ---------------------------------------------------------------------------


class TestElaborate:
    def test_elaborate_clears_pending_appraisal(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Pending appraisal memory.")
        assert mem.tag.pending_appraisal is True

        updated = em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.pending_appraisal is False

    def test_elaborate_populates_appraisal_vector(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Memory needing appraisal.")
        updated = em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.appraisal is not None

    def test_elaborate_blends_core_affect(self):
        """core_affect after elaboration should differ from raw fast-path affect."""
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.0, arousal=0.3))  # low arousal raw affect
        mem = em.encode("Memory at low arousal.")
        raw_affect = mem.tag.core_affect

        updated = em.elaborate(mem.id)
        assert updated is not None
        # Appraised affect will differ from raw; blended should be between them
        assert updated.tag.core_affect != raw_affect

    def test_elaborate_updates_consolidation_strength(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.0, arousal=0.1))
        mem = em.encode("Very low arousal memory.")
        raw_cs = mem.tag.consolidation_strength

        updated = em.elaborate(mem.id)
        assert updated is not None
        # Appraisal should raise arousal → consolidation_strength changes
        assert updated.tag.consolidation_strength != raw_cs

    def test_elaborate_opens_reconsolidation_window(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Memory to elaborate.")
        assert mem.tag.window_opened_at is None

        updated = em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.window_opened_at is not None

    def test_elaborate_returns_none_when_not_pending(self):
        em = _engine(dual_path=False)  # full appraisal — no pending flag
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Already elaborated memory.")
        assert mem.tag.pending_appraisal is False
        result = em.elaborate(mem.id)
        assert result is None

    def test_elaborate_returns_none_when_no_engine(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            appraisal_engine=None,
            config=EmotionalMemoryConfig(dual_path_encoding=True),
        )
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("No engine to elaborate with.")
        # pending_appraisal is False because no engine → fast path not activated
        result = em.elaborate(mem.id)
        assert result is None

    def test_elaborate_returns_none_for_missing_id(self):
        em = _engine(dual_path=True)
        result = em.elaborate("nonexistent-id")
        assert result is None

    def test_elaborate_persists_to_store(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = em.encode("Stored memory.")
        em.elaborate(mem.id)
        stored = em.get(mem.id)
        assert stored is not None
        assert stored.tag.pending_appraisal is False


# ---------------------------------------------------------------------------
# elaborate_pending()
# ---------------------------------------------------------------------------


class TestElaboratePending:
    def test_elaborate_pending_processes_all(self):
        em = _engine(dual_path=True)
        for i in range(3):
            em.set_affect(CoreAffect(valence=0.1 * i, arousal=0.5))
            em.encode(f"Memory {i} pending appraisal.")

        pending_before = sum(1 for m in em.list_all() if m.tag.pending_appraisal)
        assert pending_before == 3

        elaborated = em.elaborate_pending()
        assert len(elaborated) == 3
        pending_after = sum(1 for m in em.list_all() if m.tag.pending_appraisal)
        assert pending_after == 0

    def test_elaborate_pending_skips_non_pending(self):
        em = _engine(dual_path=False)  # no pending memories
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        em.encode("Already elaborated.")
        result = em.elaborate_pending()
        assert result == []

    def test_elaborate_pending_mixed_store(self):
        """Only pending memories should be elaborated."""
        # Encode one with full appraisal and one with fast path
        em_full = _engine(dual_path=False)
        em_full.set_affect(CoreAffect(valence=0.5, arousal=0.6))
        em_full.encode("Full appraisal memory.")
        full_mem = em_full.list_all()[0]

        em_fast = _engine(dual_path=True)
        em_fast.set_affect(CoreAffect(valence=0.5, arousal=0.6))
        em_fast.encode("Fast path memory.")
        fast_mem = em_fast.list_all()[0]

        assert full_mem.tag.pending_appraisal is False
        assert fast_mem.tag.pending_appraisal is True

        result = em_fast.elaborate_pending()
        assert len(result) == 1
        assert result[0].tag.pending_appraisal is False


# ---------------------------------------------------------------------------
# auto_categorize config
# ---------------------------------------------------------------------------


class TestAutoCategorize:
    def test_auto_categorize_sets_emotion_label(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=EmotionalMemoryConfig(auto_categorize=True),
        )
        # (0.8, 0.5) → a_scaled=0.0 → angle=0° → joy sector
        em.set_affect(CoreAffect(valence=0.8, arousal=0.5))
        mem = em.encode("A joyful event!")
        assert mem.tag.emotion_label is not None
        assert mem.tag.emotion_label.primary == "joy"

    def test_auto_categorize_disabled_leaves_label_none(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=EmotionalMemoryConfig(auto_categorize=False),
        )
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        mem = em.encode("Unlabelled memory.")
        assert mem.tag.emotion_label is None


# ---------------------------------------------------------------------------
# encode_batch() — dual_path_encoding and auto_categorize
# ---------------------------------------------------------------------------


class TestEncodeBatch:
    def test_batch_dual_path_sets_pending_appraisal(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        memories = em.encode_batch(["First.", "Second.", "Third."])
        assert all(m.tag.pending_appraisal is True for m in memories)

    def test_batch_dual_path_skips_appraisal_vector(self):
        em = _engine(dual_path=True)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        memories = em.encode_batch(["Fast one.", "Fast two."])
        assert all(m.tag.appraisal is None for m in memories)

    def test_batch_no_dual_path_runs_full_appraisal(self):
        em = _engine(dual_path=False)
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        memories = em.encode_batch(["Full one.", "Full two."])
        assert all(m.tag.pending_appraisal is False for m in memories)
        assert all(m.tag.appraisal is not None for m in memories)

    def test_batch_auto_categorize_sets_emotion_label(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=EmotionalMemoryConfig(auto_categorize=True),
        )
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        memories = em.encode_batch(["Joyful A.", "Joyful B."])
        assert all(m.tag.emotion_label is not None for m in memories)

    def test_batch_auto_categorize_disabled_leaves_label_none(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=EmotionalMemoryConfig(auto_categorize=False),
        )
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        memories = em.encode_batch(["A.", "B."])
        assert all(m.tag.emotion_label is None for m in memories)
