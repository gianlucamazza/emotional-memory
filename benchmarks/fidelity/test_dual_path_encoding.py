"""Fidelity benchmark: LeDoux dual-pathway encoding (LeDoux, 1996).

Validates that the fast thalamo-amygdala pathway stores a raw affective tag
immediately, and the slow thalamo-cortical pathway (elaborate()) enriches it
with full cognitive appraisal.

Reference:
  LeDoux, J. E. (1996). The emotional brain: The mysterious underpinnings of
    emotional life. Simon & Schuster.
"""

import pytest

from emotional_memory import CoreAffect, EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine

pytestmark = pytest.mark.fidelity


class _HashEmbedder:
    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(8)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _engine(appraisal_valence: float = 0.7, appraisal_arousal: float = 0.7) -> EmotionalMemory:
    appraisal = AppraisalVector(
        novelty=0.8,
        goal_relevance=0.9,
        coping_potential=0.7,
        norm_congruence=0.6,
        self_relevance=0.8,
    )
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=_HashEmbedder(),
        appraisal_engine=StaticAppraisalEngine(appraisal),
        config=EmotionalMemoryConfig(dual_path_encoding=True, elaboration_learning_rate=0.7),
    )


def test_fast_path_memory_has_no_appraisal():
    """Fast-path memory is tagged without cognitive appraisal (amygdala path)."""
    em = _engine()
    em.set_affect(CoreAffect(valence=0.1, arousal=0.4))
    mem = em.encode("Quick event perceived without deliberate evaluation.")
    assert mem.tag.appraisal is None, "Fast path should skip appraisal"
    assert mem.tag.pending_appraisal is True


def test_slow_path_adds_full_appraisal():
    """elaborate() performs the slow thalamo-cortical appraisal."""
    em = _engine()
    em.set_affect(CoreAffect(valence=0.1, arousal=0.4))
    mem = em.encode("Event awaiting cortical elaboration.")
    assert mem.tag.appraisal is None

    elaborated = em.elaborate(mem.id)
    assert elaborated is not None
    assert elaborated.tag.appraisal is not None, "Slow path should populate AppraisalVector"
    assert elaborated.tag.pending_appraisal is False


def test_elaboration_shifts_core_affect_toward_appraised():
    """core_affect after elaboration blends raw and appraised values.

    The fast path captures raw amygdala response (low valence/arousal).
    The slow path injects a positive appraisal → blended affect should move
    toward the appraised value (elaboration_learning_rate=0.7).
    """
    em = _engine()
    em.set_affect(CoreAffect(valence=-0.5, arousal=0.2))  # raw: negative low-arousal
    mem = em.encode("Event that will be reappraised positively.")
    raw_valence = mem.tag.core_affect.valence

    elaborated = em.elaborate(mem.id)
    assert elaborated is not None
    # The appraisal engine returns high novelty+goal_relevance → positive affect
    # After blending (70% appraised), valence should move toward positive
    assert elaborated.tag.core_affect.valence > raw_valence, (
        f"Elaboration should shift valence positive: raw={raw_valence:.3f} "
        f"elaborated={elaborated.tag.core_affect.valence:.3f}"
    )


def test_elaboration_opens_reconsolidation_window():
    """Slow-path appraisal opens the reconsolidation window (window_opened_at set)."""
    em = _engine()
    em.set_affect(CoreAffect(valence=0.2, arousal=0.5))
    mem = em.encode("Memory entering reconsolidation via elaboration.")
    assert mem.tag.window_opened_at is None

    elaborated = em.elaborate(mem.id)
    assert elaborated is not None
    assert elaborated.tag.window_opened_at is not None, (
        "Slow-path elaboration should open the reconsolidation lability window"
    )


def test_fast_path_memories_still_retrievable():
    """Memories with pending_appraisal=True are ranked by core_affect, momentum, and decay."""
    em = _engine()
    em.set_affect(CoreAffect(valence=0.6, arousal=0.7))
    em.encode("Fast path memory A.")
    em.encode("Fast path memory B.")

    results = em.retrieve("fast path", top_k=2)
    assert len(results) == 2
    assert all(m.tag.pending_appraisal for m in results), (
        "All returned memories should still have pending appraisal"
    )


def test_elaborate_pending_clears_all():
    """elaborate_pending() processes every unelaborated memory in the store."""
    em = _engine()
    for i in range(4):
        em.set_affect(CoreAffect(valence=0.1 * i, arousal=0.5))
        em.encode(f"Pending memory {i}.")

    pending_before = sum(1 for m in em.list_all() if m.tag.pending_appraisal)
    assert pending_before == 4

    elaborated = em.elaborate_pending()
    assert len(elaborated) == 4
    assert all(not m.tag.pending_appraisal for m in em.list_all())
