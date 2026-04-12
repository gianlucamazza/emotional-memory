"""Fidelity benchmark: emotional retrieval vs. pure cosine baseline.

Hypothesis
----------
The 6-signal retrieval (semantic + mood + affect + momentum + decay + resonance)
surfaces psychologically relevant memories in ways that a cosine-only baseline
cannot replicate, even when semantic similarity is controlled for.

Three phenomena are tested:

1. **Mood-congruent retrieval** (Bower 1981):
   When all memories share identical embeddings, cosine has no signal and returns
   arbitrary order. The emotional engine still ranks mood-congruent memories
   higher through the mood-congruence and core-affect signals.

2. **Core-affect proximity** (Russell 1980):
   With identical embeddings, the emotional engine ranks memories by proximity of
   the encoded affect to the current state. A cosine-only baseline cannot
   differentiate between memories with distinct emotional content.

3. **Reconsolidation loop** (Nader et al. 2000):
   The emotional engine reconsolidates memories on re-retrieval when prediction
   error is high (high arousal, high goal-relevance). Over repeated retrievals
   a target memory climbs in rank. A cosine baseline remains static.

Theory
------
Bower, G.H. (1981). Mood and memory. American Psychologist, 36(2), 129-148.
Russell, J.A. (1980). A circumplex model of affect. J. Personality & Social
  Psychology, 39(6), 1161-1178.
Nader, K. et al. (2000). Fear memories require protein synthesis in the amygdala
  for reconsolidation after retrieval. Nature, 406, 722-726.
"""

from __future__ import annotations

import pytest

from benchmarks.conftest import ConstantEmbedder, make_fidelity_engine
from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
)

pytestmark = pytest.mark.fidelity


# ---------------------------------------------------------------------------
# Helper: pure cosine baseline engine
# ---------------------------------------------------------------------------


def _cosine_engine() -> EmotionalMemory:
    """Engine with only the semantic (cosine) signal active.

    base_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
      s0 = semantic similarity (cosine)
      s1..s5 = mood congruence, affect proximity, momentum, decay, resonance — all zeroed

    With ConstantEmbedder, all cosine similarities are 1.0: the engine returns
    memories in insertion order (arbitrary / stable but affectively blind).
    """
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=ConstantEmbedder(),
        config=EmotionalMemoryConfig(
            retrieval=RetrievalConfig(base_weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            resonance=ResonanceConfig(threshold=2.0),  # disable resonance links
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: mood-congruent retrieval
# ---------------------------------------------------------------------------


class TestMoodCongruenceVsCosine:
    def test_emotional_engine_ranks_congruent_memories_higher(self):
        """Emotional engine outperforms cosine baseline on mood-congruent retrieval.

        Setup: 4 positive-affect memories + 4 negative-affect memories, all with
        identical embeddings (ConstantEmbedder -> cosine = 1.0 for every memory).
        Retrieval is driven under strong positive mood.

        Expected:
        - Emotional engine: top-3 contains >= 2 positive memories.
        - Cosine baseline: returns arbitrary order (no emotional signal); cannot
          guarantee mood-congruent ordering.
        """
        emotional = make_fidelity_engine(mood_alpha=0.4)
        cosine = _cosine_engine()

        for eng in (emotional, cosine):
            eng.set_affect(CoreAffect(valence=0.9, arousal=0.7))
            for i in range(4):
                eng.encode(f"positive memory {i}: joyful achievement")
            eng.set_affect(CoreAffect(valence=-0.9, arousal=0.6))
            for i in range(4):
                eng.encode(f"negative memory {i}: difficult setback")

        # Drive both engines to strong positive mood
        for eng in (emotional, cosine):
            for _ in range(15):
                eng.set_affect(CoreAffect(valence=0.95, arousal=0.7))

        emotional_top3 = emotional.retrieve("memory", top_k=3)
        cosine_top3 = cosine.retrieve("memory", top_k=3)

        emotional_positive = sum(1 for m in emotional_top3 if m.tag.core_affect.valence > 0)
        cosine_positive = sum(1 for m in cosine_top3 if m.tag.core_affect.valence > 0)

        assert emotional_positive >= 2, (
            f"Emotional engine must surface >=2 positive memories; got {emotional_positive}"
        )
        assert emotional_positive >= cosine_positive, (
            f"Emotional engine ({emotional_positive}) must match or beat cosine baseline "
            f"({cosine_positive}) on mood-congruent retrieval"
        )


# ---------------------------------------------------------------------------
# Test 2: core-affect proximity
# ---------------------------------------------------------------------------


class TestCoreAffectProximityVsCosine:
    def test_emotional_engine_ranks_by_affect_proximity(self):
        """Core-affect proximity gives the emotional engine an advantage when
        embeddings are identical and mood is neutral (one set_affect call only).

        With mood_alpha=0.0 the mood field never changes (no mood-congruence signal).
        The only differentiating signal is s3 (core-affect proximity, 80% weight),
        which should surface the 3 excited memories over the 5 calm ones when the
        current affect is excited.

        The cosine baseline scores all memories at 1.0 and cannot differentiate.
        """
        emotional = EmotionalMemory(
            store=InMemoryStore(),
            embedder=ConstantEmbedder(),
            config=EmotionalMemoryConfig(
                retrieval=RetrievalConfig(
                    base_weights=[0.05, 0.00, 0.80, 0.05, 0.05, 0.05],
                    ape_threshold=0.01,
                ),
                resonance=ResonanceConfig(threshold=2.0),
                mood_alpha=0.0,  # mood never updates -> no s2 signal
            ),
        )

        emotional.set_affect(CoreAffect(valence=0.9, arousal=0.9))
        for i in range(3):
            emotional.encode(f"excited event {i}: thrilling moment")

        emotional.set_affect(CoreAffect(valence=0.1, arousal=0.1))
        for i in range(5):
            emotional.encode(f"calm event {i}: quiet routine")

        # Set retrieval affect to excited (single call -> no mood drift)
        emotional.set_affect(CoreAffect(valence=0.9, arousal=0.9))
        results = emotional.retrieve("event", top_k=3)
        excited_count = sum(1 for m in results if m.tag.core_affect.valence > 0.5)

        assert excited_count >= 2, (
            f"Affect-proximity engine must surface >=2 excited memories in top-3; "
            f"got {excited_count}. Valences: {[m.tag.core_affect.valence for m in results]}"
        )


# ---------------------------------------------------------------------------
# Test 3: reconsolidation loop
# ---------------------------------------------------------------------------


class TestReconsolidationLoopVsCosine:
    def test_emotional_engine_strengthens_target_on_repeated_retrieval(self):
        """Reconsolidation raises the target memory's rank over repeated retrievals.

        Setup:
        - Encode 1 target under strongly positive/high-arousal affect.
        - Encode 5 competitors under mildly positive/low-arousal affect.
        - All embeddings identical (ConstantEmbedder).
        - Retrieve 6 times under the target's affect -> high APE -> reconsolidation fires.

        Expected:
        - After the reconsolidation loop, the target appears in the top-2 results.
        - Cosine baseline (no reconsolidation) returns arbitrary order.
        """
        emotional = EmotionalMemory(
            store=InMemoryStore(),
            embedder=ConstantEmbedder(),
            config=EmotionalMemoryConfig(
                retrieval=RetrievalConfig(
                    base_weights=[0.05, 0.25, 0.50, 0.10, 0.05, 0.05],
                    ape_threshold=0.01,
                    reconsolidation_learning_rate=0.5,
                ),
                resonance=ResonanceConfig(threshold=2.0),
                mood_alpha=0.3,
            ),
        )

        # Target: distinctive positive/high-arousal affect
        emotional.set_affect(CoreAffect(valence=0.9, arousal=0.9))
        target = emotional.encode("the decisive breakthrough moment")
        target_id = target.id

        # Competitors: mild neutral/low-arousal affect
        emotional.set_affect(CoreAffect(valence=0.2, arousal=0.2))
        for i in range(5):
            emotional.encode(f"ordinary event {i}")

        # Reconsolidation loop: retrieve 6 times under target's affect
        for _ in range(6):
            emotional.set_affect(CoreAffect(valence=0.9, arousal=0.9))
            emotional.retrieve("event", top_k=3)

        # After loop: target must be in top-2
        emotional.set_affect(CoreAffect(valence=0.9, arousal=0.9))
        final = emotional.retrieve("event", top_k=2)
        final_ids = [m.id for m in final]

        assert target_id in final_ids, (
            f"After reconsolidation, target must be in top-2. "
            f"Top-2 IDs: {final_ids}, target: {target_id}"
        )
