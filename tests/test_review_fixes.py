"""Regression tests for the 2026-07 project-review fixes.

Each test pins a specific defect surfaced in the review so it cannot silently
regress. Grouped by the module the fix landed in.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from emotional_memory import (
    AsyncEmotionalMemory,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
)
from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import AppraisalVector, GenericAppraisalVector
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.appraisal_schema import DIRECT_VAD_SCHEMA
from emotional_memory.async_adapters import SyncToAsyncEmbedder, SyncToAsyncStore
from emotional_memory.categorize import categorize_affect
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.state import AffectiveState


class _FixedEmbedder:
    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFF
        return [float((h >> i) & 0xFF) / 255.0 for i in range(4)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _appraisal() -> AppraisalVector:
    return AppraisalVector(
        novelty=0.5,
        goal_relevance=0.5,
        coping_potential=0.5,
        norm_congruence=0.5,
        self_relevance=0.5,
    )


# ---------------------------------------------------------------------------
# Engine: encode_batch must honor ablation flags exactly like encode()
# ---------------------------------------------------------------------------


class TestEncodeBatchAblationParity:
    def test_batch_respects_enable_appraisal_false(self) -> None:
        mock_engine: Any = MagicMock()
        mock_engine.appraise.return_value = _appraisal()
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            appraisal_engine=mock_engine,
            config=EmotionalMemoryConfig(enable_appraisal=False),
        )
        em.encode_batch(["a", "b", "c"])
        # Layer-4 ablation: the engine must not be invoked in the batch path.
        mock_engine.appraise.assert_not_called()

    def test_batch_respects_enable_resonance_false(self) -> None:
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            config=EmotionalMemoryConfig(enable_resonance=False),
        )
        mems = em.encode_batch(["first memory", "second memory", "third memory"])
        assert all(m.tag.resonance_links == [] for m in mems)

    @pytest.mark.asyncio
    async def test_async_batch_respects_enable_appraisal_false(self) -> None:
        mock_engine: Any = MagicMock()
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(_FixedEmbedder()),
            appraisal_engine=mock_engine,
            config=EmotionalMemoryConfig(enable_appraisal=False),
        )
        await em.encode_batch(["a", "b", "c"])
        mock_engine.appraise.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_batch_respects_enable_resonance_false(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(_FixedEmbedder()),
            config=EmotionalMemoryConfig(enable_resonance=False),
        )
        mems = await em.encode_batch(["first memory", "second memory", "third memory"])
        assert all(m.tag.resonance_links == [] for m in mems)


# ---------------------------------------------------------------------------
# categorize: the system baseline must not be a high-confidence emotion
# ---------------------------------------------------------------------------


class TestNeutralCategorization:
    def test_neutral_is_uncategorised(self) -> None:
        label = categorize_affect(CoreAffect.neutral())
        # Was previously "disgust"/"loathing"/high/confidence=1.0.
        assert label.confidence == 0.0
        assert label.intensity == "low"

    def test_near_baseline_is_uncategorised(self) -> None:
        label = categorize_affect(CoreAffect(valence=0.02, arousal=0.02))
        assert label.confidence == 0.0

    def test_genuine_emotion_still_categorised(self) -> None:
        label = categorize_affect(CoreAffect(valence=-0.6, arousal=0.8), dominance=0.2)
        assert label.primary == "fear"
        assert label.confidence > 0.0


# ---------------------------------------------------------------------------
# decay: the arousal floor must not boost a weak memory above its initial value
# ---------------------------------------------------------------------------


class TestDecayFloorInvariant:
    def test_floor_clamped_to_initial_strength(self) -> None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(tz=UTC)
        # Weak (0.05) but highly arousing memory: floor_value (0.1) must not lift it.
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=0.0, arousal=0.95),
            momentum=AffectiveMomentum.zero(),
            mood=MoodField.neutral(),
            consolidation_strength=0.05,
        )
        tag = tag.model_copy(update={"timestamp": now - timedelta(days=30)})
        cfg = DecayConfig(floor_value=0.1, floor_arousal_threshold=0.7)
        strength = compute_effective_strength(tag, now, cfg)
        assert strength <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# appraisal_llm: silent-degradation controls
# ---------------------------------------------------------------------------


def _llm(response: str) -> Any:
    def _call(prompt: str, schema: dict[str, Any]) -> str:
        return response

    return _call


class TestAppraisalIntegrity:
    _GOOD = (
        '{"novelty": 0.1, "goal_relevance": 0.2, "coping_potential": 0.5, '
        '"norm_congruence": 0.0, "self_relevance": 0.3}'
    )

    def test_fallback_count_tracks_parse_errors(self) -> None:
        engine = LLMAppraisalEngine(_llm("not json at all"))
        assert engine.fallback_count == 0
        engine.appraise("something happened")
        assert engine.fallback_count == 1
        engine.reset_fallback_count()
        assert engine.fallback_count == 0

    def test_missing_dimension_is_not_silently_neutral_filled(self) -> None:
        # Only 4 of 5 SECs present — must be treated as a degraded result.
        partial = (
            '{"novelty": 0.1, "goal_relevance": 0.2, '
            '"coping_potential": 0.5, "norm_congruence": 0.0}'
        )
        engine = LLMAppraisalEngine(
            _llm(partial), config=LLMAppraisalConfig(fallback_on_error=False)
        )
        with pytest.raises(ValueError):
            engine.appraise("x")

    def test_nan_is_rejected(self) -> None:
        engine = LLMAppraisalEngine(
            _llm(
                '{"novelty": NaN, "goal_relevance": 0.2, "coping_potential": 0.5, '
                '"norm_congruence": 0.0, "self_relevance": 0.3}'
            ),
            config=LLMAppraisalConfig(fallback_on_error=False),
        )
        with pytest.raises(ValueError):
            engine.appraise("x")

    def test_good_response_no_fallback(self) -> None:
        engine = LLMAppraisalEngine(_llm(self._GOOD))
        v = engine.appraise("x")
        assert isinstance(v, AppraisalVector)
        assert engine.fallback_count == 0

    def test_cache_key_components_do_not_collide(self) -> None:
        # ("ab", "c") vs ("a", "bc") must produce different keys.
        k1 = LLMAppraisalEngine._make_cache_key("c", None, "ab")
        k2 = LLMAppraisalEngine._make_cache_key("bc", None, "a")
        assert k1 != k2

    def test_error_fallback_is_not_cached(self) -> None:
        # A transient failure must not pin neutral for that key; a later success
        # with the same text should surface the real appraisal.
        responses = iter(["not json", self._GOOD])

        def _flaky(prompt: str, schema: dict[str, Any]) -> str:
            return next(responses)

        engine = LLMAppraisalEngine(_flaky)
        first = engine.appraise("same event")
        assert first == AppraisalVector.neutral()
        assert engine.fallback_count == 1
        second = engine.appraise("same event")
        assert isinstance(second, AppraisalVector)
        assert second.novelty == pytest.approx(0.1)
        assert engine.fallback_count == 1  # only the first call fell back


# ---------------------------------------------------------------------------
# Immutability invariants
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_generic_appraisal_vector_dimensions_readonly(self) -> None:
        gav = GenericAppraisalVector(
            dimensions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            schema=DIRECT_VAD_SCHEMA,
        )
        with pytest.raises(TypeError):
            gav.dimensions["valence"] = 0.9  # type: ignore[index]

    def test_affective_state_is_frozen(self) -> None:
        state = AffectiveState.initial()
        with pytest.raises(ValidationError):
            state.core_affect = CoreAffect(valence=1.0, arousal=1.0)  # type: ignore[misc]

    def test_affective_state_update_still_tracks_history(self) -> None:
        # Frozen model must still round-trip momentum history via update().
        s0 = AffectiveState.initial()
        s1 = s0.update(CoreAffect(valence=0.3, arousal=0.4))
        s2 = s1.update(CoreAffect(valence=0.6, arousal=0.5))
        assert not math.isnan(s2.momentum.d_valence)
        snap = s2.snapshot()
        restored = AffectiveState.restore(snap)
        assert restored.core_affect == s2.core_affect
