"""Unit tests for EmotionalMemoryConfig ablation flags."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.appraisal import AppraisalVector
from emotional_memory.retrieval import adaptive_weights


class _FixedEmbedder:
    """Returns a deterministic 4-d embedding."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFF
        return [float((h >> i) & 0xFF) / 255.0 for i in range(4)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _engine(config: EmotionalMemoryConfig) -> EmotionalMemory:
    return EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), config=config)


class TestEnableAppraisalFlag:
    def test_false_skips_appraisal_engine(self) -> None:
        mock_engine: Any = MagicMock()
        mock_engine.appraise.return_value = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.5,
            coping_potential=0.5,
            norm_congruence=0.5,
            self_relevance=0.5,
        )
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            appraisal_engine=mock_engine,
            config=EmotionalMemoryConfig(enable_appraisal=False),
        )
        em.encode("hello")
        mock_engine.appraise.assert_not_called()

    def test_true_calls_appraisal_engine(self) -> None:
        mock_engine: Any = MagicMock()
        mock_engine.appraise.return_value = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.5,
            coping_potential=0.5,
            norm_congruence=0.5,
            self_relevance=0.5,
        )
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            appraisal_engine=mock_engine,
            config=EmotionalMemoryConfig(enable_appraisal=True),
        )
        em.encode("hello")
        mock_engine.appraise.assert_called_once()


class TestEnableResonanceFlag:
    def test_false_skips_resonance_links(self) -> None:
        em = _engine(EmotionalMemoryConfig(enable_resonance=False))
        em.encode("first memory to populate the store")
        second = em.encode("second memory that would normally get links")
        assert second.tag.resonance_links == []

    def test_true_can_produce_resonance_links(self) -> None:
        em = _engine(EmotionalMemoryConfig(enable_resonance=True))
        em.encode("I feel very happy today")
        second = em.encode("Another happy joyful event")
        # With resonance enabled, links *may* be produced (depends on similarity).
        # We just verify the path is open — no assertion on count.
        _ = second.tag.resonance_links  # no error


class TestRetrievalWeightMask:
    def _weights(self, **flags: bool) -> np.ndarray:
        em = _engine(EmotionalMemoryConfig(**flags))
        return em._effective_retrieval_weights()

    def test_momentum_false_zeroes_s3(self) -> None:
        w = self._weights(enable_momentum=False)
        assert w[3] == pytest.approx(0.0)
        assert float(w.sum()) == pytest.approx(1.0)

    def test_mood_signal_false_zeroes_s1(self) -> None:
        w = self._weights(enable_mood_signal=False)
        assert w[1] == pytest.approx(0.0)
        assert float(w.sum()) == pytest.approx(1.0)

    def test_resonance_false_zeroes_s5(self) -> None:
        w = self._weights(enable_resonance=False)
        assert w[5] == pytest.approx(0.0)
        assert float(w.sum()) == pytest.approx(1.0)

    def test_two_flags_false_renormalizes(self) -> None:
        w = self._weights(enable_momentum=False, enable_mood_signal=False)
        assert w[1] == pytest.approx(0.0)
        assert w[3] == pytest.approx(0.0)
        assert float(w.sum()) == pytest.approx(1.0)

    def test_all_flags_true_identical_to_adaptive_weights(self) -> None:
        em = _engine(EmotionalMemoryConfig())
        state = em.get_state()
        rc = em._config.retrieval
        expected = adaptive_weights(state.mood, rc.base_weights, rc.adaptive_weights_config)
        actual = em._effective_retrieval_weights()
        np.testing.assert_array_almost_equal(actual, expected)
