"""Regression tests for excellence-audit fixes."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from emotional_memory import (
    AsyncEmotionalMemory,
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    KeywordAppraisalEngine,
)
from emotional_memory.async_adapters import SyncToAsyncEmbedder, SyncToAsyncStore
from emotional_memory.engine_shared import is_query_affect_neutral, semantic_only_weights
from emotional_memory.integrations.mem0 import EmotionalMemoryMem0Backend
from emotional_memory.query_classifier import LOCOMO_ROUTING, LLMQueryClassifier
from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig
from emotional_memory.state_stores.redis import RedisAffectiveStateStore


class _FixedEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _routing_em() -> EmotionalMemory:
    qcc = QueryClassifierConfig(mode="heuristic", routed_weights=LOCOMO_ROUTING)
    config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
    return EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), config=config)


def _routing_async_em() -> AsyncEmotionalMemory:
    qcc = QueryClassifierConfig(mode="heuristic", routed_weights=LOCOMO_ROUTING)
    config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
    return AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(_FixedEmbedder()),
        config=config,
    )


def _semantic_weight(explanations: list[Any]) -> float:
    assert explanations
    return float(explanations[0].breakdown.weights.semantic_similarity)


class TestAsyncExplanationsRouting:
    def test_sync_and_async_use_routed_weights_for_single_hop(self) -> None:
        sync = _routing_em()
        sync.encode("She works as a nurse in Paris.")
        sync_w = _semantic_weight(sync.retrieve_with_explanations("What is her job?", top_k=1))

        async def _run() -> float:
            async_em = _routing_async_em()
            await async_em.encode("She works as a nurse in Paris.")
            exps = await async_em.retrieve_with_explanations("What is her job?", top_k=1)
            return _semantic_weight(exps)

        async_w = asyncio.run(_run())
        assert sync_w == pytest.approx(async_w)
        assert sync_w >= 0.60


class TestRetrieveQueryGated:
    def test_neutral_query_uses_semantic_only_weights(self) -> None:
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            appraisal_engine=KeywordAppraisalEngine(),
        )
        em.encode("Paris is the capital of France.")
        exps = em.retrieve_with_explanations(
            "What is the capital?",
            top_k=1,
            precomputed_weights=semantic_only_weights(),
        )
        assert exps[0].breakdown.weights.mood_congruence == pytest.approx(0.0)

    def test_gated_routes_neutral_to_semantic_arm(self) -> None:
        class _NeutralAppraisal:
            def appraise(self, text: str, context: dict[str, Any] | None = None) -> Any:
                class _Result:
                    @staticmethod
                    def to_core_affect() -> CoreAffect:
                        return CoreAffect(valence=0.05, arousal=0.3)

                return _Result()

        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            appraisal_engine=_NeutralAppraisal(),
        )
        em.encode("Item A")
        em.encode("Item B")
        gated = em.retrieve_query_gated("neutral query", top_k=1)
        direct = em.retrieve("neutral query", top_k=1, precomputed_weights=semantic_only_weights())
        assert gated[0].id == direct[0].id

    def test_gated_requires_appraisal_engine(self) -> None:
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
        with pytest.raises(RuntimeError, match="retrieve_query_gated"):
            em.retrieve_query_gated("query")


class TestEngineShared:
    def test_neutral_boundary_strict_less_than(self) -> None:
        assert is_query_affect_neutral(0.19, 0.2)
        assert not is_query_affect_neutral(0.2, 0.2)


class TestMaxContentLength:
    def test_encode_rejects_oversized_content(self) -> None:
        config = EmotionalMemoryConfig(max_content_length=10)
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), config=config)
        with pytest.raises(ValueError, match="max_content_length"):
            em.encode("this content is too long")


class TestPruneThreshold:
    def test_invalid_threshold_raises(self) -> None:
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
        with pytest.raises(ValueError, match="prune threshold"):
            em.prune(threshold=1.5)


class TestMem0Get:
    def test_get_returns_memory_by_id(self) -> None:
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
        mem = em.encode("stored once")
        backend = EmotionalMemoryMem0Backend(em)
        result = backend.get(mem.id)
        assert result is not None
        assert result["memory"] == "stored once"
        assert backend.get("nonexistent-id") is None


class TestRedisStrict:
    def test_strict_mode_reraises_on_save_failure(self) -> None:
        class _BrokenClient:
            def set(self, key: str, value: str) -> None:
                raise ConnectionError("redis down")

        from emotional_memory.state import AffectiveState

        store = RedisAffectiveStateStore(client=_BrokenClient(), strict=True)
        with pytest.raises(ConnectionError, match="redis down"):
            store.save(AffectiveState.initial())


class TestLLMQueryClassifierJson:
    def test_nested_json_extracted(self) -> None:
        llm = MagicMock(return_value='{"meta": {"nested": true}, "query_type": "single_hop"}')
        clf = LLMQueryClassifier(llm=llm)
        assert clf.classify("Who is she?") == "single_hop"
