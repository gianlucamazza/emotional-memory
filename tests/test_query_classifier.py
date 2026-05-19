"""Tests for query_classifier module: HeuristicQueryClassifier, LLMQueryClassifier, routing."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from emotional_memory.query_classifier import (
    LOCOMO_ROUTING,
    QUERY_TYPES,
    HeuristicQueryClassifier,
    LLMQueryClassifier,
    QueryClassifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_em(**kwargs: Any) -> Any:
    from emotional_memory import EmotionalMemory, InMemoryStore

    return EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), **kwargs)


def _llm_returning(query_type: str) -> Any:
    """Return a fake LLMCallable that always emits a fixed query_type."""
    mock = MagicMock()
    mock.return_value = f'{{"query_type": "{query_type}"}}'
    return mock


# ---------------------------------------------------------------------------
# LOCOMO_ROUTING and QUERY_TYPES constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_query_types_is_frozenset(self) -> None:
        assert isinstance(QUERY_TYPES, frozenset)

    def test_query_types_contains_expected(self) -> None:
        assert {"single_hop", "multi_hop", "temporal", "open_domain", "default"} == QUERY_TYPES

    def test_locomo_routing_keys(self) -> None:
        assert set(LOCOMO_ROUTING.keys()) == {
            "single_hop",
            "multi_hop",
            "temporal",
            "open_domain",
            "default",
        }

    def test_locomo_routing_all_six_weights(self) -> None:
        for key, weights in LOCOMO_ROUTING.items():
            assert len(weights) == 6, f"{key} must have 6 weights, got {len(weights)}"

    def test_locomo_routing_weights_sum_to_one(self) -> None:
        import math

        for key, weights in LOCOMO_ROUTING.items():
            assert math.isclose(sum(weights), 1.0, rel_tol=1e-9), f"{key} weights don't sum to 1"

    def test_locomo_routing_single_hop_high_semantic(self) -> None:
        # W7: suppress affect, maximise semantic
        assert LOCOMO_ROUTING["single_hop"][0] >= 0.60

    def test_locomo_routing_temporal_equals_multi_hop(self) -> None:
        # Both map to W2 per Addendum J closure
        assert LOCOMO_ROUTING["temporal"] == LOCOMO_ROUTING["multi_hop"]


# ---------------------------------------------------------------------------
# QueryClassifier Protocol
# ---------------------------------------------------------------------------


class TestQueryClassifierProtocol:
    def test_heuristic_satisfies_protocol(self) -> None:
        clf = HeuristicQueryClassifier()
        assert isinstance(clf, QueryClassifier)

    def test_llm_satisfies_protocol(self) -> None:
        clf = LLMQueryClassifier(llm=_llm_returning("single_hop"))
        assert isinstance(clf, QueryClassifier)

    def test_custom_class_satisfies_protocol(self) -> None:
        class MyClassifier:
            def classify(self, query: str) -> str:
                return "default"

        assert isinstance(MyClassifier(), QueryClassifier)


# ---------------------------------------------------------------------------
# HeuristicQueryClassifier
# ---------------------------------------------------------------------------


class TestHeuristicQueryClassifier:
    def setup_method(self) -> None:
        self.clf = HeuristicQueryClassifier()

    # Temporal ---

    def test_temporal_when(self) -> None:
        assert self.clf.classify("When did she leave?") == "temporal"

    def test_temporal_before(self) -> None:
        assert self.clf.classify("Did he arrive before the meeting?") == "temporal"

    def test_temporal_after(self) -> None:
        assert self.clf.classify("What happened after the storm?") == "temporal"

    def test_temporal_how_long(self) -> None:
        assert self.clf.classify("How long did the trip take?") == "temporal"

    def test_temporal_since(self) -> None:
        assert self.clf.classify("Since when have they been friends?") == "temporal"

    def test_temporal_until(self) -> None:
        assert self.clf.classify("How much longer until dinner?") == "temporal"

    def test_temporal_how_many_years(self) -> None:
        assert self.clf.classify("How many years did they wait?") == "temporal"

    def test_temporal_how_often(self) -> None:
        assert self.clf.classify("How often does she exercise?") == "temporal"

    def test_temporal_did_before(self) -> None:
        assert self.clf.classify("Did he call her before leaving?") == "temporal"

    # Multi-hop ---

    def test_multi_hop_and_then(self) -> None:
        assert self.clf.classify("She went to school and then became a doctor.") == "multi_hop"

    def test_multi_hop_both(self) -> None:
        assert self.clf.classify("Both Alice and Bob attended the conference.") == "multi_hop"

    def test_multi_hop_also(self) -> None:
        assert self.clf.classify("He also enjoyed painting in his spare time.") == "multi_hop"

    def test_multi_hop_in_addition(self) -> None:
        assert self.clf.classify("In addition, she worked as a nurse.") == "multi_hop"

    def test_multi_hop_as_well_as(self) -> None:
        assert self.clf.classify("She likes hiking as well as swimming.") == "multi_hop"

    def test_multi_hop_furthermore(self) -> None:
        assert self.clf.classify("Furthermore, he studied abroad.") == "multi_hop"

    # Single-hop ---

    def test_single_hop_who(self) -> None:
        assert self.clf.classify("Who is her best friend?") == "single_hop"

    def test_single_hop_what(self) -> None:
        assert self.clf.classify("What is her job?") == "single_hop"

    def test_single_hop_where(self) -> None:
        assert self.clf.classify("Where does he live?") == "single_hop"

    def test_single_hop_which(self) -> None:
        assert self.clf.classify("Which team did she support?") == "single_hop"

    def test_single_hop_is(self) -> None:
        assert self.clf.classify("Is she a teacher?") == "single_hop"

    def test_single_hop_did(self) -> None:
        assert self.clf.classify("Did he pass the exam?") == "single_hop"

    def test_single_hop_was(self) -> None:
        assert self.clf.classify("Was she happy?") == "single_hop"

    def test_single_hop_too_long_falls_to_open_domain(self) -> None:
        long_q = "Who is the person that " + "x " * 40 + "?"
        assert len(long_q) > 100
        assert self.clf.classify(long_q) == "open_domain"

    # Open-domain ---

    def test_open_domain_narrative(self) -> None:
        result = self.clf.classify("Tell me about her relationship with her sister.")
        assert result == "open_domain"

    def test_open_domain_broad(self) -> None:
        result = self.clf.classify(
            "Describe her personality and how it shaped her career choices."
        )
        assert result == "open_domain"

    # Priority: temporal beats multi_hop ---

    def test_temporal_beats_multi_hop(self) -> None:
        q = "When did both of them arrive?"  # temporal AND multi_hop marker
        assert self.clf.classify(q) == "temporal"

    # Priority: temporal beats single_hop ---

    def test_temporal_beats_single_hop(self) -> None:
        q = "When did she start?"  # "when" is temporal; short + interrogative is single_hop
        assert self.clf.classify(q) == "temporal"

    # Edge cases ---

    def test_empty_string_returns_open_domain(self) -> None:
        assert self.clf.classify("") == "open_domain"

    def test_whitespace_only_returns_open_domain(self) -> None:
        assert self.clf.classify("   ") == "open_domain"

    def test_repr(self) -> None:
        assert repr(self.clf) == "HeuristicQueryClassifier()"


# ---------------------------------------------------------------------------
# LLMQueryClassifier
# ---------------------------------------------------------------------------


class TestLLMQueryClassifier:
    def test_classify_returns_llm_result(self) -> None:
        llm = _llm_returning("single_hop")
        clf = LLMQueryClassifier(llm=llm)
        assert clf.classify("What is her job?") == "single_hop"

    def test_llm_called_once_per_unique_query(self) -> None:
        llm = _llm_returning("multi_hop")
        clf = LLMQueryClassifier(llm=llm)
        clf.classify("Tell me about her family and career.")
        clf.classify("Tell me about her family and career.")
        assert llm.call_count == 1  # cached on second call

    def test_cache_hit_different_query_calls_llm_again(self) -> None:
        llm = _llm_returning("open_domain")
        clf = LLMQueryClassifier(llm=llm)
        clf.classify("First query.")
        clf.classify("Second query.")
        assert llm.call_count == 2

    def test_cache_eviction_respects_size(self) -> None:
        call_count = 0

        def counting_llm(prompt: str, schema: Any) -> str:
            nonlocal call_count
            call_count += 1
            return '{"query_type": "open_domain"}'

        clf = LLMQueryClassifier(llm=counting_llm, cache_size=2)
        clf.classify("query A")
        clf.classify("query B")
        clf.classify("query C")  # evicts A
        clf.classify("query A")  # cache miss — A was evicted
        assert call_count == 4

    def test_fallback_on_error_default(self) -> None:
        def bad_llm(prompt: str, schema: Any) -> str:
            raise RuntimeError("LLM unavailable")

        clf = LLMQueryClassifier(llm=bad_llm, default_type="open_domain")
        assert clf.classify("anything") == "open_domain"

    def test_fallback_on_error_false_raises(self) -> None:
        def bad_llm(prompt: str, schema: Any) -> str:
            raise RuntimeError("LLM unavailable")

        clf = LLMQueryClassifier(llm=bad_llm, fallback_on_error=False)
        with pytest.raises(RuntimeError):
            clf.classify("anything")

    def test_malformed_json_triggers_fallback(self) -> None:
        def bad_json_llm(prompt: str, schema: Any) -> str:
            return "not json at all"

        clf = LLMQueryClassifier(llm=bad_json_llm, default_type="default")
        assert clf.classify("anything") == "default"

    def test_missing_key_uses_default_type(self) -> None:
        def llm_no_key(prompt: str, schema: Any) -> str:
            return '{"other_key": "single_hop"}'

        clf = LLMQueryClassifier(llm=llm_no_key, default_type="multi_hop")
        assert clf.classify("anything") == "multi_hop"

    def test_thread_safety(self) -> None:
        results: list[str] = []
        lock = threading.Lock()
        llm = _llm_returning("temporal")
        clf = LLMQueryClassifier(llm=llm, cache_size=64)

        def worker() -> None:
            r = clf.classify("When did she arrive?")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == "temporal" for r in results)
        assert len(results) == 20

    def test_json_in_code_fence_extracted(self) -> None:
        def fenced_llm(prompt: str, schema: Any) -> str:
            return '```json\n{"query_type": "temporal"}\n```'

        clf = LLMQueryClassifier(llm=fenced_llm)
        assert clf.classify("When?") == "temporal"

    def test_repr_shows_cache_info(self) -> None:
        clf = LLMQueryClassifier(llm=_llm_returning("default"), cache_size=128)
        r = repr(clf)
        assert "LLMQueryClassifier" in r
        assert "cache_size=128" in r
        assert "cached=0" in r


# ---------------------------------------------------------------------------
# QueryClassifierConfig validation
# ---------------------------------------------------------------------------


class TestQueryClassifierConfig:
    def test_disabled_mode_default(self) -> None:
        from emotional_memory.retrieval import QueryClassifierConfig

        cfg = QueryClassifierConfig()
        assert cfg.mode == "disabled"

    def test_wrong_weight_length_raises(self) -> None:
        from pydantic import ValidationError

        from emotional_memory.retrieval import QueryClassifierConfig

        with pytest.raises(ValidationError):
            QueryClassifierConfig(routed_weights={"single_hop": [0.5, 0.5]})

    def test_correct_weights_accepted(self) -> None:
        from emotional_memory.retrieval import QueryClassifierConfig

        cfg = QueryClassifierConfig(
            mode="heuristic",
            routed_weights=LOCOMO_ROUTING,
        )
        assert cfg.mode == "heuristic"
        assert len(cfg.routed_weights["single_hop"]) == 6

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        from emotional_memory.retrieval import QueryClassifierConfig

        cfg = QueryClassifierConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.mode = "llm"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration: EmotionalMemory + QueryClassifierConfig
# ---------------------------------------------------------------------------


class TestQueryRoutingIntegration:
    def _make_em_with_routing(self, **kwargs: Any) -> Any:
        from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
        from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

        qcc = QueryClassifierConfig(mode="heuristic", routed_weights=LOCOMO_ROUTING)
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(query_classifier=qcc),
        )
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            config=config,
            **kwargs,
        )

    def test_retrieve_with_heuristic_routing_returns_results(self) -> None:
        em = self._make_em_with_routing()
        em.encode("She works as a nurse in Paris.")
        em.encode("He studied medicine for six years.")
        results = em.retrieve("What is her job?", top_k=2)
        assert isinstance(results, list)

    def test_retrieve_with_explanations_returns_results(self) -> None:
        em = self._make_em_with_routing()
        em.encode("She works as a nurse in Paris.")
        results = em.retrieve_with_explanations("Who is a nurse?", top_k=1)
        assert isinstance(results, list)

    def test_no_classifier_config_backward_compat(self) -> None:
        from emotional_memory import EmotionalMemory, InMemoryStore

        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
        em.encode("She works as a nurse in Paris.")
        results = em.retrieve("What is her job?", top_k=1)
        assert isinstance(results, list)

    def test_disabled_mode_behaves_as_no_classifier(self) -> None:
        from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
        from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

        qcc = QueryClassifierConfig(mode="disabled")
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), config=config)
        em.encode("Test memory.")
        results = em.retrieve("Test?", top_k=1)
        assert isinstance(results, list)

    def test_llm_mode_classifier_used(self) -> None:
        from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
        from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

        llm = _llm_returning("single_hop")
        qcc = QueryClassifierConfig(mode="llm", routed_weights=LOCOMO_ROUTING)
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_FixedEmbedder(),
            config=config,
            query_classifier=LLMQueryClassifier(llm=llm),
        )
        em.encode("She is a doctor.")
        results = em.retrieve("What is her job?", top_k=1)
        assert isinstance(results, list)
        assert llm.call_count >= 1

    def test_missing_routed_key_falls_back_to_default_type(self) -> None:
        from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
        from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

        # Only "default" key present — temporal query should fall back to default weights
        partial_routing = {"default": LOCOMO_ROUTING["default"]}
        qcc = QueryClassifierConfig(
            mode="heuristic",
            routed_weights=partial_routing,
            default_type="default",
        )
        config = EmotionalMemoryConfig(retrieval=RetrievalConfig(query_classifier=qcc))
        em = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder(), config=config)
        em.encode("She visited Paris last summer.")
        # "When" query → temporal → key missing → fall back to "default" → still works
        results = em.retrieve("When did she travel?", top_k=1)
        assert isinstance(results, list)
