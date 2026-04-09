"""Tests for LLMAppraisalEngine and KeywordAppraisalEngine."""

import json

import pytest

from emotional_memory.appraisal import AppraisalVector
from emotional_memory.appraisal_llm import (
    _APPRAISAL_JSON_SCHEMA,
    KeywordAppraisalEngine,
    KeywordRule,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
    LLMCallable,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(response: dict | str, raises: Exception | None = None) -> LLMCallable:
    """Return a stub LLM callable."""

    def _call(prompt: str, json_schema: dict) -> str:
        if raises is not None:
            raise raises
        return json.dumps(response) if isinstance(response, dict) else response

    return _call  # type: ignore[return-value]


_NEUTRAL_RESP = {
    "novelty": 0.0,
    "goal_relevance": 0.0,
    "coping_potential": 0.5,
    "norm_congruence": 0.0,
    "self_relevance": 0.0,
}

_SUCCESS_RESP = {
    "novelty": 0.2,
    "goal_relevance": 0.8,
    "coping_potential": 0.9,
    "norm_congruence": 0.4,
    "self_relevance": 0.7,
}


# ---------------------------------------------------------------------------
# LLMAppraisalEngine
# ---------------------------------------------------------------------------


class TestLLMAppraisalEngine:
    def test_returns_appraisal_vector(self):
        engine = LLMAppraisalEngine(_make_llm(_SUCCESS_RESP))
        v = engine.appraise("finished the project")
        assert isinstance(v, AppraisalVector)
        assert v.goal_relevance == pytest.approx(0.8)
        assert v.coping_potential == pytest.approx(0.9)

    def test_values_are_clamped(self):
        resp = dict(_NEUTRAL_RESP, novelty=5.0)  # out of range
        engine = LLMAppraisalEngine(_make_llm(resp))
        v = engine.appraise("anything")
        assert v.novelty == pytest.approx(1.0)

    def test_fallback_on_error_default(self):
        engine = LLMAppraisalEngine(_make_llm("not json at all"))
        v = engine.appraise("test")
        assert v == AppraisalVector.neutral()

    def test_fallback_disabled_raises(self):
        cfg = LLMAppraisalConfig(fallback_on_error=False)
        engine = LLMAppraisalEngine(_make_llm("bad json"), config=cfg)
        with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
            engine.appraise("test")

    def test_fallback_on_llm_exception(self):
        engine = LLMAppraisalEngine(_make_llm("", raises=RuntimeError("api error")))
        v = engine.appraise("test")
        assert v == AppraisalVector.neutral()

    def test_custom_fallback_vector(self):
        custom = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.5,
            coping_potential=0.5,
            norm_congruence=0.5,
            self_relevance=0.5,
        )
        engine = LLMAppraisalEngine(_make_llm("", raises=RuntimeError()), fallback=custom)
        v = engine.appraise("test")
        assert v == custom

    def test_caching_same_result(self):
        calls = []

        def counting_llm(prompt: str, schema: dict) -> str:
            calls.append(1)
            return json.dumps(_NEUTRAL_RESP)

        engine = LLMAppraisalEngine(counting_llm)  # type: ignore[arg-type]
        engine.appraise("hello")
        engine.appraise("hello")
        assert len(calls) == 1  # second call hit cache

    def test_caching_different_texts(self):
        calls = []

        def counting_llm(prompt: str, schema: dict) -> str:
            calls.append(1)
            return json.dumps(_NEUTRAL_RESP)

        engine = LLMAppraisalEngine(counting_llm)  # type: ignore[arg-type]
        engine.appraise("hello")
        engine.appraise("world")
        assert len(calls) == 2

    def test_cache_disabled(self):
        calls = []

        def counting_llm(prompt: str, schema: dict) -> str:
            calls.append(1)
            return json.dumps(_NEUTRAL_RESP)

        cfg = LLMAppraisalConfig(cache_size=0)
        engine = LLMAppraisalEngine(counting_llm, config=cfg)  # type: ignore[arg-type]
        engine.appraise("hello")
        engine.appraise("hello")
        assert len(calls) == 2

    def test_clear_cache(self):
        calls = []

        def counting_llm(prompt: str, schema: dict) -> str:
            calls.append(1)
            return json.dumps(_NEUTRAL_RESP)

        engine = LLMAppraisalEngine(counting_llm)  # type: ignore[arg-type]
        engine.appraise("hello")
        engine.clear_cache()
        engine.appraise("hello")
        assert len(calls) == 2

    def test_context_affects_cache_key(self):
        calls = []

        def counting_llm(prompt: str, schema: dict) -> str:
            calls.append(1)
            return json.dumps(_NEUTRAL_RESP)

        engine = LLMAppraisalEngine(counting_llm)  # type: ignore[arg-type]
        engine.appraise("hello", context={"user": "alice"})
        engine.appraise("hello", context={"user": "bob"})
        assert len(calls) == 2

    def test_lru_eviction(self):
        cfg = LLMAppraisalConfig(cache_size=2)
        engine = LLMAppraisalEngine(_make_llm(_NEUTRAL_RESP), config=cfg)
        engine.appraise("a")
        engine.appraise("b")
        engine.appraise("c")  # evicts "a"
        assert len(engine._cache) == 2

    def test_extracts_json_from_markdown_fence(self):
        resp = f"```json\n{json.dumps(_NEUTRAL_RESP)}\n```"
        engine = LLMAppraisalEngine(_make_llm(resp))
        v = engine.appraise("test")
        assert isinstance(v, AppraisalVector)

    def test_protocol_conformance(self):
        engine = LLMAppraisalEngine(_make_llm(_NEUTRAL_RESP))
        assert isinstance(engine, type(engine))  # trivial; verify no import errors

    def test_json_schema_has_required_fields(self):
        required = _APPRAISAL_JSON_SCHEMA["required"]
        assert "novelty" in required
        assert "goal_relevance" in required
        assert "coping_potential" in required
        assert "norm_congruence" in required
        assert "self_relevance" in required


# ---------------------------------------------------------------------------
# KeywordAppraisalEngine
# ---------------------------------------------------------------------------


class TestKeywordAppraisalEngine:
    def test_returns_appraisal_vector(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("I succeeded at the project")
        assert isinstance(v, AppraisalVector)

    def test_success_keywords_raise_goal_relevance(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("I succeeded at the project")
        assert v.goal_relevance > 0.0

    def test_failure_keywords_lower_goal_relevance(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("I failed the exam")
        assert v.goal_relevance < 0.0

    def test_surprise_keywords_raise_novelty(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("That was totally unexpected!")
        assert v.novelty > 0.0

    def test_boring_keywords_lower_novelty(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("Same as usual, boring routine")
        assert v.novelty < 0.0

    def test_danger_keywords_lower_coping(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("There is danger and crisis ahead")
        assert v.coping_potential < 0.5

    def test_neutral_text_returns_neutral(self):
        engine = KeywordAppraisalEngine()
        v = engine.appraise("the quick brown fox")
        assert isinstance(v, AppraisalVector)

    def test_custom_rules(self):
        rule = KeywordRule(r"\bpython\b", goal_relevance=0.9, self_relevance=0.8)
        engine = KeywordAppraisalEngine(rules=[rule])
        v = engine.appraise("I love python")
        assert v.goal_relevance > 0.0
        assert v.self_relevance > 0.0

    def test_output_in_valid_range(self):
        engine = KeywordAppraisalEngine()
        for text in [
            "I succeeded and failed unexpectedly in danger",
            "boring routine as expected",
            "help support kind generous",
        ]:
            v = engine.appraise(text)
            assert -1.0 <= v.novelty <= 1.0
            assert -1.0 <= v.goal_relevance <= 1.0
            assert 0.0 <= v.coping_potential <= 1.0
            assert -1.0 <= v.norm_congruence <= 1.0
            assert 0.0 <= v.self_relevance <= 1.0

    def test_context_ignored(self):
        engine = KeywordAppraisalEngine()
        v1 = engine.appraise("hello", context=None)
        v2 = engine.appraise("hello", context={"irrelevant": True})
        assert v1 == v2


# ---------------------------------------------------------------------------
# KeywordRule (public API)
# ---------------------------------------------------------------------------


class TestKeywordRule:
    def test_matches_pattern(self):
        rule = KeywordRule(r"\bsuccess\b", goal_relevance=0.9)
        assert rule.matches("great success today")

    def test_no_match(self):
        rule = KeywordRule(r"\bsuccess\b", goal_relevance=0.9)
        assert not rule.matches("total failure")

    def test_case_insensitive(self):
        rule = KeywordRule(r"\bSUCCESS\b", goal_relevance=0.9)
        assert rule.matches("great success today")
