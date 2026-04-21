"""Integration tests that wire a real LLM into the EmotionalMemory pipeline.

Gated behind:
    - pytest.mark.llm
    - EMOTIONAL_MEMORY_LLM_API_KEY env var (skipped if missing)

Run with:
    EMOTIONAL_MEMORY_LLM_API_KEY=... uv run pytest tests/test_llm_integration.py -v -m llm
"""

from __future__ import annotations

import pytest
from conftest import DeterministicEmbedder
from llm_helpers import make_llm_or_skip

from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.appraisal import AppraisalVector
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.models import Memory

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def real_llm() -> object:
    return make_llm_or_skip()


@pytest.fixture
def llm_engine(real_llm: object) -> EmotionalMemory:
    from emotional_memory.appraisal_llm import LLMCallable

    assert isinstance(real_llm, LLMCallable)
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=DeterministicEmbedder(),
        appraisal_engine=LLMAppraisalEngine(
            llm=real_llm,
            config=LLMAppraisalConfig(fallback_on_error=False),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_produces_nonzero_appraisal(llm_engine: EmotionalMemory) -> None:
    """Real LLM should return a non-neutral appraisal for an emotionally charged phrase."""
    memory = llm_engine.encode("I just received a promotion and a significant raise!")
    assert memory.tag.appraisal is not None
    neutral = AppraisalVector.neutral()
    appraisal = memory.tag.appraisal
    assert appraisal != neutral, "LLM returned a neutral (unchanged) appraisal vector"


def test_encode_positive_vs_negative_differ(llm_engine: EmotionalMemory) -> None:
    """Positive and negative phrases should yield different core affect valences."""
    positive = llm_engine.encode("I successfully defended my PhD dissertation!")
    negative = llm_engine.encode("I just learned my closest friend has terminal cancer.")
    assert positive.tag.core_affect.valence > negative.tag.core_affect.valence, (
        f"Expected positive valence ({positive.tag.core_affect.valence:.3f}) "
        f"> negative valence ({negative.tag.core_affect.valence:.3f})"
    )


def test_encode_then_retrieve_returns_results(llm_engine: EmotionalMemory) -> None:
    """Encode multiple memories, then retrieve — should return Memory instances."""
    llm_engine.encode("I finished a challenging project at work.")
    llm_engine.encode("The team celebrated our product launch.")
    llm_engine.encode("I got positive feedback from my manager.")

    results = llm_engine.retrieve("work achievement success", top_k=3)
    assert len(results) > 0
    for mem in results:
        assert isinstance(mem, Memory)


def test_emotional_retrieval_bias(real_llm: object) -> None:
    """With strongly negative affect, negative memories should rank above positive ones.

    Uses a FixedEmbedder so both memories are equidistant from the query —
    semantic similarity is equal for all, leaving mood congruence as the only
    differentiating signal.  This mirrors TestMoodCongruentRetrieval from
    test_integration.py but with real LLM appraisal.
    """
    from conftest import FixedEmbedder

    from emotional_memory.affect import CoreAffect
    from emotional_memory.appraisal_llm import LLMCallable

    assert isinstance(real_llm, LLMCallable)

    # Equal-distance embedder: semantic similarity is identical for all texts
    engine = EmotionalMemory(
        store=InMemoryStore(),
        embedder=FixedEmbedder([0.5, 0.5, 0.0, 0.0]),
        appraisal_engine=LLMAppraisalEngine(
            llm=real_llm,
            config=LLMAppraisalConfig(fallback_on_error=False),
        ),
    )

    neg_memory = engine.encode("My father passed away and I feel completely devastated.")
    pos_memory = engine.encode("I got married and it was the happiest day of my life.")

    # Set strong negative affect before retrieval
    engine.set_affect(CoreAffect(valence=-0.9, arousal=0.7))

    results = engine.retrieve("important life event", top_k=2)
    assert len(results) >= 2

    neg_rank = next((i for i, m in enumerate(results) if m.id == neg_memory.id), None)
    pos_rank = next((i for i, m in enumerate(results) if m.id == pos_memory.id), None)

    assert neg_rank is not None and pos_rank is not None, (
        "Both memories should appear in top-2 results"
    )
    assert neg_rank < pos_rank, (
        f"With negative affect, negative memory should rank higher: "
        f"neg_rank={neg_rank}, pos_rank={pos_rank}"
    )


def test_llm_appraisal_caching(real_llm: object) -> None:
    """Same text twice should hit the cache — only one LLM call should occur."""
    from emotional_memory.appraisal_llm import LLMCallable

    assert isinstance(real_llm, LLMCallable)

    call_count = 0
    original_llm = real_llm

    def counting_llm(prompt: str, json_schema: dict) -> str:  # type: ignore[type-arg]
        nonlocal call_count
        call_count += 1
        return original_llm(prompt, json_schema)

    assert isinstance(counting_llm, LLMCallable)
    engine = LLMAppraisalEngine(
        llm=counting_llm,
        config=LLMAppraisalConfig(cache_size=128, fallback_on_error=False),
    )

    text = "I aced my job interview!"
    engine.appraise(text)
    engine.appraise(text)

    assert call_count == 1, f"Expected 1 LLM call (cache hit on second), got {call_count}"
