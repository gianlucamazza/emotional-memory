"""Performance benchmarks: isolate 6-signal scoring vs pure cosine rank.

Answers: how much does AFT scoring cost on a *fixed* candidate set, without
embedder / store adapter noise?

Run with:
    pytest benchmarks/perf/bench_scoring.py --benchmark-only --benchmark-sort=mean
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmarks.conftest import ScalableEmbedder, make_engine, populate_store
from emotional_memory import CoreAffect
from emotional_memory._math import cosine_similarity
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.decay import DecayConfig
from emotional_memory.mood import MoodField
from emotional_memory.resonance import spreading_activation
from emotional_memory.retrieval import RetrievalConfig, build_retrieval_plan


def _fixed_candidates(n: int, dim: int = 64):
    """Build n memories via a disposable engine (hash embedder)."""
    engine = make_engine(resonance_threshold=2.0, dim=dim)
    populate_store(engine, n)
    return engine._store.list_all(), engine


def _cosine_rank(query: list[float], candidates, top_k: int):
    scored = [
        (cosine_similarity(query, m.embedding or []), m)
        for m in candidates
        if m.embedding is not None
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:top_k]]


@pytest.mark.parametrize("n_candidates", [15, 1_000])
def bench_cosine_rank_fixed_pool(benchmark, n_candidates):
    """Pure cosine rank over a fixed candidate list (no store matrix)."""
    candidates, _engine = _fixed_candidates(n_candidates)
    embedder = ScalableEmbedder(dim=64)
    query = embedder.embed("project work accomplishment")

    benchmark(_cosine_rank, query, candidates, 5)


@pytest.mark.parametrize("n_candidates", [15, 1_000])
def bench_aft_plan_fixed_pool(benchmark, n_candidates):
    """Full two-pass build_retrieval_plan on the same fixed candidate list."""
    candidates, engine = _fixed_candidates(n_candidates)
    embedder = ScalableEmbedder(dim=64)
    query = embedder.embed("project work accomplishment")
    now = datetime.now(tz=UTC)
    query_affect = CoreAffect(valence=0.5, arousal=0.6)
    mood = MoodField.neutral()
    momentum = AffectiveMomentum.zero()
    decay = DecayConfig()
    retrieval = RetrievalConfig()
    hops = engine._config.resonance.propagation_hops

    def run() -> None:
        build_retrieval_plan(
            query_embedding=query,
            query_affect=query_affect,
            current_mood=mood,
            current_momentum=momentum,
            candidates=candidates,
            top_k=5,
            now=now,
            decay_config=decay,
            retrieval_config=retrieval,
            propagation_hops=hops,
            spreading_activation_fn=spreading_activation,
        )

    benchmark(run)


def bench_inmemory_search_cached(benchmark):
    """Repeated search_by_embedding on a warm InMemoryStore matrix cache."""
    engine = make_engine(resonance_threshold=2.0)
    populate_store(engine, 1_000)
    store = engine._store
    embedder = ScalableEmbedder(dim=64)
    query = embedder.embed("project work accomplishment")
    # Warm the cache once outside the timed region
    store.search_by_embedding(query, 15)

    benchmark(store.search_by_embedding, query, 15)
