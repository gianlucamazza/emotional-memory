"""Performance benchmarks: retrieval latency vs store size.

Run with:
    pytest benchmarks/perf/bench_retrieve.py --benchmark-only --benchmark-sort=mean
"""

import pytest

from benchmarks.conftest import make_engine, populate_store
from emotional_memory import CoreAffect


@pytest.mark.parametrize("store_size", [100, 1_000, 10_000])
def bench_retrieve_top5(benchmark, store_size):
    """Retrieve top-5 from a store of varying size — measures O(n) scoring."""
    engine = make_engine(resonance_threshold=0.9)
    populate_store(engine, store_size)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))

    benchmark(engine.retrieve, "project work accomplishment", 5)


@pytest.mark.parametrize("top_k", [1, 5, 10, 25])
def bench_retrieve_varying_topk(benchmark, top_k):
    """Retrieve varying top-k from a 1000-memory store."""
    engine = make_engine(resonance_threshold=0.9)
    populate_store(engine, 1_000)
    engine.set_affect(CoreAffect(valence=0.3, arousal=0.5))

    benchmark(engine.retrieve, "query text", top_k)


def bench_retrieve_with_reconsolidation(benchmark):
    """Retrieval with reconsolidation active (low APE threshold)."""
    from benchmarks.conftest import ScalableEmbedder
    from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
    from emotional_memory.retrieval import RetrievalConfig

    engine = EmotionalMemory(
        store=InMemoryStore(),
        embedder=ScalableEmbedder(),
        config=EmotionalMemoryConfig(
            retrieval=RetrievalConfig(ape_threshold=0.01),
        ),
    )
    populate_store(engine, 200)
    engine.set_affect(CoreAffect(valence=-0.9, arousal=0.8))

    benchmark(engine.retrieve, "any query to trigger reconsolidation", 5)
