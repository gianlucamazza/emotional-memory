"""Performance benchmarks: resonance graph build time.

Run with:
    pytest benchmarks/perf/bench_resonance.py --benchmark-only --benchmark-sort=mean
"""

import pytest

from benchmarks.conftest import ScalableEmbedder, make_engine, populate_store
from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.resonance import ResonanceConfig, build_resonance_links
from emotional_memory.stimmung import StimmungField


def _make_memories(n: int, dim: int = 64) -> list[Memory]:
    """Create n Memory objects with distinct embeddings."""
    embedder = ScalableEmbedder(dim=dim)
    memories = []
    for i in range(n):
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=0.5 if i % 2 == 0 else -0.5, arousal=0.5),
            momentum=AffectiveMomentum.zero(),
            stimmung=StimmungField.neutral(),
            consolidation_strength=0.7,
        )
        mem = Memory.create(
            content=f"memory {i}",
            tag=tag,
            embedding=embedder.embed(f"memory {i}"),
        )
        memories.append(mem)
    return memories


@pytest.mark.parametrize("n_candidates", [50, 200, 500])
def bench_resonance_build(benchmark, n_candidates):
    """Build resonance links for one new memory against n candidates."""
    memories = _make_memories(n_candidates + 1)
    new_memory = memories[0]
    candidates = memories[1:]
    config = ResonanceConfig(threshold=0.2, max_links=5)

    benchmark(build_resonance_links, new_memory, candidates, config)


def bench_encode_with_large_resonance_graph(benchmark):
    """Full encode cycle including resonance graph scan against 500 memories."""
    engine = make_engine(resonance_threshold=0.2)
    populate_store(engine, 500)
    engine.set_affect(CoreAffect(valence=0.7, arousal=0.8))

    benchmark(engine.encode, "New memory entering a large resonance graph.")


def test_resonance_links_count_bounded():
    """build_resonance_links never exceeds max_links regardless of candidates."""
    memories = _make_memories(100)
    new_memory = memories[0]
    candidates = memories[1:]
    config = ResonanceConfig(threshold=0.0, max_links=5)

    links = build_resonance_links(new_memory, candidates, config)
    assert len(links) <= config.max_links
