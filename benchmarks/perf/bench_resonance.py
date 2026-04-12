"""Performance benchmarks: resonance graph build time and spreading activation.

Run with:
    pytest benchmarks/perf/bench_resonance.py --benchmark-only --benchmark-sort=mean
"""

import pytest

from benchmarks.conftest import ScalableEmbedder, make_engine, populate_store
from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.models import Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.resonance import (
    ResonanceConfig,
    build_resonance_links,
    spreading_activation,
)


def _make_memories(n: int, dim: int = 64) -> list[Memory]:
    """Create n Memory objects with distinct embeddings."""
    embedder = ScalableEmbedder(dim=dim)
    memories = []
    for i in range(n):
        tag = make_emotional_tag(
            core_affect=CoreAffect(valence=0.5 if i % 2 == 0 else -0.5, arousal=0.5),
            momentum=AffectiveMomentum.zero(),
            mood=MoodField.neutral(),
            consolidation_strength=0.7,
        )
        mem = Memory.create(
            content=f"memory {i}",
            tag=tag,
            embedding=embedder.embed(f"memory {i}"),
        )
        memories.append(mem)
    return memories


def _make_linked_memories(n: int, links_per_memory: int = 3) -> list[Memory]:
    """Create n memories each with ``links_per_memory`` resonance links to predecessors."""
    base = _make_memories(n)
    result: list[Memory] = []
    for i, mem in enumerate(base):
        links = []
        for j in range(max(0, i - links_per_memory), i):
            links.append(
                ResonanceLink(
                    source_id=mem.id,
                    target_id=base[j].id,
                    strength=0.6,
                    link_type="semantic",
                )
            )
        tag = mem.tag.model_copy(update={"resonance_links": links})
        result.append(mem.model_copy(update={"tag": tag}))
    return result


@pytest.mark.parametrize("n_candidates", [50, 200, 500])
def bench_resonance_build(benchmark, n_candidates):
    """Build resonance links for one new memory against n candidates."""
    memories = _make_memories(n_candidates + 1)
    new_memory = memories[0]
    candidates = memories[1:]
    config = ResonanceConfig(threshold=0.2, max_links=5)

    benchmark(build_resonance_links, new_memory, candidates, config)


def bench_encode_with_large_resonance_graph(benchmark):
    """Full encode cycle including resonance graph scan and backward links (500 memories)."""
    engine = make_engine(resonance_threshold=0.2)
    populate_store(engine, 500)
    engine.set_affect(CoreAffect(valence=0.7, arousal=0.8))

    benchmark(engine.encode, "New memory entering a large resonance graph.")


@pytest.mark.parametrize("n_memories,hops", [(100, 1), (100, 2), (500, 1), (500, 2)])
def bench_spreading_activation(benchmark, n_memories, hops):
    """Spreading activation latency as a function of store size and hop count."""
    memories = _make_linked_memories(n_memories, links_per_memory=5)
    seed_ids = {m.id for m in memories[:5]}

    benchmark(spreading_activation, seed_ids, memories, hops)


def test_resonance_links_count_bounded():
    """build_resonance_links never exceeds max_links regardless of candidates."""
    memories = _make_memories(100)
    new_memory = memories[0]
    candidates = memories[1:]
    config = ResonanceConfig(threshold=0.0, max_links=5)

    links = build_resonance_links(new_memory, candidates, config)
    assert len(links) <= config.max_links


def test_spreading_activation_bounded():
    """spreading_activation activation values are always in (0, 1]."""
    memories = _make_linked_memories(50, links_per_memory=4)
    seed_ids = {m.id for m in memories[:5]}

    act = spreading_activation(seed_ids, memories, hops=3)
    for val in act.values():
        assert 0.0 < val <= 1.0
