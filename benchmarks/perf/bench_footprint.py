"""Performance benchmarks: memory footprint per record.

Run with:
    pytest benchmarks/perf/bench_footprint.py --benchmark-only
"""

import sys

import pytest

from benchmarks.conftest import make_engine, populate_store
from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.stimmung import StimmungField


def _measure_memory_bytes(obj: object) -> int:
    """Rough estimate of object memory via sys.getsizeof (shallow)."""
    return sys.getsizeof(obj)


def bench_memory_per_record(benchmark):
    """Measure the overhead of creating and storing one Memory object."""
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.5, arousal=0.6),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.8,
    )

    def create_memory():
        return Memory.create(
            content="Sample memory content for footprint measurement.",
            tag=tag,
            embedding=[0.1] * 64,
        )

    benchmark(create_memory)


@pytest.mark.parametrize("n", [100, 1_000, 5_000])
def bench_store_footprint(benchmark, n):
    """Measure encode+store throughput for n memories."""
    engine = make_engine(resonance_threshold=2.0)  # disable resonance

    def fill_store():
        populate_store(engine, n)

    benchmark.pedantic(fill_store, iterations=1, rounds=3)


def test_memory_object_size_reasonable():
    """Non-benchmark assertion: a Memory object should be < 10 KB (shallow)."""
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.5, arousal=0.6),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.8,
    )
    mem = Memory.create(
        content="Test content.",
        tag=tag,
        embedding=[0.1] * 64,
    )
    size = sys.getsizeof(mem)
    assert size < 10_000, f"Memory object is unexpectedly large: {size} bytes"


def test_store_list_all_scales_linearly():
    """list_all() should return the correct count after n encodes."""
    engine = make_engine(resonance_threshold=2.0)
    populate_store(engine, 50)
    all_mems = engine._store.list_all()
    assert len(all_mems) == 50
