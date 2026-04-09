"""Performance benchmarks: encode throughput scaling.

Run with:
    pytest benchmarks/perf/bench_encode.py --benchmark-only --benchmark-sort=mean
"""

import pytest

from benchmarks.conftest import make_engine, populate_store
from emotional_memory import CoreAffect


def bench_encode_single(benchmark):
    """Baseline: single encode into an empty store."""
    engine = make_engine(resonance_threshold=0.9)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    benchmark(engine.encode, "A short memory content for encode throughput benchmarking.")


def bench_encode_with_resonance(benchmark):
    """Encode into a pre-populated store with resonance links active."""
    engine = make_engine(resonance_threshold=0.1)
    populate_store(engine, 100)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    benchmark(engine.encode, "New memory into a populated store with resonance active.")


def bench_encode_no_resonance(benchmark):
    """Encode with resonance effectively disabled (high threshold)."""
    engine = make_engine(resonance_threshold=2.0)
    populate_store(engine, 100)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    benchmark(engine.encode, "New memory with resonance disabled.")


@pytest.mark.parametrize("n_existing", [10, 100, 1_000])
def bench_encode_scaling(benchmark, n_existing):
    """Encode time vs store size — measures O(n) resonance scan."""
    engine = make_engine()
    populate_store(engine, n_existing)
    engine.set_affect(CoreAffect(valence=0.3, arousal=0.5))
    benchmark(engine.encode, f"Scaling test with {n_existing} existing entries.")
