"""Advanced configuration — decay, Stimmung decay, and adaptive weights.

Demonstrates how to tune the ACT-R decay engine, time-based Stimmung
regression, and the adaptive retrieval weight system. Includes side-by-side
comparisons showing how config choices change retrieval rankings.

Run with:
    python examples/advanced_config.py
"""

from datetime import UTC, datetime, timedelta

from emotional_memory import (
    AdaptiveWeightsConfig,
    CoreAffect,
    DecayConfig,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    RetrievalConfig,
    StimmungDecayConfig,
)
from emotional_memory.decay import compute_effective_strength

# ---------------------------------------------------------------------------
# Minimal embedder — no ML dependencies required
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic 8-dim embedder based on string hashing."""

    DIM = 8

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(self.DIM)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Helper — encode the same four memories into a given engine
# ---------------------------------------------------------------------------


def seed_memories(em: EmotionalMemory) -> None:
    em.set_affect(CoreAffect(valence=0.9, arousal=0.9))
    em.encode("Record quarter — all targets exceeded by 40%.")

    em.set_affect(CoreAffect(valence=-0.8, arousal=0.8))
    em.encode("Production outage on launch day — five hours of downtime.")

    em.set_affect(CoreAffect(valence=0.4, arousal=0.2))
    em.encode("Quiet sprint — mostly documentation and minor refactors.")

    em.set_affect(CoreAffect(valence=0.7, arousal=0.5))
    em.encode("One-on-one with the CEO — positive feedback on the roadmap.")


# ---------------------------------------------------------------------------
# DecayConfig — aggressive vs. floor-protected
# ---------------------------------------------------------------------------

print("=== DecayConfig comparison ===\n")

# Default: moderate decay, floor kicks in for high-arousal memories
em_default = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(decay=DecayConfig()),
)
seed_memories(em_default)

# Aggressive: faster decay, no meaningful floor
em_fast = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        decay=DecayConfig(
            base_decay=2.0,  # much steeper power-law exponent
            arousal_modulation=0.1,  # arousal barely slows decay
            floor_value=0.01,  # floor is almost zero
        )
    ),
)
seed_memories(em_fast)

now = datetime.now(UTC)
print(f"{'Memory':<45}  {'default':>8}  {'fast':>8}")
print("-" * 65)
for mem_d, mem_f in zip(em_default.list_all(), em_fast.list_all(), strict=False):
    s_d = compute_effective_strength(mem_d.tag, now, DecayConfig())
    s_f = compute_effective_strength(
        mem_f.tag,
        now,
        DecayConfig(base_decay=2.0, arousal_modulation=0.1, floor_value=0.01),
    )
    arousal = mem_d.tag.core_affect.arousal
    print(f"  {mem_d.content[:42]:<42}  {s_d:8.4f}  {s_f:8.4f}  (arousal={arousal:.1f})")

print("\nHigh-arousal memories (outage, record quarter) retain more strength")
print("under the default config thanks to the arousal_modulation floor.\n")

# ---------------------------------------------------------------------------
# StimmungDecayConfig — mood regresses toward baseline over time
# ---------------------------------------------------------------------------

print("=== StimmungDecayConfig — time-based mood regression ===\n")

em_sd = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        stimmung_alpha=0.3,
        stimmung_decay=StimmungDecayConfig(
            base_half_life_seconds=60.0,  # fast decay for demo (1 minute half-life)
            inertia_scale=1.0,
            baseline_valence=0.0,
            baseline_arousal=0.3,
        ),
    ),
)

em_sd.set_affect(CoreAffect(valence=0.9, arousal=0.9))
em_sd.set_affect(CoreAffect(valence=0.9, arousal=0.9))  # repeat to build Stimmung

sm_now = em_sd.get_current_stimmung()
print("Stimmung right after encoding:")
print(
    f"  valence={sm_now.valence:.3f}  arousal={sm_now.arousal:.3f}  inertia={sm_now.inertia:.3f}"
)

# Simulate what Stimmung would be an hour from now (read-only, no mutation)
sm_future = em_sd.get_current_stimmung(now=datetime.now(UTC) + timedelta(hours=1))
print("\nStimmung projected 1 hour from now (toward baseline):")
print(f"  valence={sm_future.valence:.3f}  arousal={sm_future.arousal:.3f}")
print("  baseline: valence=0.000  arousal=0.300")
print("\nget_current_stimmung() is read-only — the stored state is unchanged.")
sm_check = em_sd.get_state().stimmung
print(f"Stored stimmung still: valence={sm_check.valence:.3f}\n")

# ---------------------------------------------------------------------------
# AdaptiveWeightsConfig — how Stimmung modulates retrieval signal weights
# ---------------------------------------------------------------------------

print("=== AdaptiveWeightsConfig — Stimmung-driven weight modulation ===\n")

# Exaggerated config: strong negative-mood boost to emotional signals
em_aw = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            base_weights=[0.30, 0.20, 0.20, 0.10, 0.15, 0.05],
            adaptive_weights_config=AdaptiveWeightsConfig(
                negative_mood_strength=0.25,  # stronger emotional boost under bad mood
                negative_mood_center=-0.3,  # activates earlier (less negative threshold)
                calm_strength=0.25,  # stronger semantic boost when calm
            ),
        ),
    ),
)
seed_memories(em_aw)

# Retrieval under positive Stimmung (calm semantic search dominates)
em_aw.set_affect(CoreAffect(valence=0.7, arousal=0.2))  # calm
results_calm = em_aw.retrieve("work performance review", top_k=4)

# Retrieval under negative Stimmung (emotional congruence dominates)
for _ in range(6):
    em_aw.set_affect(CoreAffect(valence=-0.9, arousal=0.7))
results_neg = em_aw.retrieve("work performance review", top_k=4)

print(f"{'Rank':<5}  {'Calm mood (semantic bias)':<45}  {'Negative mood (emotional bias)'}")
print("-" * 100)
for i, (r_calm, r_neg) in enumerate(zip(results_calm, results_neg, strict=False), 1):
    c = r_calm.content[:40]
    n = r_neg.content[:40]
    cv = r_calm.tag.core_affect.valence
    nv = r_neg.tag.core_affect.valence
    print(f"  {i}    {c:<40} ({cv:+.2f})   {n:<40} ({nv:+.2f})")

print("\nUnder negative mood, high-arousal / negative-valence memories rise in ranking.")

print("\nDone.")
