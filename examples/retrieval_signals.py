"""Retrieval signals — inspecting the public explainable retrieval API.

Demonstrates ``retrieve_with_explanations()`` as the canonical introspection
path for the 6-signal retrieval stack. The example prints:

- ranking-time raw signals
- weighted signals and total score
- pass-1 vs pass-2 rank changes
- adaptive retrieval weights under several Mood scenarios

Requires (for plots):
    pip install emotional-memory[viz]

Run with:
    python examples/retrieval_signals.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import matplotlib.pyplot as plt

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    MoodField,
    ResonanceConfig,
    RetrievalConfig,
)
from emotional_memory.retrieval import RetrievalSignals, adaptive_weights
from emotional_memory.visualization import (
    SIGNAL_LABELS,
    plot_adaptive_weights_heatmap,
    plot_retrieval_radar,
)


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


def _signals_to_list(signals: RetrievalSignals) -> list[float]:
    return [
        signals.semantic_similarity,
        signals.mood_congruence,
        signals.affect_proximity,
        signals.momentum_alignment,
        signals.recency,
        signals.resonance,
    ]


embedder = HashEmbedder()

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=embedder,
    config=EmotionalMemoryConfig(
        resonance=ResonanceConfig(threshold=0.05, max_links=5),
        retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
        mood_alpha=0.3,
    ),
)

events = [
    (
        CoreAffect(valence=0.9, arousal=0.9),
        "Breakthrough: the model finally converged after weeks of tuning.",
    ),
    (
        CoreAffect(valence=-0.8, arousal=0.8),
        "Critical production outage — all services down at peak traffic.",
    ),
    (
        CoreAffect(valence=0.3, arousal=0.2),
        "Quiet afternoon reading the quarterly engineering newsletter.",
    ),
    (
        CoreAffect(valence=0.7, arousal=0.5),
        "Productive pair-programming session with a new team member.",
    ),
]

for affect, text in events:
    em.set_affect(affect)
    em.encode(text)

# Prepare query: set a positive retrieval state
em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
state = em.get_state()
query_text = "technical achievement model training"

ca = state.core_affect
sm = state.mood
print(f"Query: '{query_text}'")
print(f"Current affect: valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}")
print(f"Current mood:   valence={sm.valence:+.3f}  arousal={sm.arousal:.3f}")

print("\n=== Per-memory signal breakdown (public explainable API) ===\n")

explanations = em.retrieve_with_explanations(query_text, top_k=4)
header = (
    f"{'Memory':<48}"
    f"  {'semantic':>8}  {'mood':>8}  {'affect':>8}"
    f"  {'moment':>8}  {'recency':>8}  {'resonance':>10}"
    f"  {'score':>8}  {'p1':>3}  {'p2':>3}"
)
print(header)
print("-" * len(header))

for explanation in explanations:
    raw = explanation.breakdown.raw_signals
    row = (
        f"{explanation.memory.content[:46]:<48}"
        f"  {raw.semantic_similarity:>8.3f}"
        f"  {raw.mood_congruence:>8.3f}"
        f"  {raw.affect_proximity:>8.3f}"
        f"  {raw.momentum_alignment:>8.3f}"
        f"  {raw.recency:>8.3f}"
        f"  {raw.resonance:>10.3f}"
        f"  {explanation.score:>8.3f}"
        f"  {explanation.pass1_rank or 0:>3}"
        f"  {explanation.pass2_rank or 0:>3}"
    )
    if explanation.activation_level > 0.0:
        row += f"  ← activation {explanation.activation_level:.3f}"
    elif explanation.selected_as_seed:
        row += "  ← pass-1 seed"
    print(row)

top = explanations[0]
print(f"\n=== Top memory: '{top.memory.content[:55]}' ===")
print("Raw signals:", top.breakdown.raw_signals.model_dump())
print("Weighted signals:", top.breakdown.weighted_signals.model_dump())
print(f"Score: {top.score:.3f}")
print(
    "Interpretation: ranking uses the breakdown above; the returned memory has already "
    "gone through retrieval-side updates such as reconsolidation and counter increments."
)

fig1 = plot_retrieval_radar(
    _signals_to_list(top.breakdown.raw_signals),
    SIGNAL_LABELS,
    title=f"Raw retrieval signals: '{top.memory.content[:34]}...'",
    color="#4C72B0",
)
fig2 = plot_retrieval_radar(
    _signals_to_list(top.breakdown.weighted_signals),
    SIGNAL_LABELS,
    title=f"Weighted retrieval signals: '{top.memory.content[:30]}...'",
    color="#DD5555",
)

print("\n=== Adaptive weights by mood ===\n")

_ts = datetime.now(UTC)
mood_scenarios = [
    (
        "Negative mood  (val=-0.8, ar=0.6)",
        MoodField(valence=-0.8, arousal=0.6, dominance=0.3, inertia=0.5, timestamp=_ts),
    ),
    (
        "High arousal   (val=+0.1, ar=0.9)",
        MoodField(valence=0.1, arousal=0.9, dominance=0.5, inertia=0.5, timestamp=_ts),
    ),
    (
        "Calm / neutral (val=+0.1, ar=0.2)",
        MoodField(valence=0.1, arousal=0.2, dominance=0.6, inertia=0.5, timestamp=_ts),
    ),
]

base_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
short_labels = ["semant", "mood", "affect", "moment", "recency", "resonan"]

print(f"{'Scenario':<35}" + "".join(f"  {label:>7}" for label in short_labels))
print("-" * 78)
for label, sf in mood_scenarios:
    weights = adaptive_weights(sf, base_weights)
    print(f"  {label:<33}" + "".join(f"  {value:>7.3f}" for value in weights))

print()
print("Negative mood  → emotional signals (mood, affect) boosted.")
print("High arousal   → momentum boosted.")
print("Calm/neutral   → semantic weight boosted.")

print("\n=== Adaptive weights heatmap (full mood space) ===")
fig3 = plot_adaptive_weights_heatmap(
    base_weights=base_weights,
    resolution=25,
    title="Retrieval weight landscape vs mood (valence x arousal)",
)

_ = (fig1, fig2, fig3)
plt.show()
print("\nDone.")
