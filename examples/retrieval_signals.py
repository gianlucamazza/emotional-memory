"""Retrieval signals — decomposing the 6-signal composite score.

Imports the internal signal helpers from emotional_memory.retrieval to compute
each signal individually, showing which dimensions drive retrieval for each
memory. Also demonstrates adaptive_weights() — how Stimmung modulates signal
importance — and visualises both with radar and heatmap plots.

Note: the private helpers (_cosine, _stimmung_congruence, etc.) are intentionally
accessed here for introspection. They are not part of the stable public API.

Requires (for plots):
    pip install emotional-memory[viz]

Run with:
    python examples/retrieval_signals.py
"""

from datetime import UTC, datetime

import matplotlib.pyplot as plt

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
    StimmungField,
)
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.retrieval import (
    _affect_proximity,
    _cosine,
    _momentum_alignment,
    _resonance_boost,
    _stimmung_congruence,
    adaptive_weights,
)
from emotional_memory.visualization import (
    SIGNAL_LABELS,
    plot_adaptive_weights_heatmap,
    plot_retrieval_radar,
)

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
# Setup and encode 4 memories with distinct emotional profiles
# ---------------------------------------------------------------------------

embedder = HashEmbedder()

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=embedder,
    config=EmotionalMemoryConfig(
        resonance=ResonanceConfig(threshold=0.05, max_links=5),
        retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
        stimmung_alpha=0.3,
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

# ---------------------------------------------------------------------------
# Prepare query: set a positive retrieval state
# ---------------------------------------------------------------------------

em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
state = em.get_state()
query_text = "technical achievement model training"
query_emb = embedder.embed(query_text)
now = datetime.now(UTC)
decay_cfg = DecayConfig()

ca = state.core_affect
sm = state.stimmung
print(f"Query: '{query_text}'")
print(f"Current affect:   valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}")
print(f"Current Stimmung: valence={sm.valence:+.3f}  arousal={sm.arousal:.3f}")

# ---------------------------------------------------------------------------
# Compute 6 signals per memory
# ---------------------------------------------------------------------------

print("\n=== Per-memory signal breakdown ===\n")

SIGNAL_NAMES = [
    "s1:semantic",
    "s2:stimmung",
    "s3:affect",
    "s4:momentum",
    "s5:recency",
    "s6:resonance",
]

# Collect all active IDs from a prior retrieve for resonance boost (Pass 2)
prior_results = em.retrieve(query_text, top_k=4)
active_ids = [m.id for m in prior_results]

all_mems = em.list_all()
memory_signals: list[tuple[str, list[float]]] = []

header = f"{'Memory':<48}" + "".join(f"  {n[3:]:>8}" for n in SIGNAL_NAMES)
print(header)
print("-" * len(header))

for mem in all_mems:
    emb = mem.embedding or []
    s1 = _cosine(query_emb, emb) if emb else 0.0
    s2 = _stimmung_congruence(state.stimmung, mem.tag.stimmung_snapshot)
    s3 = _affect_proximity(state.core_affect, mem.tag.core_affect)
    s4 = _momentum_alignment(state.momentum, mem.tag.momentum)
    s5 = compute_effective_strength(mem.tag, now, decay_cfg)
    s6 = _resonance_boost(active_ids, mem.tag.resonance_links)
    signals = [s1, s2, s3, s4, s5, s6]
    memory_signals.append((mem.content, signals))

    dominant_idx = signals.index(max(signals))
    dominant = SIGNAL_NAMES[dominant_idx][3:]
    row = f"{mem.content[:46]:<48}" + "".join(f"  {s:>8.3f}" for s in signals)
    print(f"{row}  ← {dominant}")

# ---------------------------------------------------------------------------
# Radar chart: top memory by composite score
# ---------------------------------------------------------------------------

base_w = [0.20, 0.30, 0.25, 0.10, 0.10, 0.05]
top_content, top_signals = max(
    memory_signals, key=lambda t: sum(s * w for s, w in zip(t[1], base_w, strict=True))
)

print(f"\n=== Radar chart: '{top_content[:55]}' ===")
fig1 = plot_retrieval_radar(
    top_signals,
    SIGNAL_LABELS,
    title=f"Retrieval Signals: '{top_content[:40]}...'",
    color="#4C72B0",
)

# ---------------------------------------------------------------------------
# Adaptive weights under 3 Stimmung scenarios
# ---------------------------------------------------------------------------

print("\n=== Adaptive weights by Stimmung ===\n")

_ts = datetime.now(UTC)
stimmung_scenarios = [
    (
        "Negative mood  (val=-0.8, ar=0.6)",
        StimmungField(valence=-0.8, arousal=0.6, dominance=0.3, inertia=0.5, timestamp=_ts),
    ),
    (
        "High arousal   (val=+0.1, ar=0.9)",
        StimmungField(valence=0.1, arousal=0.9, dominance=0.5, inertia=0.5, timestamp=_ts),
    ),
    (
        "Calm / neutral (val=+0.1, ar=0.2)",
        StimmungField(valence=0.1, arousal=0.2, dominance=0.6, inertia=0.5, timestamp=_ts),
    ),
]

base_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
short_labels = ["semant", "stimm", "affect", "moment", "recency", "resonan"]

print(f"{'Scenario':<35}" + "".join(f"  {lbl:>7}" for lbl in short_labels))
print("-" * 78)
for label, sf in stimmung_scenarios:
    w = adaptive_weights(sf, base_weights)
    print(f"  {label:<33}" + "".join(f"  {v:>7.3f}" for v in w))

print()
print("Negative mood  → emotional signals (s2, s3) boosted, semantic (s1) reduced.")
print("High arousal   → momentum (s4) boosted.")
print("Calm/neutral   → semantic (s1) boosted, emotional signals reduced.")

# ---------------------------------------------------------------------------
# Adaptive weights heatmap — full Stimmung space
# ---------------------------------------------------------------------------

print("\n=== Adaptive weights heatmap (full Stimmung space) ===")
fig2 = plot_adaptive_weights_heatmap(
    base_weights=base_weights,
    resolution=25,
    title="Retrieval Weight Landscape vs Stimmung (valence x arousal)",
)

plt.show()
print("\nDone.")
