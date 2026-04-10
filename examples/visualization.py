"""Visualization — plotting the emotional landscape of memories.

Demonstrates 5 of the 8 matplotlib-based plot functions:
  - plot_circumplex        → memories on the Russell valence-arousal plane
  - plot_decay_curves      → ACT-R power-law decay families
  - plot_yerkes_dodson     → consolidation sweet spot (inverted-U)
  - plot_stimmung_evolution→ mood drift over a sequence of events
  - plot_appraisal_radar   → Scherer CPM spider chart for one memory

Requires:
    pip install emotional-memory[viz]

Run with:
    python examples/visualization.py
"""

import matplotlib.pyplot as plt

from emotional_memory import (
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    KeywordAppraisalEngine,
    RetrievalConfig,
)
from emotional_memory.visualization import (
    plot_appraisal_radar,
    plot_circumplex,
    plot_decay_curves,
    plot_stimmung_evolution,
    plot_yerkes_dodson,
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
# Build up a set of memories with varied emotional profiles
# ---------------------------------------------------------------------------

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    appraisal_engine=KeywordAppraisalEngine(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
        stimmung_alpha=0.25,
    ),
)

events = [
    "Received a standing ovation after the product demo.",
    "Critical security vulnerability found in the auth layer.",
    "Stumbled upon an elegant solution to the caching problem.",
    "Missed the quarterly OKR due to resource constraints.",
    "Team lunch to celebrate shipping the redesign.",
    "Woke up at 3 AM to fix a flaky CI pipeline again.",
    "Peaceful morning reviewing architecture diagrams.",
    "Investor call went sideways — many hard questions.",
]

stimmung_history: list[tuple[float, float, float, float]] = []
t = 0.0

for text in events:
    mem = em.encode(text)
    sm = em.get_state().stimmung
    stimmung_history.append((t, sm.valence, sm.arousal, sm.dominance))
    t += 60.0  # 60 seconds between each event

all_mems = em.list_all()
print(f"Encoded {len(all_mems)} memories.")

# ---------------------------------------------------------------------------
# Plot 1: Circumplex — memories on the Russell valence-arousal plane
# ---------------------------------------------------------------------------

circumplex_data = [
    (m.tag.core_affect.valence, m.tag.core_affect.arousal, m.tag.consolidation_strength)
    for m in all_mems
]
fig1 = plot_circumplex(circumplex_data, title="Memory Circumplex (Russell 1980)")
print("Plot 1: circumplex ready.")

# ---------------------------------------------------------------------------
# Plot 2: Decay curves — how arousal and retrieval count affect strength
# ---------------------------------------------------------------------------

fig2 = plot_decay_curves(
    base_decay=0.5,
    arousal_modulation=0.5,
    retrieval_boost=0.1,
    floor_arousal_threshold=0.7,
    floor_value=0.1,
    arousal_values=(0.0, 0.5, 1.0),
    retrieval_counts=(0, 5),
    title="ACT-R Power-Law Decay by Arousal & Retrieval Count",
)
print("Plot 2: decay curves ready.")

# ---------------------------------------------------------------------------
# Plot 3: Yerkes-Dodson — consolidation sweet spot
# ---------------------------------------------------------------------------

fig3 = plot_yerkes_dodson(
    stimmung_arousal=0.3,
    title="Yerkes-Dodson: Consolidation vs. Encoding Arousal",
)
print("Plot 3: Yerkes-Dodson ready.")

# ---------------------------------------------------------------------------
# Plot 4: Stimmung evolution — mood drift over the event sequence
# ---------------------------------------------------------------------------

fig4 = plot_stimmung_evolution(
    stimmung_history,
    title="Stimmung Evolution Across Events",
)
print("Plot 4: stimmung evolution ready.")

# ---------------------------------------------------------------------------
# Plot 5: Appraisal radar — pick the memory with the most vivid appraisal
# ---------------------------------------------------------------------------

appraisal_mems = [m for m in all_mems if m.tag.appraisal is not None]
if appraisal_mems:
    # Use the memory with highest self_relevance for a clear radar shape
    target = max(appraisal_mems, key=lambda m: m.tag.appraisal.self_relevance)  # type: ignore[union-attr]
    fig5 = plot_appraisal_radar(
        target.tag.appraisal,  # type: ignore[arg-type]
        title=f"Appraisal Radar: '{target.content[:40]}...'",
    )
    print(f"Plot 5: appraisal radar for '{target.content[:40]}...'")
else:
    print("Plot 5: no memories with appraisal vectors (keyword engine produced none).")

# ---------------------------------------------------------------------------
# Show all figures
# ---------------------------------------------------------------------------

print("\nShowing all plots — close the windows to exit.")
plt.show()
