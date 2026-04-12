"""Resonance network — how memories form an associative graph.

Encodes 8 thematically clustered memories and visualises the directed
resonance graph that emerges. Each link carries a type (semantic, emotional,
temporal, causal, contrastive) and a strength score.

Requires (for the network plot):
    pip install emotional-memory[viz]

Run with:
    python examples/resonance_network.py
"""

from collections import Counter

import matplotlib.pyplot as plt

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
)
from emotional_memory.visualization import plot_resonance_network

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
# Setup — low threshold to get rich resonance links
# ---------------------------------------------------------------------------

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        resonance=ResonanceConfig(
            threshold=0.05,  # low → more links (default 0.3)
            max_links=8,  # more links per memory (default 5)
        ),
        retrieval=RetrievalConfig(
            base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05],
        ),
        mood_alpha=0.25,
    ),
)

# ---------------------------------------------------------------------------
# Encode 8 memories in three thematic clusters
# ---------------------------------------------------------------------------

print("=== Encoding memories ===\n")

memories_meta = [
    # Cluster 1: Work — positive
    (CoreAffect(valence=0.9, arousal=0.8), "Shipped the redesign — users love it.", "work"),
    (
        CoreAffect(valence=0.8, arousal=0.7),
        "Team celebration after the successful launch.",
        "work",
    ),
    (CoreAffect(valence=0.7, arousal=0.6), "Stellar Q3 performance review with a bonus.", "work"),
    # Cluster 2: Work — negative
    (
        CoreAffect(valence=-0.8, arousal=0.8),
        "Production outage — five hours of downtime on Friday.",
        "work",
    ),
    (CoreAffect(valence=-0.6, arousal=0.6), "Sprint retro turned into a heated argument.", "work"),
    # Cluster 3: Personal
    (
        CoreAffect(valence=0.5, arousal=0.2),
        "Peaceful morning walk through the park before work.",
        "personal",
    ),
    (
        CoreAffect(valence=0.6, arousal=0.3),
        "Finished reading an inspiring book on systems thinking.",
        "personal",
    ),
    (
        CoreAffect(valence=0.7, arousal=0.4),
        "Caught up with an old friend over dinner.",
        "personal",
    ),
]

encoded = []
for affect, text, category in memories_meta:
    em.set_affect(affect)
    mem = em.encode(text, metadata={"category": category})
    encoded.append(mem)
    sign = "[+]" if affect.valence > 0 else "[-]"
    print(f"{sign} [{category:8s}] {text[:55]}")
    print(
        f"           valence={affect.valence:+.1f}  arousal={affect.arousal:.1f}  "
        f"strength={mem.tag.consolidation_strength:.2f}"
    )

# ---------------------------------------------------------------------------
# Collect all resonance links from the store
# ---------------------------------------------------------------------------

print("\n=== Resonance link graph ===\n")

all_mems = em.list_all()
all_links = []
node_labels: dict[str, str] = {}

for mem in all_mems:
    node_labels[mem.id] = mem.content[:28]
    all_links.extend(mem.tag.resonance_links)

print(f"Memories encoded:  {len(all_mems)}")
print(f"Total links found: {len(all_links)}")
print("(First memory encoded always has 0 links — no prior candidates)\n")

# ---------------------------------------------------------------------------
# Link-type distribution
# ---------------------------------------------------------------------------

type_counts: Counter[str] = Counter(lnk.link_type for lnk in all_links)
type_strengths: dict[str, list[float]] = {}
for lnk in all_links:
    type_strengths.setdefault(lnk.link_type, []).append(lnk.strength)

print(f"{'Link type':<14}  {'count':>5}  {'avg strength':>12}")
print("-" * 36)
for lt in ["temporal", "emotional", "semantic", "causal", "contrastive"]:
    if lt not in type_counts:
        continue
    strengths = type_strengths[lt]
    avg = sum(strengths) / len(strengths)
    print(f"  {lt:<12}  {type_counts[lt]:>5}  {avg:>12.3f}")

# ---------------------------------------------------------------------------
# Show the strongest link per memory
# ---------------------------------------------------------------------------

print("\n=== Strongest link per memory ===\n")

for mem in all_mems:
    links = mem.tag.resonance_links
    if not links:
        print(f"  '{mem.content[:40]}'  — no outgoing links")
        continue
    top = max(links, key=lambda lnk: lnk.strength)
    target = next((m for m in all_mems if m.id == top.target_id), None)
    target_snippet = target.content[:35] if target else top.target_id[:8]
    print(f"  '{mem.content[:35]}' →")
    print(f"    {top.link_type:12s}  strength={top.strength:.3f}  → '{target_snippet}'")

# ---------------------------------------------------------------------------
# Visualise the resonance network
# ---------------------------------------------------------------------------

print("\n=== Plotting resonance network (close window to exit) ===")

fig = plot_resonance_network(
    all_links,
    node_labels=node_labels,
    title="Memory Resonance Network (8 memories, threshold=0.05)",
)
plt.show()
