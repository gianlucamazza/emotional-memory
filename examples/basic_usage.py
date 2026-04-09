"""Basic usage example for emotional_memory.

Demonstrates the full encode → retrieve pipeline without any ML dependencies.
Uses a deterministic hash-based embedder so the script runs standalone.

Run with:
    python examples/basic_usage.py
"""

from emotional_memory import (
    AffectiveState,
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
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
# Setup
# ---------------------------------------------------------------------------

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05],  # stronger emotional bias
        ),
        resonance=ResonanceConfig(threshold=0.05),
        stimmung_alpha=0.2,
    ),
)

# ---------------------------------------------------------------------------
# Encode memories under different emotional states
# ---------------------------------------------------------------------------

print("=== Encoding memories ===\n")

em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
m1 = em.encode(
    "Shipped the feature after three weeks — the team is ecstatic.",
    metadata={"category": "work"},
)
ca1 = m1.tag.core_affect
print(f"[+] {m1.content[:50]}")
print(f"    affect: valence={ca1.valence:.2f}, arousal={ca1.arousal:.2f}")
print(f"    consolidation: {m1.tag.consolidation_strength:.2f}")

em.set_affect(CoreAffect(valence=-0.7, arousal=0.6))
m2 = em.encode(
    "Sprint retrospective turned into an argument about priorities.",
    metadata={"category": "work"},
)
ca2 = m2.tag.core_affect
print(f"\n[-] {m2.content[:50]}")
print(f"    affect: valence={ca2.valence:.2f}, arousal={ca2.arousal:.2f}")

em.set_affect(CoreAffect(valence=0.3, arousal=0.2))
m3 = em.encode(
    "Quiet afternoon reviewing documentation and catching up on reading.",
    metadata={"category": "personal"},
)
ca3 = m3.tag.core_affect
print(f"\n[~] {m3.content[:50]}")
print(f"    affect: valence={ca3.valence:.2f}, arousal={ca3.arousal:.2f}")

em.set_affect(CoreAffect(valence=0.9, arousal=0.8))
m4 = em.encode(
    "Breakthrough on the architecture problem — everything clicked.",
    metadata={"category": "work"},
)
ca4 = m4.tag.core_affect
print(f"\n[+] {m4.content[:50]}")
print(f"    affect: valence={ca4.valence:.2f}, arousal={ca4.arousal:.2f}")
print(f"    consolidation: {m4.tag.consolidation_strength:.2f}")

# ---------------------------------------------------------------------------
# Retrieve under positive mood — expect positive memories to surface
# ---------------------------------------------------------------------------

print("\n=== Retrieving under positive mood ===\n")

em.set_affect(CoreAffect(valence=0.8, arousal=0.6))
state: AffectiveState = em.get_state()
sm = state.stimmung
print(f"Current Stimmung: valence={sm.valence:.3f}, arousal={sm.arousal:.3f}\n")

results = em.retrieve("project work accomplishment", top_k=3)
for i, mem in enumerate(results, 1):
    ca = mem.tag.core_affect
    sign = "+" if ca.valence > 0 else "-"
    print(f"  {i}. [{sign}] {mem.content[:60]}")
    print(f"       valence={ca.valence:.2f}  retrieval_count={mem.tag.retrieval_count}")

# ---------------------------------------------------------------------------
# Retrieve under negative mood — expect negative memories to surface
# ---------------------------------------------------------------------------

print("\n=== Retrieving under negative mood ===\n")

for _ in range(5):
    em.set_affect(CoreAffect(valence=-0.9, arousal=0.5))

state = em.get_state()
sm = state.stimmung
print(f"Current Stimmung: valence={sm.valence:.3f}, arousal={sm.arousal:.3f}\n")

results = em.retrieve("team meeting discussion", top_k=3)
for i, mem in enumerate(results, 1):
    ca = mem.tag.core_affect
    sign = "+" if ca.valence > 0 else "-"
    recon = mem.tag.reconsolidation_count > 0
    print(f"  {i}. [{sign}] {mem.content[:60]}")
    print(f"       valence={ca.valence:.2f}  reconsolidated={recon}")

# ---------------------------------------------------------------------------
# Resonance links
# ---------------------------------------------------------------------------

print("\n=== Resonance links on m4 ===\n")
# Re-fetch from store to see any links built after encoding
all_mems = em._store.list_all()
m4_stored = next(m for m in all_mems if m.id == m4.id)
if m4_stored.tag.resonance_links:
    for link in m4_stored.tag.resonance_links:
        print(f"  {link.link_type}: strength={link.strength:.3f} → {link.target_id[:8]}...")
else:
    print("  (no resonance links above threshold — embeddings are hash-based and sparse)")

print("\nDone.")
