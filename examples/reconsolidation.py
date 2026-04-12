"""Reconsolidation — how memories change when retrieved under different emotions.

Demonstrates the APE-gated reconsolidation model (Nader & Schiller 2000):

  The lability window is opened ONLY by retrievals with high affective prediction
  error (APE > threshold).  This separates two distinct events:

    1. Window-opening  — high-APE retrieval sets window_opened_at on the tag
    2. Reconsolidation — any retrieval within the open window updates core_affect

  Cases shown:
    Case 1: high APE retrieval → opens window AND reconsolidates
    Case 2: high APE opens window → low-APE retrieval within window → still reconsolidates
    Case 3: low APE retrieval → window stays closed → no reconsolidation

Run with:
    python examples/reconsolidation.py
"""

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    RetrievalConfig,
    compute_ape,
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


# ---------------------------------------------------------------------------
# Case 1 — high APE retrieval opens window AND reconsolidates
# ---------------------------------------------------------------------------

print("=== Case 1: high-APE retrieval opens window + reconsolidates ===\n")

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,
            reconsolidation_learning_rate=0.3,
            reconsolidation_window_seconds=300.0,
        ),
        mood_alpha=0.3,
    ),
)

# Encode a neutral memory
em.set_affect(CoreAffect(valence=0.0, arousal=0.3))
em.encode("Routine standup meeting — nothing unusual to report.")

print("Encoded under neutral affect (valence=0.0).\n")

# Shift affect strongly positive before retrieval
for _ in range(6):
    em.set_affect(CoreAffect(valence=0.9, arousal=0.8))

# Retrieval under positive affect → high APE → opens window_opened_at + reconsolidates
results = em.retrieve("standup meeting", top_k=1)
mem_r1 = results[0]

ape = compute_ape(mem_r1.tag, em.get_state().core_affect)
print(f"APE at retrieval: {ape:.3f}  (threshold: 0.2)")
print(f"window_opened_at set: {mem_r1.tag.window_opened_at is not None}")
print(f"valence after:        {mem_r1.tag.core_affect.valence:+.3f}  (was 0.000)")
print(f"reconsolidation_count: {mem_r1.tag.reconsolidation_count}")

# ---------------------------------------------------------------------------
# Case 2 — within open window: low-APE retrieval still reconsolidates
# ---------------------------------------------------------------------------

print("\n=== Case 2: low-APE retrieval within open window still reconsolidates ===\n")

em2 = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,
            reconsolidation_learning_rate=0.3,
            reconsolidation_window_seconds=300.0,
        ),
        mood_alpha=0.3,
    ),
)

em2.set_affect(CoreAffect(valence=0.0, arousal=0.3))
em2.encode("Team sync — all status updates collected.")

# First retrieval: strongly positive affect → high APE → opens window
for _ in range(6):
    em2.set_affect(CoreAffect(valence=0.9, arousal=0.8))

r1 = em2.retrieve("team sync", top_k=1)
v_after_r1 = r1[0].tag.core_affect.valence
window_open = r1[0].tag.window_opened_at is not None
print(f"After high-APE retrieval: valence={v_after_r1:+.3f}, window open={window_open}")

# Return to similar affect — APE will be LOW relative to the now-shifted prediction
em2.set_affect(CoreAffect(valence=0.85, arousal=0.75))

r2 = em2.retrieve("team sync", top_k=1)
v_after_r2 = r2[0].tag.core_affect.valence
ape2 = compute_ape(r1[0].tag, em2.get_state().core_affect)
print(f"APE at second retrieval: {ape2:.3f}  (below threshold)")
print(f"valence after second retrieval: {v_after_r2:+.3f}")
print(f"reconsolidation_count: {r2[0].tag.reconsolidation_count}  (window was open → still fires)")

# ---------------------------------------------------------------------------
# Case 3 — low APE: window stays closed, no reconsolidation
# ---------------------------------------------------------------------------

print("\n=== Case 3: low APE — window never opens, no reconsolidation ===\n")

em3 = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,
            reconsolidation_learning_rate=0.3,
            reconsolidation_window_seconds=300.0,
        ),
        mood_alpha=0.3,
    ),
)

em3.set_affect(CoreAffect(valence=0.5, arousal=0.5))
em3.encode("Reviewed the architecture diagram — looks solid.")
v_encoded = 0.5

# Retrieve under nearly identical affect — APE will be below threshold
em3.set_affect(CoreAffect(valence=0.55, arousal=0.52))
r1_3 = em3.retrieve("architecture", top_k=1)
ape3 = compute_ape(r1_3[0].tag, em3.get_state().core_affect)

print(f"APE at retrieval: {ape3:.3f}  (below threshold 0.2)")
print(f"window_opened_at set: {r1_3[0].tag.window_opened_at is not None}  (stays closed)")
print(f"valence unchanged:    {r1_3[0].tag.core_affect.valence:+.3f}  (was {v_encoded:+.3f})")
print(f"reconsolidation_count: {r1_3[0].tag.reconsolidation_count}  (still 0)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Summary ===\n")
print("APE-gated reconsolidation (Nader & Schiller 2000):")
print("  - High APE → window_opened_at = now, core_affect updated")
print("  - Within open window → core_affect updated even with low APE")
print("  - Low APE + closed window → no change")
print()
triggered_1 = mem_r1.tag.reconsolidation_count > 0
triggered_2 = r2[0].tag.reconsolidation_count > 1
triggered_3 = r1_3[0].tag.reconsolidation_count > 0
print(f"{'Case':<48}  {'fired':>5}")
print("-" * 56)
print(
    f"  {'1. High APE → opens window + reconsolidates':<46}  {'yes' if triggered_1 else 'no':>5}"
)
label2 = "yes" if triggered_2 else "no"
print(f"  {'2. Low APE within open window → reconsolidates':<46}  {label2:>5}")
label3 = "no" if not triggered_3 else "yes"
print(f"  {'3. Low APE, window closed → no reconsolidation':<46}  {label3:>5}")

print("\nDone.")
