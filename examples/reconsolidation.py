"""Reconsolidation — how memories change when retrieved under different emotions.

Demonstrates the two-retrieval lability pattern (Nader & Schiller 2000):
  1. First retrieval opens a lability window (sets last_retrieved)
  2. Second retrieval within the window under a different affect:
     if affective prediction error > threshold → core_affect is updated

Also shows the two cases where reconsolidation does NOT trigger:
  - Lability window expired (reconsolidation_window_seconds=0.001)
  - Affect at retrieval is similar to encoding affect (APE below threshold)

Run with:
    python examples/reconsolidation.py
"""

import time

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    RetrievalConfig,
)
from emotional_memory.retrieval import affective_prediction_error

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
# Case 1 — reconsolidation TRIGGERS
# ---------------------------------------------------------------------------

print("=== Case 1: reconsolidation triggers ===\n")

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,  # lower than default 0.3 → easier to trigger
            reconsolidation_learning_rate=0.3,  # higher than default 0.2 → bigger shift
            reconsolidation_window_seconds=300.0,  # 5-minute lability window
        ),
        stimmung_alpha=0.3,
    ),
)

# Encode a neutral memory
em.set_affect(CoreAffect(valence=0.0, arousal=0.3))
em.encode("Routine standup meeting — nothing unusual to report.")
em.encode("Quick look at the metrics dashboard — numbers stable.")

print("Encoded 2 memories under neutral affect (valence=0.0).\n")

# --- First retrieval: opens the lability window ---
results = em.retrieve("standup meeting", top_k=1)
mem_r1 = results[0]

valence_before = mem_r1.tag.core_affect.valence
print("After first retrieval:")
print(f"  core_affect.valence:    {valence_before:+.3f}")
print(f"  retrieval_count:        {mem_r1.tag.retrieval_count}")
print(f"  reconsolidation_count:  {mem_r1.tag.reconsolidation_count}")
print(f"  last_retrieved set:     {mem_r1.tag.last_retrieved is not None}")

# --- Shift affect strongly toward positive ---
for _ in range(6):
    em.set_affect(CoreAffect(valence=0.9, arousal=0.8))

current_affect = em.get_state().core_affect
ape = affective_prediction_error(mem_r1.tag.core_affect, current_affect)
print("\nAfter 6x set_affect(valence=+0.9, arousal=0.8):")
print(f"  current Stimmung valence: {em.get_state().stimmung.valence:+.3f}")
print(f"  APE vs encoded affect:    {ape:.3f}  (threshold: 0.2)")

# --- Second retrieval within lability window → reconsolidation ---
results2 = em.retrieve("standup meeting", top_k=1)
mem_r2 = results2[0]

valence_after = mem_r2.tag.core_affect.valence
shift = valence_after - valence_before
print("\nAfter second retrieval (reconsolidation check):")
print(f"  core_affect.valence:    {valence_after:+.3f}  (was {valence_before:+.3f})")
print(f"  valence shift:          {shift:+.3f}")
print(f"  reconsolidation_count:  {mem_r2.tag.reconsolidation_count}")
triggered = mem_r2.tag.reconsolidation_count > 0
print(f"  reconsolidation fired:  {triggered}")
if triggered:
    print("\n  The neutral memory has been tinted positive by retrieval")
    print("  under a strongly positive emotional state — Nader & Schiller (2000).")

# ---------------------------------------------------------------------------
# Case 2 — reconsolidation does NOT trigger: window expired
# ---------------------------------------------------------------------------

print("\n=== Case 2: no reconsolidation — lability window expired ===\n")

em2 = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,
            reconsolidation_learning_rate=0.3,
            reconsolidation_window_seconds=0.01,  # 10 ms window
        ),
        stimmung_alpha=0.3,
    ),
)

em2.set_affect(CoreAffect(valence=0.0, arousal=0.3))
em2.encode("Team sync — all status updates collected.")

# First retrieval: sets last_retrieved, opening the 10 ms lability window
r1 = em2.retrieve("team sync", top_k=1)
v_before2 = r1[0].tag.core_affect.valence

# Wait for the lability window to expire
time.sleep(0.05)  # 50 ms > 10 ms window

# Shift affect strongly
for _ in range(6):
    em2.set_affect(CoreAffect(valence=0.9, arousal=0.8))

# Second retrieval: window has expired → no reconsolidation despite high APE
r2 = em2.retrieve("team sync", top_k=1)
v_after2 = r2[0].tag.core_affect.valence

print(f"valence before: {v_before2:+.3f}")
print(f"valence after:  {v_after2:+.3f}  (unchanged — window expired)")
print(f"reconsolidation_count: {r2[0].tag.reconsolidation_count}  (still 0)")

# ---------------------------------------------------------------------------
# Case 3 — reconsolidation does NOT trigger: APE below threshold
# ---------------------------------------------------------------------------

print("\n=== Case 3: no reconsolidation — APE below threshold ===\n")

em3 = EmotionalMemory(
    store=InMemoryStore(),
    embedder=HashEmbedder(),
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            ape_threshold=0.2,
            reconsolidation_learning_rate=0.3,
            reconsolidation_window_seconds=300.0,
        ),
        stimmung_alpha=0.3,
    ),
)

em3.set_affect(CoreAffect(valence=0.5, arousal=0.5))
em3.encode("Reviewed the architecture diagram — looks solid.")

# First retrieval under similar affect
r1_3 = em3.retrieve("architecture", top_k=1)
v_before3 = r1_3[0].tag.core_affect.valence

# Stay near the same affect (small shift only)
em3.set_affect(CoreAffect(valence=0.55, arousal=0.55))

current3 = em3.get_state().core_affect
ape3 = affective_prediction_error(r1_3[0].tag.core_affect, current3)

r2_3 = em3.retrieve("architecture", top_k=1)
v_after3 = r2_3[0].tag.core_affect.valence

print(f"APE at second retrieval: {ape3:.3f}  (threshold: 0.2)")
print(f"valence before: {v_before3:+.3f}")
print(f"valence after:  {v_after3:+.3f}  (unchanged — APE too small)")
print(f"reconsolidation_count: {r2_3[0].tag.reconsolidation_count}  (still 0)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Summary ===\n")
print(f"{'Case':<40}  {'triggered':>9}  {'valence shift':>13}")
print("-" * 66)
print(f"  {'1. Strong affect shift, within window':<38}  {'yes':>9}  {shift:>+13.3f}")
print(
    f"  {'2. Strong shift, window expired (50 ms)':<38}  {'no':>9}  {v_after2 - v_before2:>+13.3f}"
)
print(f"  {'3. Same affect, APE below threshold':<38}  {'no':>9}  {v_after3 - v_before3:>+13.3f}")
print()
print("Reconsolidation requires: APE > threshold AND last_retrieved within window.")

print("\nDone.")
