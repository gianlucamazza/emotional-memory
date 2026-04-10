"""Emotional journal — multi-session journaling with full lifecycle.

Capstone example combining:
  - KeywordAppraisalEngine for automatic emotional tagging
  - SQLiteStore for persistence across sessions
  - Stimmung evolution and time-based regression
  - Mood-congruent retrieval that shifts as mood changes
  - Session save → close → reopen → load → continue
  - Pruning to remove faded low-arousal memories

Requires (for persistence):
    pip install emotional-memory[sqlite]

Run with:
    python examples/emotional_journal.py
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta

from emotional_memory import (
    DecayConfig,
    EmotionalMemory,
    EmotionalMemoryConfig,
    KeywordAppraisalEngine,
    ResonanceConfig,
    RetrievalConfig,
    SQLiteStore,
    StimmungDecayConfig,
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
# Shared config
# ---------------------------------------------------------------------------

embedder = HashEmbedder()
appraisal = KeywordAppraisalEngine()

config = EmotionalMemoryConfig(
    resonance=ResonanceConfig(threshold=0.1, max_links=5),
    retrieval=RetrievalConfig(
        base_weights=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
        ape_threshold=0.25,
        reconsolidation_window_seconds=3600.0,  # 1-hour lability window
        reconsolidation_learning_rate=0.25,
    ),
    stimmung_alpha=0.25,
    stimmung_decay=StimmungDecayConfig(base_half_life_seconds=1800.0),  # 30-min half-life
    decay=DecayConfig(base_decay=1.5),  # aggressive decay for pruning demo
)


# ---------------------------------------------------------------------------
# Mood dashboard helper
# ---------------------------------------------------------------------------


def print_mood(em: EmotionalMemory, label: str) -> None:
    s = em.get_state()
    sm = s.stimmung
    mom = s.momentum
    print(
        f"  [{label}]  "
        f"affect={s.core_affect.valence:+.2f}  "
        f"stimmung={sm.valence:+.3f}  "
        f"momentum={mom.magnitude():.3f}"
    )


# ---------------------------------------------------------------------------
# Session 1 — Morning journal
# ---------------------------------------------------------------------------

print("=== Session 1: Morning journal ===\n")

# Use a temp file so the example is always self-contained and repeatable
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as _f:
    db_path = _f.name

morning_entries = [
    "Woke up feeling rested — slept well for the first time this week.",
    "Morning coffee and a calm review of yesterday's progress.",
    "Unexpected critical bug reported — production is degraded.",
    "Spent two hours debugging — the root cause was a config typo.",
    "Quick team sync — morale is low after the incident.",
]

saved_state: dict = {}
exported_memories: list[dict] = []

with EmotionalMemory(
    store=SQLiteStore(db_path),
    embedder=embedder,
    appraisal_engine=appraisal,
    config=config,
) as em:
    print("Encoding 5 morning journal entries (KeywordAppraisalEngine):\n")

    for i, text in enumerate(morning_entries, 1):
        mem = em.encode(text)
        valence = mem.tag.core_affect.valence
        sign = "[+]" if valence > 0 else "[-]"
        print(f"  {i}. {sign} {text[:60]}")
        print_mood(em, f"after entry {i}")

        # Mid-session retrieval after the incident entry
        if i == 4:
            print()
            print("  >>> Retrieve 'production incident' under current mood:")
            results = em.retrieve("production incident", top_k=2)
            for r in results:
                ca = r.tag.core_affect
                label = "[+]" if ca.valence > 0 else "[-]"
                print(f"      {label} {r.content[:55]}  (valence={ca.valence:+.2f})")
            print()

    count_s1 = len(em.list_all())
    print(f"\n  Memories stored after session 1: {count_s1}")

    # Persist state for session 2
    saved_state = em.save_state()
    exported_memories = em.export_memories()

print()
print(f"Session 1 closed. State saved. {len(exported_memories)} memories exported.")

# ---------------------------------------------------------------------------
# Session 2 — Afternoon journal (1 hour later)
# ---------------------------------------------------------------------------

print("\n=== Session 2: Afternoon journal (simulating 1h later) ===\n")

afternoon_entries = [
    "Fixed the root cause and deployed a hotfix. All systems green.",
    "Positive post-mortem — team learned from the incident.",
    "Ended the day with a relaxing walk in the park.",
]

with EmotionalMemory(
    store=SQLiteStore(db_path),
    embedder=embedder,
    appraisal_engine=appraisal,
    config=config,
) as em2:
    # Restore affective state from session 1
    em2.load_state(saved_state)

    # Inspect Stimmung regression 1 hour after session 1
    future = datetime.now(UTC) + timedelta(hours=1)
    regressed = em2.get_current_stimmung(now=future)
    current = em2.get_state().stimmung

    print("Stimmung continuity check:")
    print(f"  At session close: valence={current.valence:+.3f}  arousal={current.arousal:.3f}")
    print(
        f"  Regressed +1h:    valence={regressed.valence:+.3f}  "
        f"arousal={regressed.arousal:.3f}  (toward baseline)"
    )
    print()

    memories_at_start = len(em2.list_all())
    print(f"  Memories loaded from SQLite: {memories_at_start}")
    print()

    print("Encoding 3 afternoon entries:\n")

    reconsolidated_content: str = ""

    for i, text in enumerate(afternoon_entries, 1):
        mem = em2.encode(text)
        valence = mem.tag.core_affect.valence
        sign = "[+]" if valence > 0 else "[-]"
        print(f"  {6 + i - 1}. {sign} {text[:60]}")
        print_mood(em2, f"after entry {6 + i - 1}")

    # Retrieve same query — ranking should shift toward positive resolution memories
    print()
    print("  >>> Retrieve 'production incident' under improved afternoon mood:")
    results2 = em2.retrieve("production incident", top_k=3)
    for r in results2:
        ca = r.tag.core_affect
        label = "[+]" if ca.valence > 0 else "[-]"
        recon = (
            f"  reconsolidated x{r.tag.reconsolidation_count}"
            if r.tag.reconsolidation_count > 0
            else ""
        )
        print(f"      {label} {r.content[:55]}  (valence={ca.valence:+.2f}){recon}")
        if r.tag.reconsolidation_count > 0 and not reconsolidated_content:
            reconsolidated_content = r.content[:40]

    if reconsolidated_content:
        print(f"\n  Note: '{reconsolidated_content}' was reconsolidated")
        print("        under a different emotional state (Nader & Schiller 2000).")

    count_s2 = len(em2.list_all())

    # ---------------------------------------------------------------------------
    # Maintenance — pruning faded memories
    # ---------------------------------------------------------------------------

    print("\n=== Maintenance: pruning faded memories ===\n")

    print(f"  Before pruning: {count_s2} memories")
    removed = em2.prune(threshold=0.30)
    count_after = len(em2.list_all())
    print(f"  Pruned {removed} low-strength memories (threshold=0.30, base_decay=1.5)")
    print(f"  After pruning:  {count_after} memories")

    # Final export as backup
    backup = em2.export_memories()
    backup_json = json.dumps(backup, indent=2)
    print(f"\n  Backup export: {len(backup)} memories ({len(backup_json)} bytes JSON)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Summary ===\n")
print(f"  {'Session 1 entries encoded:':<38} {len(morning_entries)}")
print(f"  {'Session 2 entries encoded:':<38} {len(afternoon_entries)}")
print(f"  {'Memories after session 1:':<38} {count_s1}")
print(f"  {'Memories after session 2:':<38} {count_s2}")
print(f"  {'Memories pruned:':<38} {removed}")
print(f"  {'Memories in final backup:':<38} {len(backup)}")
print()
print("  Lifecycle completed:")
print("    encode → save_state → close → reopen → load_state")
print("    → get_current_stimmung (time regression) → encode")
print("    → mood-congruent retrieval → prune → export_memories")

print("\nDone.")
