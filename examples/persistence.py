"""Persistence — save state and memories across sessions.

Demonstrates SQLiteStore for durable storage, save_state/load_state for
affective state continuity, export_memories/import_memories for backup and
migration, prune() for maintenance, and the context-manager pattern.

Requires:
    pip install emotional-memory[sqlite]

Run with:
    python examples/persistence.py
"""

import json
import tempfile

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    RetrievalConfig,
    SQLiteStore,
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
# Session 1 — encode memories into a SQLite file, then save state and close
# ---------------------------------------------------------------------------

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
    db_path = db_file.name

print(f"=== Session 1: encoding into {db_path} ===\n")

config = EmotionalMemoryConfig(
    retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
)

with EmotionalMemory(store=SQLiteStore(db_path), embedder=HashEmbedder(), config=config) as em:
    em.set_affect(CoreAffect(valence=0.9, arousal=0.8))
    em.encode(
        "Landed a major client — six-figure contract signed.", metadata={"category": "sales"}
    )

    em.set_affect(CoreAffect(valence=-0.6, arousal=0.7))
    em.encode("Server outage lasted three hours on Black Friday.", metadata={"category": "ops"})

    em.set_affect(CoreAffect(valence=0.5, arousal=0.3))
    em.encode(
        "Slow afternoon catching up on technical reading.", metadata={"category": "personal"}
    )

    em.set_affect(CoreAffect(valence=0.8, arousal=0.6))
    em.encode("Team retrospective went really well, high energy.", metadata={"category": "work"})

    print(f"Encoded {len(em)} memories.")

    # Save affective state so the next session resumes the same mood trajectory
    state_snapshot = em.save_state()
    print(f"Mood at close: valence={state_snapshot['mood']['valence']:.3f}")

    # Export all memories as JSON-serialisable dicts (for backup / migration)
    exported = em.export_memories()
    print(f"Exported {len(exported)} memory dicts.")

# Context manager called close() — SQLite connection is now flushed and closed.

# ---------------------------------------------------------------------------
# Session 2 — reopen the same file, restore state, verify continuity
# ---------------------------------------------------------------------------

print(f"\n=== Session 2: reopening {db_path} ===\n")

with EmotionalMemory(store=SQLiteStore(db_path), embedder=HashEmbedder(), config=config) as em2:
    print(f"Memories in store: {len(em2)}")

    # Restore affective state — mood and momentum history resume
    em2.load_state(state_snapshot)
    sm = em2.get_state().mood
    print(f"Restored Mood: valence={sm.valence:.3f}  arousal={sm.arousal:.3f}")

    # Retrieval works identically to session 1
    results = em2.retrieve("client deal revenue", top_k=2)
    print("\nTop-2 results for 'client deal revenue':")
    for i, mem in enumerate(results, 1):
        ca = mem.tag.core_affect
        print(f"  {i}. {mem.content[:60]}")
        print(f"     valence={ca.valence:+.2f}  retrieval_count={mem.tag.retrieval_count}")

# ---------------------------------------------------------------------------
# Import/export round-trip — migrate memories to a fresh in-memory store
# ---------------------------------------------------------------------------

print("\n=== Import into a fresh in-memory store ===\n")

em_fresh = EmotionalMemory(store=InMemoryStore(), embedder=HashEmbedder(), config=config)
count = em_fresh.import_memories(exported)
print(f"Imported {count} memories.")
print(f"Store size: {len(em_fresh)}")

# import_memories is idempotent by default — duplicates are skipped
count2 = em_fresh.import_memories(exported)
print(f"Re-import (overwrite=False): {count2} written (0 = all skipped as duplicates).")

count3 = em_fresh.import_memories(exported, overwrite=True)
print(f"Re-import (overwrite=True):  {count3} written.")

# ---------------------------------------------------------------------------
# Prune — remove memories whose strength has decayed below threshold
# ---------------------------------------------------------------------------

print("\n=== Pruning weak memories ===\n")

# These memories are brand-new, so none will be pruned at the default threshold.
# Lower the threshold drastically to illustrate the API.
removed = em_fresh.prune(threshold=0.0)
print(f"prune(threshold=0.0) removed: {removed}")
print(f"Remaining: {len(em_fresh)}")

# Verify the state snapshot round-trips through JSON cleanly
snapshot_json = json.dumps(state_snapshot)
reloaded = json.loads(snapshot_json)
em_fresh.load_state(reloaded)
print("\nState round-tripped through JSON without errors.")

print("\nDone.")
