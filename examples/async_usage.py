"""Async usage — EmotionalMemory in async/await contexts.

Demonstrates two paths to async operation:
  1. as_async()     — wrap an existing sync engine (fastest to set up)
  2. Manual construction with SyncToAsyncStore / SyncToAsyncEmbedder

Also shows encode_batch(), prune(), count(), and the async context manager.

Run with:
    python examples/async_usage.py
"""

import asyncio

from emotional_memory import (
    AsyncEmotionalMemory,
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    RetrievalConfig,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    as_async,
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


config = EmotionalMemoryConfig(
    retrieval=RetrievalConfig(base_weights=[0.20, 0.30, 0.25, 0.10, 0.10, 0.05]),
)


async def main() -> None:
    # -----------------------------------------------------------------------
    # Path 1: as_async() — wrap a sync engine in one call
    # -----------------------------------------------------------------------

    print("=== Path 1: as_async() wrapper ===\n")

    sync_em = EmotionalMemory(store=InMemoryStore(), embedder=HashEmbedder(), config=config)
    sync_em.set_affect(CoreAffect(valence=0.8, arousal=0.7))

    # as_async() shares the live _state with the sync engine — do not use both
    # concurrently from separate threads.
    aem = as_async(sync_em)

    mem = await aem.encode("Shipped the new feature ahead of schedule.")
    ca = mem.tag.core_affect
    print(f"[+] {mem.content}")
    strength = mem.tag.consolidation_strength
    print(f"    valence={ca.valence:+.2f}  arousal={ca.arousal:.2f}  strength={strength:.2f}\n")

    results = await aem.retrieve("feature release", top_k=2)
    print(f"Retrieved {len(results)} memory/memories via as_async().\n")

    # -----------------------------------------------------------------------
    # Path 2: Manual construction with SyncToAsync* adapters
    # -----------------------------------------------------------------------

    print("=== Path 2: manual SyncToAsync* construction ===\n")

    async_store = SyncToAsyncStore(InMemoryStore())
    async_embedder = SyncToAsyncEmbedder(HashEmbedder())

    async with AsyncEmotionalMemory(
        store=async_store,
        embedder=async_embedder,
        config=config,
    ) as aem2:
        aem2.set_affect(CoreAffect(valence=0.9, arousal=0.8))
        m1 = await aem2.encode("Breakthrough moment — the algorithm finally converged.")

        aem2.set_affect(CoreAffect(valence=-0.7, arousal=0.6))
        m2 = await aem2.encode("Critical bug found in production at 2 AM.")

        aem2.set_affect(CoreAffect(valence=0.3, arousal=0.2))
        m3 = await aem2.encode("Routine code review, nothing remarkable.")

        for mem in (m1, m2, m3):
            ca = mem.tag.core_affect
            sign = "[+]" if ca.valence > 0 else "[-]"
            print(f"  {sign} {mem.content[:55]}  (valence={ca.valence:+.2f})")

        # -------------------------------------------------------------------
        # encode_batch — embed all texts in a single call to embed_batch()
        # -------------------------------------------------------------------

        print("\n=== encode_batch ===\n")

        batch_texts = [
            "Quarterly OKRs reviewed, on track.",
            "Customer escalation resolved within SLA.",
            "New team member onboarded successfully.",
            "Pipeline test suite is flaky again.",
        ]
        batch_mems = await aem2.encode_batch(batch_texts)
        print(f"Batch-encoded {len(batch_mems)} memories.")
        for mem in batch_mems:
            ca = mem.tag.core_affect
            sign = "[+]" if ca.valence > 0.1 else "[-]" if ca.valence < -0.1 else "[~]"
            print(f"  {sign} {mem.content[:55]}")

        # -------------------------------------------------------------------
        # count() and prune()
        # -------------------------------------------------------------------

        print("\n=== Maintenance ===\n")

        total = await aem2.count()
        print(f"Total memories: {total}")

        removed = await aem2.prune(threshold=0.0)  # nothing old enough to prune
        print(f"prune(threshold=0.0) removed: {removed}")

        # State persistence is sync — no await needed
        snapshot = aem2.save_state()
        aem2.load_state(snapshot)
        sm = aem2.get_state().stimmung
        print(f"Stimmung after load_state: valence={sm.valence:.3f}  arousal={sm.arousal:.3f}")

    # async with block called aem2.close() automatically

    # -----------------------------------------------------------------------
    # Retrieval under different moods
    # -----------------------------------------------------------------------

    print("\n=== Mood-congruent retrieval (manual engine) ===\n")

    # Rebuild from scratch to show retrieval in isolation
    aem3 = AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(HashEmbedder()),
        config=config,
    )

    aem3.set_affect(CoreAffect(valence=0.85, arousal=0.75))
    await aem3.encode("Won the internal hackathon with the team.")
    aem3.set_affect(CoreAffect(valence=-0.8, arousal=0.65))
    await aem3.encode("Missed the release deadline due to a last-minute blocker.")
    aem3.set_affect(CoreAffect(valence=0.4, arousal=0.3))
    await aem3.encode("Read an interesting paper on retrieval-augmented generation.")

    aem3.set_affect(CoreAffect(valence=0.8, arousal=0.6))
    results = await aem3.retrieve("team achievement", top_k=3)
    print("Top-3 under positive mood:")
    for i, mem in enumerate(results, 1):
        ca = mem.tag.core_affect
        sign = "[+]" if ca.valence > 0 else "[-]"
        print(f"  {i}. {sign} {mem.content[:60]}")

    await aem3.close()

    print("\nDone.")


asyncio.run(main())
