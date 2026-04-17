# Async usage

`emotional_memory` exposes a full async API via `AsyncEmotionalMemory` and a
thin `as_async()` convenience wrapper.  Both share the same 5-layer AFT
pipeline as the sync engine — the only difference is that `encode`,
`retrieve`, and I/O-bound store calls are `await`-able.

## Installation

```bash
pip install emotional-memory
```

No extra extras required for async — it is included in the base package.

## Two paths to async

### Path 1 — `as_async()` wrapper (recommended for migration)

Wrap an existing sync engine in one call.  The wrapper shares the live
`_state` object, so mood and momentum persist across both sync and async calls.
Do **not** drive the same engine from multiple threads concurrently.

```python
import asyncio
from emotional_memory import (
    EmotionalMemory, AsyncEmotionalMemory,
    InMemoryStore, CoreAffect,
    SyncToAsyncEmbedder, SyncToAsyncStore,
    as_async,
)


class HashEmbedder:
    """Deterministic 8-dim embedder — no ML dependencies required."""
    DIM = 8

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(self.DIM)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


async def main() -> None:
    sync_em = EmotionalMemory(store=InMemoryStore(), embedder=HashEmbedder())
    sync_em.set_affect(CoreAffect(valence=0.8, arousal=0.7))

    aem = as_async(sync_em)          # wrap — one line

    mem = await aem.encode("Shipped the new feature ahead of schedule.")
    print(mem.content, mem.tag.core_affect)

    results = await aem.retrieve("feature release", top_k=2)
    print(f"Retrieved {len(results)} memories.")


asyncio.run(main())
```

### Path 2 — native `AsyncEmotionalMemory` with async adapters

Use `SyncToAsyncStore` and `SyncToAsyncEmbedder` to bridge sync I/O objects,
or supply natively-async implementations of `AsyncEmbedder` /
`AsyncMemoryStore`.

```python
async def main() -> None:
    async_store = SyncToAsyncStore(InMemoryStore())
    async_embedder = SyncToAsyncEmbedder(HashEmbedder())

    async with AsyncEmotionalMemory(
        store=async_store,
        embedder=async_embedder,
    ) as aem:
        aem.set_affect(CoreAffect(valence=0.9, arousal=0.8))
        await aem.encode("Breakthrough — the algorithm finally converged.")

        aem.set_affect(CoreAffect(valence=-0.7, arousal=0.6))
        await aem.encode("Critical bug found in production at 2 AM.")

        results = await aem.retrieve("algorithm bug", top_k=2)
        for r in results:
            print(r.content, r.tag.core_affect)
    # close() called automatically by async context manager
```

## Batch encoding

`encode_batch()` calls `embed_batch()` once for all texts — more efficient
than looping over `encode()` when the embedder supports batching (e.g.
`SentenceTransformerEmbedder`).

```python
async with AsyncEmotionalMemory(store=async_store, embedder=async_embedder) as aem:
    texts = [
        "Quarterly OKRs reviewed, on track.",
        "Customer escalation resolved within SLA.",
        "New team member onboarded successfully.",
    ]
    mems = await aem.encode_batch(texts)
    print(f"Encoded {len(mems)} memories.")
```

## Maintenance — count and prune

```python
async with AsyncEmotionalMemory(store=async_store, embedder=async_embedder) as aem:
    total = await aem.count()
    print(f"Total memories: {total}")

    removed = await aem.prune(threshold=0.05)
    print(f"Pruned {removed} weak memories.")
```

`prune()` removes memories whose `compute_effective_strength()` score falls
below `threshold` (ACT-R power-law decay — see [Decay](../api/decay.md)).

## State persistence (sync)

`save_state()` and `load_state()` are synchronous — no `await` required.
State captures mood, momentum history, and the mood-decay config.

```python
snapshot = aem.save_state()   # returns a JSON-serialisable dict
# ... across sessions ...
aem.load_state(snapshot)      # restores valence/arousal/momentum
```

## Mood-congruent retrieval

The async engine applies the same 6-signal scoring as the sync engine
(semantic similarity, mood congruence, core affect proximity, momentum
alignment, recency, resonance).  Set mood with `set_affect()` before
`retrieve()` to steer results toward emotionally matching memories.

```python
aem.set_affect(CoreAffect(valence=0.8, arousal=0.6))
results = await aem.retrieve("team achievement", top_k=3)
```

## See also

- [`AsyncEmotionalMemory` API reference](../api/async_engine.md)
- [`EmotionalMemory` (sync)](../api/engine.md)
- [Persistence tutorial](persistence.md)
- `examples/async_usage.py` in the repository
