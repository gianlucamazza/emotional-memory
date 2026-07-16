# Performance & Scaling

How much AFT costs relative to pure cosine, when the cost is dominated by the
LLM, and which store to pick as N grows.

This page is an **engineering budget**, not a claim that AFT is faster or
slower than production memory systems in every regime. For measured latency
tables see [Benchmarks](../benchmarks.md); for scientific boundaries see
[Limitations](../research/08_limitations.md).

## Cost budget by layer

| Layer | When it runs | Typical order of magnitude | Notes |
|---|---|---|---|
| Embedder (hash / SBERT / e5) | encode + retrieve | sub-ms (hash) to tens of ms (SBERT on CPU) | Shared with any RAG pipeline |
| Appraisal LLM | encode slow-path (and optional query appraisal) | **200–2000 ms/item** | Dominates when enabled |
| Keyword / preset appraisal | encode | ~0 ms | Fallback / oracle-affect benches |
| Affective state update | encode / observe | O(1), sub-ms | Core affect, momentum, mood EMA |
| Resonance link build | encode | O(candidates) scan, top-5 links | Pre-filtered when store is large |
| 6-signal score + Pass 2 | retrieve | ms on the G2 candidate pool | Pool size ≈ `top_k × candidate_multiplier` |
| OTel spans | always | **zero** without the `[otel]` extra | No-op helpers |

**Rule of thumb:** if you use `LLMAppraisalEngine` on the hot path, appraisal
latency dwarfs scoring. Use `dual_path_encoding=True` and `elaborate()` /
`elaborate_pending()` off the critical path, or batch with
`encode_batch` + `appraisal_max_concurrency`.

## Retrieve path (what actually scales with N)

1. Embed the query.
2. **G2 pre-filter** — if `len(store) > top_k × candidate_multiplier`
   (default multiplier `3`), call `search_by_embedding` for that many
   candidates; otherwise `list_all()`.
3. Pass 1: 6-signal score on the candidate pool (no spreading).
4. Spreading activation on the seed set (skipped when resonance is off or
   the activation map is empty).
5. Pass 2: re-score with resonance boost.
6. Optional APE-gated reconsolidation + Hebbian on co-retrieved links.

`InMemoryStore.search_by_embedding` is a full-scan cosine (vectorized numpy
with a **lazily rebuilt embedding matrix**). It is O(n · d) in store size.
Above ~10k memories prefer an ANN-backed store.

## Store decision tree

```text
N ≲ 1k–10k, single process, no durability needed
  → InMemoryStore  (dev, tests, small agents)

Need durability on one machine, up to ~10^6 vectors
  → SQLiteStore with [sqlite] extra (sqlite-vec ANN)

Need multi-process / remote vector DB
  → QdrantStore ([qdrant]) or ChromaStore ([chroma])

Affective mood continuity across sessions
  → pass state_store= (SQLite / Redis AffectiveStateStore)
```

| Store | Search | Persistence | Extra |
|---|---|---|---|
| `InMemoryStore` | Brute cosine + matrix cache | No | none |
| `SQLiteStore` | sqlite-vec ANN (brute only if vec table empty) | File | `[sqlite]` |
| `QdrantStore` | ANN | Server / local | `[qdrant]` |
| `ChromaStore` | ANN | Ephemeral or persistent | `[chroma]` |

API reference: [Stores](../api/stores.md).

## Config knobs that cut cost

| Knob | Effect |
|---|---|
| `dual_path_encoding=True` | Skip appraisal on encode; run `elaborate()` later |
| `appraisal_max_concurrency` | Bound parallel LLM appraisal in `encode_batch` (default 8) |
| `RetrievalConfig.candidate_multiplier` | Smaller → cheaper Pass 1/2, risk of missing non-semantic hits |
| `enable_resonance=False` | Skip link build + Pass 2 spreading (ablation / cheap path) |
| `enable_reconsolidation=False` | Skip APE tag updates on retrieve |
| Keyword / no appraisal engine | Remove LLM latency entirely |

See the [configuration guide](../tutorials/configuration_guide.md) for full
trees.

## What comparative latencies are *not*

Paper Table 3 (`aft` vs `naive_cosine` encode/retrieve ms) mixes **different
adapters** (AFT engine vs cosine baseline harness). Use it for rough
orientation, not as a pure “6-signal overhead vs cosine” micro-measurement.
The performance suite (`make bench-perf`) and the scorer microbench
(`benchmarks/perf/bench_scoring.py`) isolate in-process AFT costs with a
hash embedder.

## Related

- [Benchmarks](../benchmarks.md) — committed latency table
- [Troubleshooting](../troubleshooting.md) — slow SQLite / store issues
- [Limitations §3](../research/08_limitations.md) — LLM appraisal, SQLite threading, store coverage
- [Observability](../tutorials/observability.md) — zero-overhead OTel no-ops
