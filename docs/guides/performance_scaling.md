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

## Measured breakdown (Wave 2 — H12)

Local medians from
`uv run python -m benchmarks.perf.bench_profile_breakdown`
(CPU, `InMemoryStore`, warm matrix cache; SBERT = `all-MiniLM-L6-v2`).
Regenerate on your machine; absolute ms vary, **shares** are the decision signal.

| Embedder | N | embed ms | prefilter ms | AFT plan ms | e2e retrieve ms | plan/e2e | embed/e2e |
|---|---:|---:|---:|---:|---:|---:|---:|
| hash-64 | 100 | 0.02 | 0.06 | 0.91 | 1.78 | 51% | 1% |
| hash-64 | 1 000 | 0.02 | 0.10 | 1.01 | 8.92 | 11% | 0% |
| sbert-minilm | 100 | 23.9 | 0.13 | 2.22 | 29.1 | **7.6%** | **82%** |
| sbert-minilm | 1 000 | 19.9 | 0.34 | 1.52 | 66.2 | **2.3%** | **30%** |

**Gate for further scorer optimisations (H5):** open a code PR only if AFT plan
share of e2e under SBERT is **≥ 15%**. Measured max is **7.6%** → **DECLINE H5**
(inter-pass s1–s5 cache would save sub-ms under G2 and is not worth fidelity risk).

Notes:

- Under **hash**, plan can look large as a *percentage* because e2e is only a few
  ms — absolute cost stays ~1 ms on the G2 pool (~15 candidates).
- Under **SBERT**, query embedding dominates; e2e − (embed+prefilter+plan) also
  includes reconsolidation / Hebbian store updates on the top-k path.
- Scorer-only isolation (no embedder): `benchmarks/perf/bench_scoring.py`
  (~2.4–3× pure cosine rank on a fixed pool).

### H13 — dual-path encode (appraisal cost) — **CLOSED PASS**

**Hypothesis:** with appraisal on the encode path, `dual_path_encoding=True`
removes appraisal latency from the hot path and defers it to
`elaborate()` / `elaborate_pending()` (work is deferred, not free).

Harness:

```bash
make bench-perf-h13-sim          # offline structural (no network)
# Live (OpenAI-compatible). Example with local Ollama:
EMOTIONAL_MEMORY_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
EMOTIONAL_MEMORY_LLM_API_KEY=ollama \
EMOTIONAL_MEMORY_LLM_MODEL=llama3.2:1b \
  uv run python -m benchmarks.perf.bench_profile_breakdown --h13-only --llm-encode
```

Live harness uses `DIRECT_VAD_SCHEMA` + `cache_size=0` (latency measure, not
Scherer quality). Reports **INCONCLUSIVE** if `fallback_count > 0` (bad key /
parse). Unit coverage: `tests/test_h13_overhead.py` (in `make check`).

**Measured — sim** (delay=150 ms/appraise, n=5, hash embedder, resonance off):

| Arm | Wall time | Appraise calls |
|-----|----------:|---------------:|
| Sync encode | ~151 ms/item | 5 |
| Dual-path encode | ~0 ms/item | **0** |
| `elaborate_pending` after dual | ~150 ms/item | 5 |
| Dual hot + elaborate combined | ~1.0× sync | — |

**Measured — live** (Ollama `llama3.2:1b` @ `127.0.0.1:11434`, n=4,
`fallback_count=0`, 2026-07-16):

| Arm | Wall time | Notes |
|-----|----------:|-------|
| Sync encode | ~7.6 s/item | full LLM appraisal on hot path |
| Dual-path encode | ~1 ms/item | **0.00×** sync; no LLM |
| `elaborate_pending` | ~7.1 s/item | deferred LLM |
| Dual + elaborate combined | ~0.93× sync | work deferred, not free |

**Verdict H13 PASS (structural + live):** dual-path hot path ≈ 0× sync; combined
work ≈ sync. Absolute live ms are model/hardware-specific (local 1B vs cloud
API); the **ratio** is the durable claim. See also
[Limitations §3.1](../research/08_limitations.md).

**Do not** flip the library default to dual-path without a product decision —
it changes encode semantics (`pending_appraisal=True` until elaborate).

## Async note (H10)

`AsyncEmotionalMemory` awaits embed/store/appraise I/O, but **CPU scoring**
(`build_retrieval_plan`, decay, resonance) still runs **inline on the event
loop**. For multi-tenant high-QPS services, offload retrieve to a worker thread
or process at the application boundary; the library does not auto-`run_in_executor`
(keeps determinism and test simplicity). See [Async usage](../tutorials/async.md).

## Config anti-patterns (H14)

- Setting `candidate_multiplier` very large (or always calling
  `list_all`-scale scoring) defeats G2 and makes the 6-signal loop O(n).
- Leaving `LLMAppraisalEngine` on every encode without dual-path/batch is the
  usual “AFT feels slow” report — it is the LLM, not the scorer.

## Evaluated & declined (overhead levers)

Written so a future pass does not silently re-open them without new evidence:

| Lever | Verdict | Why |
|---|---|---|
| H5 Pass2 recompute s1–s5 cache | **DECLINED** (2026-07) | plan/e2e under SBERT ≤ 7.6% < 15% gate |
| H6 encode `list_all` below resonance prefilter | **DECLINED** | only for small N; absolute cost tiny |
| H7 `encode_batch` shared embedding matrix for links | **DECLINED** | encode dominated by embed/LLM, not link build |
| H8 SQLite brute-force vectorize | **DECLINED** | path only when vec table empty |
| H9 Pydantic `model_copy` on reconsolidation | **DECLINED** | micro-allocation; no prod signal |
| H11 `prune` / `elaborate_pending` full scan | **DECLINED** | batch/offline paths; ANN store still lists by design |
| Spreading adjacency cache | **DECLINED** (ROADMAP) | G2 pool ~15 nodes |
| ACT-R multi-trace spacing | **DECLINED** (ROADMAP) | breaks fidelity log-log invariant |
| Hebbian LTD | **DEFERRED** (research addendum only) | needs pre-reg; not an engineering perf PR |

## Related

- [Benchmarks](../benchmarks.md) — committed latency table
- [Troubleshooting](../troubleshooting.md) — slow SQLite / store issues
- [Limitations §3](../research/08_limitations.md) — LLM appraisal, SQLite threading, store coverage
- [Observability](../tutorials/observability.md) — zero-overhead OTel no-ops
- Profile CLI: `make bench-perf-profile`
