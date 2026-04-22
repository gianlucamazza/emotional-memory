# Comparative Benchmark

Measures **recall@k** (mood-congruent retrieval accuracy) and **latency** for
emotional-memory (AFT) against alternative memory systems on the
`affect_reference_v1` dataset (258 affect-labeled examples).

This is a **controlled synthetic benchmark** focused on affect-aware retrieval.
It is useful for probing whether AFT changes ranking behavior in the intended
direction, but it does **not establish general downstream superiority** over
production memory systems.

The stable interpretation contract lives in [protocol.md](protocol.md).

## Quick start

```bash
# AFT vs naive cosine baseline (no external deps)
python -m benchmarks.comparative.runner

# Add mem0 if installed
python -m benchmarks.comparative.runner --systems aft,naive_cosine,mem0

# Custom top-k and output path
python -m benchmarks.comparative.runner --top-k 10 --out my_results.csv
```

## Protocol

- Dataset: `affect_reference_v1.jsonl` (synthetic, affect-labeled, public in-repo)
- Query set: 4 representative queries, one per Russell quadrant
- Score: fraction of top-k results whose stored affect falls in the correct quadrant
- Goal: compare retrieval behavior under a controlled affect-aware probe

For affect-aware adapters such as AFT, the benchmark passes the query centroid
(`valence`, `arousal`) directly to the adapter so mood-congruent scoring is
actually active. General-purpose baselines may ignore those fields entirely.

This means the protocol is fair for the narrow question:

> does the system surface memories from the intended affective region under a
> controlled query-state setup?

It is **not** a fair answer to broader questions such as:

- which memory system is best overall for agent products
- which system gives the best downstream answer quality
- which system has the best production latency across heterogeneous backends

## Metric — recall@k (mood-congruent)

For each of 4 Russell quadrant queries, the benchmark retrieves top-k items and
counts how many are **mood-congruent** (same valence sign × arousal level as the
query).  `recall@k = congruent / min(k, retrieved)`.  Final score is the mean
across all 4 queries.

A system that retrieves purely by semantic cosine similarity will score ~0.25
on a balanced dataset (random quadrant hit rate). AFT's multi-signal scorer
is expected to score higher on this benchmark because it explicitly uses mood
congruence, affect proximity, and momentum alignment.

## Systems

| System | Status | Notes |
|---|---|---|
| `aft` | ✅ always available | emotional-memory with HashEmbedder |
| `naive_cosine` | ✅ always available | pure cosine, same embedder as AFT (intra-paper baseline) |
| `recency` | ✅ always available | non-semantic recency-only baseline |
| `mem0` | ⚠️ optional | `pip install mem0ai` + OpenAI key required |
| `langmem` | ⚠️ optional | `pip install langmem langgraph` + OpenAI key required |
| `letta` | ⚠️ optional | cloud/API-key gated; may remain `not_evaluated` |

Optional adapters such as `mem0`, `langmem`, and `letta` are informative
reference points, but they were designed for broader product use cases than this
single retrieval probe. Treat their results here as exploratory comparisons, not
leaderboard-style rankings.

### Adding a new system

1. Create `adapters/my_system.py` implementing `MemoryAdapter` (see `adapters/base.py`)
2. Register the class in `runner.py::_make_adapters()`
3. Add a row to the table above

**Time-box rule**: if a baseline cannot be installed and running within 4 hours,
document `status="not_evaluated"` with `reason` in the CSV and continue.

## Output files

| File | Description |
|---|---|
| `results.csv` | Machine-readable results |
| `results.md` | Human-readable Markdown table |
| `results.protocol.json` | Machine-readable protocol metadata and caveats |

Run `make bench-comparative` to regenerate the results and protocol metadata.
