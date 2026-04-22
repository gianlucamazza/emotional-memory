# Realistic Replay Benchmark

This benchmark is the next step after the synthetic comparative probe. It uses
scripted multi-session scenarios, persisted affective state, and memory carry-
over across sessions to compare AFT with simpler controls under more realistic
conditions.

Headline metric: `top1_accuracy`. `hit@k` remains secondary support.

## What it is for

- replaying multi-session memory traces
- testing persisted affective state across session boundaries
- checking whether the right memories are recovered from natural-language queries
- inspecting retrieval explanations on realistic but still controlled scenarios
- comparing AFT against semantic-only and recency-only controls

## What it is not

- a production agent benchmark
- a human evaluation
- proof of general downstream superiority

The stable interpretation contract lives in [protocol.md](protocol.md).

## Quick start

```bash
make bench-realistic
```

This writes:

- `benchmarks/realistic/results.json`
- `benchmarks/realistic/results.md`
- `benchmarks/realistic/results.protocol.json`

## Dataset shape

The benchmark reads `benchmarks/datasets/realistic_recall_v1.json`.

Each scenario contains:

- multiple sessions
- non-trivial candidate pools at each query (`candidate_count > top_k`)
- per-session encoded events with explicit affective context
- distractors from the same semantic domain and emotionally adjacent distractors
- natural-language queries with expected target memories
- an explicit `challenge_type` for each query
- optional query-time affective state for retrieval probing

## Execution model

- AFT carries memories across sessions via `export_memories()` / `import_memories()`
- AFT persists affective state via `SQLiteAffectiveStateStore`
- semantic-only and recency baselines share the same scripted scenario format
- AFT retrieval uses `retrieve_with_explanations()` so reports can capture ranking-time signals

This keeps the benchmark reproducible and local while still exercising
state continuity, which was absent from the synthetic comparative probe.

The runner also reports, for every query:

- candidate memory count
- expected target recency ranks
- challenge type
- whether the query would be trivial for a recency baseline

## Systems

- `aft`: full AFT replay with persisted affective state
- `naive_cosine`: semantic-only control with the same deterministic embedder
- `recency`: no semantic or affective reasoning

Current interpretation:

- this benchmark now separates AFT from recency-only controls under non-trivial
  candidate pools
- the current small dataset also shows an AFT edge over `naive_cosine`, with the
  strongest gains concentrated in `affective_arc` queries
- the stressed `semantic_confound` subset remains difficult and does **not**
  currently show an AFT advantage over `naive_cosine`
- it still does **not** establish a robust general advantage over semantic-only
  baselines or production memory systems
