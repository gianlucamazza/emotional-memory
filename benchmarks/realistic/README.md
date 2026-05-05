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

Current interpretation (v2, N=200):

- **v2 (50 scenarios / 200 queries, 5 challenge types)**: SBERT Δ top1 = +0.125
  [p<0.001, d=0.286]; e5-small-v2 Δ = +0.155 [p<0.001]; aggregate hit@k AFT
  advantage +0.205 (SBERT) and +0.155 (e5). Per-challenge breakdown: `affective_arc`
  (+0.275) and `momentum_alignment` (+0.27) drive the signal; `same_topic_distractor`
  and `semantic_confound` show no significant advantage.
- **Italian slice (G6, 20 scenarios / 80 queries)**: SBERT hit@k Δ = +0.15 [p=0.0005];
  me5-small Δ = +0.16 [p=0.001]. EN-only embedder is the accuracy bottleneck, not AFT.
- **Spanish slice (Hd2_ES, 20 scenarios / 80 queries)**: SBERT Δ = +0.138 [p=0.045,
  d=0.233] — PASS; me5-small Δ = +0.113 [p=0.110] — FAIL (borderline).
- 4 of 5 challenge types in v2 are by construction favourable to affective retrieval —
  aggregate advantage should be read as scope-conditional, not general superiority.
