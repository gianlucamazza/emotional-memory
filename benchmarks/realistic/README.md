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
- **French slice (Hm1, Addendum M, 30 scenarios / 120 queries, me5)**: AFT top1=0.31
  [0.23, 0.39] vs naive_cosine top1=0.12 [0.07, 0.18], Δ=+0.18 [0.11, 0.26],
  p_bootstrap<0.0001, Hedges g=0.424, p_McNemar<0.0001 — Branch A PASS. Cross-language
  signal confirmed: same_topic_distractor (0.42) and affective_arc (0.38) drive the
  advantage; momentum_alignment weak (0.12) consistent with EN findings.
- 4 of 5 challenge types in v2 are by construction favourable to affective retrieval —
  aggregate advantage should be read as scope-conditional, not general superiority.


## Opt-in shared-history replay

```bash
uv run python -m benchmarks.realistic.runner --shared-history --embedder hash \
  --seed 7 --out-json /tmp/shared.json --out-md /tmp/shared.md \
  --out-protocol /tmp/shared.protocol.json
```

Interleaved shared-history (`shared_history_recall_v1`; encodes after recalls;
declared near-neutral query share) is offline with the hash embedder:

```bash
python -m benchmarks.realistic.runner --shared-history --embedder hash --seed 7 \
  --dataset benchmarks/datasets/shared_history_recall_v1.json
```

To measure write-back cost, restore a deep-copied snapshot after each retrieve:

```bash
python -m benchmarks.realistic.runner --shared-history --embedder hash --seed 7 \
  --dataset benchmarks/datasets/shared_history_recall_v1.json \
  --read-only-retrieves
```

Scores from these commands are a mechanism test, not a semantic-quality result.
They may be committed or published (including on the Hugging Face dataset card)
only with a receipt: the repo commit SHA they were produced at; the report
files (`results.shared_history.json` and `results.shared_history.protocol.json`, with input hashes);
the embedder and seed; whether `--read-only-retrieves` was set; and the CI run
that reproduced them. Every published score carries the label «mechanism test
(hash embedder), not a semantic-quality benchmark». No receipt, no score.

This separate mode reuses the dataset and report pipeline without changing legacy
results. With no output flags it writes `results.shared_history.json`,
`results.shared_history.md` and `results.shared_history.protocol.json`. Every scenario starts from the same empty checkpoint in three independent
engines: cosine, query-gated and always-on query-affect ranking. All retain retrieval
write-backs. Events encode the dataset's affect; query `state` supplies recorded query
affect to the scorer without resetting runtime state. Missing query affect is rejected.

The deterministic offline backend uses TokenHashEmbedder, UUID5 IDs and a fixed UTC
clock advancing one second per operation (including session boundaries). It is a
mechanism/reproducibility fixture, not a semantic-quality benchmark. Run it in a
dedicated single-threaded process: scoped library clock substitutions are not safe
alongside concurrent production engine calls. No remote models or LLM calls occur.
Each report includes input hashes and full memory/state snapshots after every operation.
Affective state can remain equal while retrieval counts, predictions or links diverge.
