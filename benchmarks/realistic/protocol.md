# Realistic Replay Benchmark Protocol

## Question being answered

> Can AFT recover the intended memories across scripted multi-session scenarios
> when affective state is persisted between sessions, and how does that compare
> with simpler controls?

## Benchmark shape

- Dataset: `realistic_recall_v1.json` (v1), `realistic_recall_v2.json` (v2, 50 scenarios, 200 queries, 5 challenge types), `realistic_recall_v2_it.json` (v2 Italian multilingual slice, G6 N=80), `realistic_recall_v2_es.json` (Spanish slice, Hd2_ES N=80), `realistic_recall_v2_fr.json` (French slice, Addendum M Hm1 N=120)
- Type: scripted multi-session replay benchmark
- Execution model:
  - persistent affective state for AFT
  - memory carry-over across sessions
  - natural-language retrieval queries
- Systems:
  - `aft`
  - `naive_cosine`
  - `recency`
- Primary metrics:
  - `top1_accuracy`
  - `hit@k`
- Secondary metrics:
  - `stateful_session_rate`
  - `memory_count_growth`
  - `candidate_count`
  - `recency_triviality`
  - `challenge_type`

## Difficulty guardrails

- Dataset validation rejects any query where `candidate_count <= top_k`.
- Each scenario must contain at least one query that a pure recency baseline
  would miss at the configured `top_k`.
- The dataset must declare explicit `challenge_type` values, including at least
  one `semantic_confound` query.
- Generated results report per-query candidate counts and expected recency ranks
  so trivial windows are inspectable.

## What this benchmark supports claiming

- the repo can now run replayable multi-session studies
- affective state continuity can be exercised and compared under a documented protocol
- the current protocol distinguishes AFT from recency-only retrieval under
  non-trivial candidate pools
- the current small replay dataset shows the clearest AFT gains on
  `affective_arc` queries
- the stressed `semantic_confound` slice is now large enough to inspect directly,
  and currently remains a weak area for AFT
- retrieval explanations can be inspected on more realistic scenarios than the
  quadrant-only synthetic probe

## What this benchmark does not support claiming

- production superiority over other memory systems
- broad downstream task superiority
- a robust advantage over semantic-only retrieval in general settings
- ecological or human validation
- robust cross-domain performance outside the scripted scenarios

## Required reporting

Every generated result set should report:

- dataset name and version
- difficulty profile (`minimum_candidate_count`, `nontrivial_query_rate`)
- challenge-type counts
- systems included
- scenario count and query count
- memory/state persistence model used
- aggregate and per-scenario metrics
- challenge-type aggregates
- per-query candidate counts and expected target recency ranks
- the interpretation caveats above

## Shared-history longitudinal mode (opt-in)

`--shared-history --embedder hash` selects the #170 shared-history replay
(`shared_history_v1`) when the dataset has no `timeline`. This is a new
protocol, not a reinterpretation of earlier benchmark reports. Each scenario
starts with an identical empty checkpoint in three independent engine instances:

| Arm | Ranking | Retrieval write-back |
|---|---|---|
| cosine | Semantic-only weights | Retained locally |
| gated | Cosine for neutral recorded query affect; affect-aware otherwise | Retained locally |
| always_on | Query-affect ranking on every query | Retained locally |

The cosine arm controls ranking inside the same engine; it is distinct from the
legacy `naive_cosine` adapter, which has no reconsolidation. The always-on arm uses
query affect, not the demo's mood-state strategy. The manifest records the gate
threshold and complete effective engine configuration.

Events and queries follow dataset order.

For `realistic_recall_v1` and any session without a `timeline`, that order is:
session start, all session encode events, all session queries, session end. That
is the #170 shared-history replay and remains byte-identical.

For `shared_history_recall_v1` (`interleave: true`), each session declares a
`timeline` of `{kind, id}` steps. Shared-history mode honors that order, so
encodes may follow recalls inside a session (for example `E E E R E R E R`).
An empty `timeline` is rejected; omit the field entirely for the #170
events-then-queries order. Expected memory IDs may only name events already
encoded at that query. The comparative runner rejects interleaved datasets;
they are for `--shared-history` only. Reports use protocol mode
`shared_history_timeline` and `event_order: dataset_timeline`.
`--read-only-retrieves` restores a deep-copied engine snapshot after each
retrieve so write-backs do not persist; the read-only path asserts that store
and affective-state snapshots are unchanged after every retrieve, on every
arm. Use it as a control when measuring write-back cost.

Encode-after-recall counts are scenario-wide (24 on this dataset). Counted
within a single session they are 22: two encodes sit after a recall in an
earlier session of the same scenario.

No synchronization of states occurs after the initial checkpoint. Session
boundaries round-trip the affective state through its public serialization API.
Query affect is a recorded oracle input, never a claim about appraisal model
accuracy. Missing query affect fails validation. Near-neutral queries use the
production rule `|valence| < query_affect_gate_tau` (default `0.2`, strict `<`);
values exactly at `0.2` are not gated to cosine.

Reports include ordered external inputs, timestamps, deterministic IDs and full
post-event snapshots for every arm. Divergence is reported against cosine as
snapshot equality, affective-state equality and the count of changed memories.
A difference is not evidence of better memory. Recall can change tags, predictions
and links without changing the affective state.

### Relevance and ranking metrics

Expected memory IDs are dataset judgments fixed before executing any strategy.
They must all reference already encoded memories. Relevance is binary, not graded;
this protocol does not invent graded labels for nDCG.

- Recall@k: distinct relevant returned IDs divided by all relevant IDs.
- nDCG@k: binary gain discounted by `log2(rank + 1)`, divided by ideal DCG@k.
- MRR@k: mean reciprocal rank of the first relevant result within the returned k;
  zero for a miss. This is truncated MRR, not an estimate of unobserved ranks.
- A repeated returned ID earns no additional relevance gain.
- With one relevant ID, Recall@k equals Hit@k.

All three are macro-averaged by query, scenario and challenge type. Existing
headline metrics stay unchanged. Pairing uses `(scenario_id, session_id, query_id)`
to avoid collisions across sessions or scenarios. Existing query-level bootstrap
intervals are descriptive: dependent queries within a scenario do not provide
independent experimental samples. Confirmatory claims require a preregistered
scenario-level resampling design and held-out evaluation; no new superiority
claim is made here.

### Reproducibility scope

The JSON and protocol exports include dataset hash, source-file hashes, Git
revision when available, package/Python/dependency versions, platform, seed,
configuration, embedder and clock policy. Source hashes identify working-tree code
even when it differs from the Git revision. The seed controls deterministic IDs
and bootstrap resampling. The clock advances one second per operation from a
fixed UTC epoch. Traces are deterministic in the declared environment; portability
across floating-point runtimes is not promised.

The offline TokenHashEmbedder fixture validates mechanics, not semantic quality.
Use a dedicated single-threaded process because clock substitutions are scoped
module patches. Scaling to neural embedders or external appraisal requires frozen,
versioned outputs and a separately declared protocol. Full snapshots can be large;
these reports are research artifacts, not an online telemetry format.
