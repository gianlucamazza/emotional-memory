# Realistic Replay Benchmark Protocol

## Question being answered

> Can AFT recover the intended memories across scripted multi-session scenarios
> when affective state is persisted between sessions, and how does that compare
> with simpler controls?

## Benchmark shape

- Dataset: `realistic_recall_v1.json`
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
