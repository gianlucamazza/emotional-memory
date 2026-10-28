# ADR 0002: Semantic pilot before shared-history v2 sizing

Status: Proposed — 2026-10-28. Documentation-only decision; no new run or result.

## Context

[External feedback](https://huggingface.co/posts/gianlucamazza/864972692893223#6ae0c3c3df87bc9e011e1acd)
questions using hash-embedder discordance to size shared-history v2. Hash
collisions may contribute to disagreements; their share has not been established
by this ADR. Repeating a result across hash variants does not establish semantic
retrieval quality, and variants on the same queries are not independent samples.

Near-neutral gated routing selects semantic-only weights. Matching cosine on
the same candidates and embeddings is an implementation invariant, not evidence
of affective benefit. In longitudinal replay, verify candidate/history equivalence
before asserting ranking equality across arms: earlier write-backs can differ.
Non-neutral gains and near-neutral always-on losses remain empirical observations
conditional on the dataset, embedder and write-back policy.

The existing 36-query, 12-scenario `shared_history_recall_v1` is a development
pilot, with oracle affect and author-constructed scenarios. It cannot become an
independent confirmatory set by changing its embedder.

## Decision

Complete semantic-embedder support, rerun the unchanged v1 pilot, inspect paired
discordance, and only then size and preregister v2. Preserve the hash runner as a
mechanism/regression control. Do not freeze N from hash-only discordance or treat
the previously discussed 54/70 query counts as a powered semantic study.

Use the existing draft [PR #186](https://github.com/gianlucamazza/emotional-memory/pull/186)
for runner configuration rather than duplicate its implementation. As inspected
on 2026-10-28, that PR is open and draft at `b50e57cb6506a4f5eb9613d43a7476b050da4667`;
this ADR neither merges it nor attests that it satisfies every requirement below.

### Implementation audit and acceptance criteria

At main `f833450ae9c7f50f73d77cbbe291dcddf249070b`:

- `benchmarks/realistic/shared_history.py::SharedHistoryAdapter.reset` constructs
  `TokenHashEmbedder()` inline; `adapters/base.py` gives it a default dimension of 64.
- `interfaces.py::Embedder` returns `list[float]`; `models.py::Memory.embedding`
  has no fixed dimension. `stores/in_memory.py` builds its matrix from stored
  vectors. `_math.py::cosine_similarity` checks equal lengths, not length 64.
- The inspected in-memory retrieval path therefore has no explicit 64-dimensional
  contract. A literal constructor swap is still insufficient for a reproducible
  benchmark: configuration, lifecycle, provenance and regression coverage matter.

Before running the pilot, require:

1. Embedder injection through runner, adapter construction and reset; fresh stores
   per scenario/arm; identical model configuration for all arms. Re-encode all
   memories when changing models; never reuse hash-vector snapshots.
2. Explicit CPU execution and lazy optional dependencies, with a clear failure
   when semantic support is unavailable; no silent fallback to hash.
3. Deterministic synthetic tests with non-64 dimensions, reset and snapshot
   round trips, mixed-dimension rejection, and same-checkpoint neutral ranking
   equivalence. Preserve unsalted hash behavior and read-only restoration checks.
4. A separately identified real-model integration test. Synthetic vector tests
   establish plumbing only; neither is a retrieval-quality result.
5. Receipts with runner commit, dataset SHA256, model id and immutable model
   revision (or verified snapshot digest), observed dimension, preprocessing,
   normalization, device, dependency versions, seed, top-k, gate threshold,
   bootstrap settings, and explicit `read_only_retrieves` true/false. Model name
   alone is insufficient. Archive commands, result files and checksums under the
   existing shared-history receipt policy, including the reproducing CI run URL.

### Pilot protocol

Before any scored or smoke execution, commit and merge a separate exploratory
pilot protocol following the repository research lifecycle. Pin
`BAAI/bge-small-en-v1.5` to an immutable revision before execution; this choice
aligns with the existing benchmark embedder factory and is not selected by pilot
scores. Freeze preprocessing and all analysis choices there. The configurable
runner's MiniLM default is not an implicit model choice for this pilot.

Run all 36 queries unchanged with cosine, gated and always-on, both with normal
write-backs and read-only retrieval. Keep event order, IDs, top-k, gate threshold,
seed and all non-embedder settings matched. Report near-neutral (`abs(v) < tau`)
and non-neutral (`abs(v) >= tau`) strata; identify boundary cases explicitly.
Use normal write-backs as the primary pilot condition and read-only as a
mechanism diagnostic, not additional independent observations.

For each comparison against cosine and each stratum, retain aligned query IDs
and top-1 correctness, and report both-correct, both-wrong, method-only-correct
(`b`) and cosine-only-correct (`c`). Report `b+c`, `(b+c)/N` and `(b-c)/N`.
Ranking disagreement is a separate diagnostic: different retrieved IDs can both
be wrong and do not constitute discordant correctness pairs. Inspect ties and
candidate sets. Report zero discordance as an observation, not equivalence.

Preserve scenario membership. Queries within a shared history are dependent;
uncertainty and planning must respect scenario clusters. With only 12 scenarios,
report sensitivity to assumptions rather than a precise sample-size promise.
Any query-level exact paired test is descriptive and must disclose its
independence assumption. Do not pool write-back modes or hash variants into N.

### Gate to v2

After publishing the pilot receipt, write a separate v2 preregistration before
authoring or scoring its confirmatory set. Fix the target population, independent
scenario sampling, primary contrast/stratum, smallest useful effect, alpha,
power, multiplicity policy, stopping rule and resource limit. Estimate sample
size using semantic paired outcomes and their cluster structure, with sensitivity
to uncertainty in discordance and effect size; discordance alone is insufficient.
Report required independent scenarios as well as queries. Keep pilot histories
out of confirmation and freeze model/configuration before confirmation.

If the pilot is too sparse, the projected budget is infeasible, or no useful
effect is supported, publish an inconclusive/negative pilot and either stop or
preregister additional development data. Do not assume all future discordant
pairs will favor gated, switch models after seeing scores without labeling the
new analysis exploratory, or reduce N after looking at confirmatory outcomes.

## Consequences

This separates gate wiring, embedder sensitivity and evidence of retrieval
benefit. It delays v2 sizing until a relevant pilot exists and adds model-download
and receipt costs. Semantic embeddings still do not remove oracle affect,
author-construction bias or establish production/human utility. Existing results,
receipts and public claims remain unchanged. ADR 0001's human-grounded study is
a separate evidence track and is not replaced by this pilot.

## Alternatives

- Size v2 immediately from hash discordance: rejected because the effect and
  discordance may not transfer to semantic embeddings.
- Add more hash salts first: useful regression diagnostics, insufficient for
  semantic power planning.
- Replace only the inline constructor: insufficient provenance and lifecycle
  coverage, despite dimension-generic in-memory interfaces.
- Treat the 36-query rerun as confirmation: rejected because it is already
  inspected development data.
