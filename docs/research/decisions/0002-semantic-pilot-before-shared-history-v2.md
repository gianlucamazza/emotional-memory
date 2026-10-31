# ADR 0002: Semantic pilot before shared-history v2 sizing

Status: Proposed — 2026-10-28. Documentation-only decision; no new run or result.

## Context

[External feedback](https://huggingface.co/posts/gianlucamazza/864972692893223#6ae0c3c3df87bc9e011e1acd)
questions using hash-embedder discordance to size shared-history v2. Hash
collisions may contribute to disagreements; their share has not been established
by this ADR. Repeating a result across hash variants does not establish semantic
retrieval quality, and variants on the same queries are not independent samples.

Near-neutral gated routing calls the same retrieval path as the cosine arm
with weights `[1,0,0,0,0,0]` (`engine_shared.semantic_only_weights`,
`engine_shared.is_query_affect_neutral`; `retrieval.retrieval_breakdown`
and `retrieval.build_retrieval_plan` then score only query–memory cosine
via `retrieval._component_values`). Write-back
(`EmotionalMemory._apply_retrieval_updates`) updates memory tags only
(counters, reconsolidated affect, predictions, resonance links); it never
changes content, embeddings, IDs or store membership.
`InMemoryStore.update` reassigns an existing key, so
`InMemoryStore.list_all` and `InMemoryStore.search_by_embedding` keep
candidate order. In the in-memory replay with a deterministic embedder,
near-neutral gated therefore ranks identically to cosine even after the
two arms' write-back histories diverge, because both stores keep the
same content, embeddings, IDs and candidate order. The reported 8/8
near-neutral agreement across hash-embedder variants is a wiring check,
not evidence of benefit. Re-verify if write-back starts changing
content, embeddings or membership, the embedder becomes
non-deterministic, or a store backend other than `InMemoryStore` is
used (`SQLiteStore.update` re-inserts the vector row, so tie order is
not shown to be preserved there; approximate-index backends may also
reorder).
Non-neutral gains and near-neutral always-on losses remain empirical
observations conditional on the dataset, embedder and write-back policy.

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

Track the pre-run protocol and diagnostics in [issue #193](https://github.com/gianlucamazza/emotional-memory/issues/193).

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
   round trips, mixed-dimension rejection, same-checkpoint neutral ranking
   equivalence, and neutral ranking equivalence after divergent write-back
   histories (tags differ, embeddings equal). Preserve unsalted hash behavior
   and read-only restoration checks.
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

Dipankar Sarkar observed that gated-versus-cosine top-1 discordant pairs
on `shared_history_recall_v1` (hash embedder, commit
`441207d13f83b3dbefe4f17039ff68707072085f`) differ by write-back mode.
Write-back: `b=1`, `c=0` on bakery `bk_q_inspection_not_gasket`.
Read-only: `b=1`, `c=0` on ferry `fs_q_no_sailing`. Both are query 3 of 3.
Our follow-up ablation, swapping one tag field at a time before query 3,
attributed both to carry-over: the reconsolidated `core_affect` for the
ferry, and `core_affect` together with `retrieval_count` for the bakery.

Freeze this labelled sensitivity rule as part of the pilot protocol.
Pairs discordant under write-back only are labelled
`carry-over-dependent`. Pairs discordant under read-only only are
labelled `carry-over-suppressed`. Report primary `b`, `c` and ICC both
with and without carry-over-dependent pairs. This is a sensitivity
analysis, not an exclusion. Dropping only write-back-only pairs would
bias the result toward read-only.

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

### Within-scenario discordance and planning sensitivity

The semantic pilot must report within-scenario dependence, not assume it away.
For each method-versus-cosine comparison, write-back condition and affect
stratum, define correctness discordance as
`D_i = 1[top1_correct_method != top1_correct_cosine]`. Retain the signed paired
outcome `Z_i = top1_correct_method - top1_correct_cosine` separately: correlation
of discordance incidence does not by itself specify variance of the paired
effect. Ranking disagreement is not the definition of either variable.

Publish a scenario table with query counts, stratum counts, `b`, `c`, discordance
rates and signed effects. Include a column that marks each discordant pair
as write-back only, read-only only, or both. Report that column per arm
and stratum, listing the discordant query IDs per mode. Predeclare in the
separate pilot protocol the observed-scale ICC estimator for `D_i`, its
uncertainty procedure, and handling of sparse strata, unequal cluster sizes
and boundary/degenerate estimates.
Report the exploratory ICC and uncertainty when estimable. If outcomes are
constant, the fit is unidentified, or a stratum has insufficient informative
clusters, report it as not estimable with the reason; never substitute zero.
Near-neutral gated-versus-cosine discordance is identically zero by
mechanism, so its ICC is not estimable there.
Do not pool arms, strata or write-back conditions to manufacture replication.

Twelve author-constructed histories may permit a descriptive estimate; they do
not justify treating it as a precise population ICC for v2. Publish the number
of informative clusters and limitations alongside any estimate. Freeze these
rules before execution, and do not choose a sizing assumption from whichever
pilot ICC gives the smallest N.

As a transparent sensitivity illustration, for equal cluster size `m` and
exchangeable observed-scale correlation `rho`, the usual mean-variance design
effect is `DE = 1 + (m - 1) * rho`
([Campbell et al., 2012, CONSORT cluster extension](https://doi.org/10.1136/bmj.e5661)).
With three queries per history, `DE = 1 + 2 * rho`; `rho = 0.25` gives `DE = 1.5`
and nominal `N_eff = 36 / 1.5 = 24`. These are illustrative assumptions, not
measured pilot results or a validated power calculation for paired correctness.
Stratum-specific cluster sizes can differ from three. In
`shared_history_recall_v1` they are 1–2 per scenario (near-neutral:
10 scenarios with 1 query and 2 with 2; non-neutral: 10 with 2 and
2 with 1), so most near-neutral clusters carry no within-scenario
information.

For planning, show at least `rho = 0, 0.10, 0.25, 0.50, 1.0` as declared
sensitivity cases, not confidence bounds. Model the joint paired outcomes and
scenario dependence of the signed effect for the intended primary analysis;
do not inflate a query-level exact test using the ICC of `D_i` alone. Report
required independent histories and total queries, including cluster-size
assumptions. Additional queries in an existing history do not create additional
independent histories.

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
