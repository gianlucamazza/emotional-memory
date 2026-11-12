# Addendum AB — Semantic shared-history exploratory pilot

**Status:** OPEN — documentation-only pre-registration. This document does not
authorize a scored run, a smoke run, or any change to published scores.
**Date (pre-reg):** 2026-11-12
**Parent decision:** [`docs/research/decisions/0002-semantic-pilot-before-shared-history-v2.md`](../docs/research/decisions/0002-semantic-pilot-before-shared-history-v2.md)
(ADR 0002, merged via PR #192).
**Tracking issue:** [#193](https://github.com/gianlucamazza/emotional-memory/issues/193).
**Runner:** PR #186 merged at `b209397816936e90dec6dbe435f145866ab03e84`.
**Development set:** unchanged `shared_history_recall_v1` (36 queries, 12 scenarios).

This is the separate exploratory pilot protocol required by ADR 0002. It freezes
analysis choices **before** execution. It is not confirmation, not a public-benefit
claim, and not a confirmatory sample-size calculation.

---

## Non-claims (apply to this document and to any later closure)

- Do not run the pilot from this pull request. Merge this protocol first.
- Do not change published scores, receipts, or cards.
- Do not add results, predicted scores, or “expected” numerical outcomes.
- Do not claim public, production, or human retrieval benefit.
- Do not promise a powered v2 N, an effective sample size as a planning target,
  or that the illustrative design-effect arithmetic below is a validated power
  calculation.
- Do not treat hash-embedder discordance, the previously discussed 54/70 query
  counts, or the timeline 8/8 near-neutral agreement as semantic evidence.
- Do not name or publish any private Hugging Face Space. Do not use paid Hugging
  Face hardware, Inference Endpoints, or Spaces for this pilot.
- Do not pool write-back modes, affect strata, arms, or hash salts into an
  independent N.
- Recorded query affect remains an oracle input. Author-constructed scenarios
  remain development data. Semantic embeddings do not remove those limits.

---

## 1. Frozen configuration and receipts

Issue #193 box 1. All of the following are frozen **now**. Execution records the
same fields on the receipt; a missing field is a failed receipt, not a silent
default.

### 1.1 Model

| Field | Frozen value |
| --- | --- |
| Hugging Face id | `BAAI/bge-small-en-v1.5` |
| Runner alias | `--embedder sbert-bge` |
| Immutable revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Revision source | Hugging Face Hub `sha` for this model id, `lastModified` 2024-02-22T03:36:23.000Z, looked up 2026-11-12. Not selected by pilot scores. |

This choice follows ADR 0002 and the existing benchmark embedder factory
(`SentenceTransformerEmbedder.make_bge_small`, runner alias `sbert-bge`). It is
**not** the configurable shared-history MiniLM default
(`sentence-transformers/all-MiniLM-L6-v2` / `--embedder sbert-mini` /
`--embedder sentence-transformers`). That default is not an implicit model
choice for this pilot.

A model **name** is not sufficient provenance. Execution must construct the
sentence-transformers backend with this revision (or an equivalent verified
snapshot digest of the loaded weights) and write the revision on the receipt.
If the installed runner still calls `SentenceTransformer(model_name, device="cpu")`
without `revision=`, that is a **pre-execution blocker**, not a licence to
resolve `main` at download time.

### 1.2 Preprocessing and device

Frozen to the existing `SentenceTransformerEmbedder` path used by
`benchmarks.realistic.shared_history.build_shared_history_embedder` when the
semantic extra is present:

- Device: `"cpu"` (explicit constructor argument). No CUDA, MPS, or remote
  accelerator.
- Encode call: `SentenceTransformer.encode(text)` / `encode(texts)` with **no**
  extra kwargs from this library (no `normalize_embeddings=`, no
  `convert_to_numpy=` override, no prompt / instruction argument).
- No BGE retrieval instruction prefix is added. Query and memory strings are
  the dataset `content` / `query` fields as stored.
- Hub modules at the frozen revision (`modules.json`) are Transformer →
  Pooling (`1_Pooling`) → Normalize (`2_Normalize`). Whatever those saved
  modules do is the preprocessing; this protocol does not add a second
  normalization step.
- Re-encode every memory when the model or revision changes. Never reuse
  hash-vector snapshots or embeddings from another model.

Observed embedding dimension is recorded at execution. The library documents
384 dimensions for this model id; the receipt must store the **observed**
length of a produced vector, not that documentation line.

### 1.3 CPU / execution host

- GitHub-hosted `ubuntu-latest` CPU, same class as
  `.github/workflows/ci.yml` job `extra-tests`.
- Optional extra: `sentence-transformers` (lazy import; clear `ImportError` if
  missing; **no** silent fallback to `TokenHashEmbedder`).
- Dedicated single-threaded process (scoped clock patches in
  `controlled_runtime`).
- No paid Hugging Face hardware. No Spaces. No private Space names in public
  text.

### 1.4 Dataset

Unchanged development set. Do not add, drop, rewrite, or reorder scenarios or
queries.

| Field | Frozen value |
| --- | --- |
| Path | `benchmarks/datasets/shared_history_recall_v1.json` |
| Name / version | `shared_history_recall_v1` / `1.0` |
| File SHA256 | `f748e37f4d7184e8979824873159738eedab661eb11c969bb7b1d42d0a617858` |
| Parsed `dataset_sha256` | `6d13f55ed51c70e6cdc4bd91ad214d497cafb057fc2492d53f2b3725317ef949` (sort-keyed JSON of the loaded Pydantic dump; already recorded on the hash timeline receipt at `ce2e6579`) |
| Scenarios / queries | 12 / 36 |
| Protocol mode | `shared_history_timeline` (`interleave: true`, `event_order: dataset_timeline`) |
| `default_top_k` | 2 |
| `query_affect_gate_tau` | 0.2 |

The two digests are different by construction (file bytes vs parsed object).
Both belong on the receipt. If either changes, stop; this protocol no longer
applies.

Declared affect counts already in the dataset file (not new measurements):
14 / 36 queries have `|valence| < 0.2`; 4 sit on the boundary
`|valence| = 0.2`. Boundary query ids, verified from the frozen file:

- `grant_review_reversal` / `gr_q_revised_aims`
- `orchestra_chair_challenge` / `or_q_callback`
- `ferry_strike_homecoming` / `fs_q_union_deal`
- `immigration_paperwork_scare` / `im_q_biometrics`

Boundary queries are **non-neutral** under the strict `< tau` rule
(`engine_shared.is_query_affect_neutral`). Identify them explicitly in the
report; do not drop them.

### 1.5 Runner revision

| Field | Frozen value |
| --- | --- |
| Implementation floor | PR #186 merge commit `b209397816936e90dec6dbe435f145866ab03e84` must be an ancestor of the execution `HEAD` |
| Execution `git_revision` | `git rev-parse HEAD` at run time (receipt field) |
| This protocol | blob SHA256 of this file at the commit that merged it, recorded on the receipt |

Non-embedder settings stay matched to the published hash timeline receipt
(`benchmarks/realistic/receipts/shared_history_recall_v1/ce2e6579/`, arms 3–4):
same dataset path, event order, IDs, clock policy, empty checkpoint, arms, and
engine config defaults. Only the embedder class/model/revision/device change.

### 1.6 Seeds, thresholds, top-k, bootstrap

| Field | Frozen value | Role |
| --- | --- | --- |
| `seed` | 7 | UUID5 memory ids and bootstrap resampling (same as hash receipts) |
| `n_bootstrap` | 2000 | Percentile 95% CI (`benchmarks.common.statistics.DEFAULT_N_BOOTSTRAP`; hash timeline markdown already reports `n=2000, seed=7`) |
| `top_k` | 2 | Dataset `default_top_k`; do not pass `--top-k` |
| Gate τ | 0.2 | `EmotionalMemoryConfig.query_affect_gate_tau`; near-neutral iff `abs(valence) < τ` |
| Confidence | 0.95 | Bootstrap percentile |
| `read_only_retrieves` | written true/false | Always present on the protocol JSON |

Clock: epoch `2026-01-01T00:00:00+00:00`, one second per operation (including
session boundaries), UUID5 from `(seed, scenario_id, index)`.

### 1.7 Receipt settings

Follow the existing shared-history receipt policy
(`benchmarks/realistic/README.md`, ADR 0002 criterion 5,
`.github/workflows/shared-history-receipts.yml` as the hash-control template).

Every semantic-pilot receipt **must** include:

- runner commit (`git_revision`) and reproducing CI run URL
- dataset path, file SHA256, parsed `dataset_sha256`
- model id **and** immutable revision (or verified snapshot digest)
- observed dimension
- preprocessing summary (encode kwargs, instruction prefix used/not used,
  Hub modules)
- normalization (Hub `2_Normalize` present/absent as loaded)
- device (`cpu`)
- dependency versions (`sentence-transformers`, `transformers`, `torch`,
  `numpy`, `pydantic`, Python, platform)
- seed, top-k, gate threshold, bootstrap settings
- `read_only_retrieves` true/false
- commands actually run
- result files and checksums (`SHA256SUMS`)
- this protocol path and its blob digest

Archive under
`benchmarks/realistic/receipts/shared_history_recall_v1/<execution-sha>/`
without replacing the hash receipt at `ce2e6579/`. Large `arm*.json` reports
may be omitted for size, as in the hash receipt README, if `SHA256SUMS` still
lists them. Label every published semantic-pilot number
«exploratory development-set rerun (semantic embedder), not confirmation».

Hash-embedder scores already on file remain
«mechanism test (hash embedder), not a semantic-quality benchmark».

---

## 2. Injection, reset, dimension coverage, and CI

Issue #193 box 2. These checks are plumbing. None of them is a retrieval-quality
result. They must pass **before** any scored or smoke execution of this protocol.

### 2.1 Required behaviour

1. Embedder injection through the CLI (`--embedder` / `--embedder-model`),
   `build_shared_history_embedder`, `SharedHistoryAdapter.__init__`, and
   `SharedHistoryAdapter.reset` (fresh `EmotionalMemory(InMemoryStore(), self.embedder, …)`).
2. Fresh stores per arm and per scenario. Arms do not share config objects or
   memory dicts (`test_shared_arms_do_not_share_store_config_or_tags`,
   `test_shared_history_resets_each_scenario`).
3. Identical model configuration for cosine, gated, and always-on.
4. Re-encode all memories on this model/revision; never reuse hash vectors.
5. CPU execution; lazy optional import; `ImportError` if the extra is missing
   (`test_semantic_embedder_missing_extra_error`). No silent hash fallback.
6. Deterministic synthetic coverage:
   - non-64-dimensional vectors through adapter reset / snapshot round trip
   - mixed-dimension rejection (`emotional_memory._math.cosine_similarity`
     raises `ValueError` on length mismatch; `tests/test_math.py::test_mismatched_dims_raises`;
     SQLite store mismatch tests remain a backend control, not the pilot store)
   - snapshot restoration and read-only restoration
     (`test_shared_history_read_only_retrieves_do_not_persist_writebacks`;
     adapter raises `RuntimeError` if a read-only retrieve mutates store or state)
   - same-checkpoint near-neutral ranking equivalence, including after
     divergent tag-only write-back histories
     (`tests/test_engine.py::TestNearNeutralGatedEqualsCosineAfterDivergentWriteback`)
   - unsalted hash behaviour preserved
     (`test_token_hash_embedder_empty_salt_matches_unsalted`; empty salt list)
7. A separately identified real-model integration test. The
   `sentence_transformers`-marked test in `tests/test_shared_history.py` is
   plumbing/metadata (it may mock `encode`). Synthetic vector tests establish
   plumbing only. Neither is a retrieval-quality result.
8. Reproducing CI: GitHub-hosted CPU job that installs
   `[sentence-transformers]` and runs the marked integration test (existing
   `extra-tests` job). The semantic **pilot receipt** needs its own CPU CI run
   URL at execution time; the hash workflow
   `.github/workflows/shared-history-receipts.yml` stays the hash control and
   is not this pilot.

### 2.2 Store caveat (from ADR 0002)

Near-neutral gated vs cosine ranking equality is a wiring property of
**tag-only** write-back on `InMemoryStore` with a deterministic embedder.
`SQLiteStore.update` re-inserts the vector row; tie order is not shown there.
Re-verify if write-back starts changing content, embeddings, or membership, if
the embedder becomes non-deterministic, or if a store other than
`InMemoryStore` is used. This pilot uses `InMemoryStore` only.

---

## 3. Pre-registered comparisons

Issue #193 box 3.

### 3.1 Arms

Unchanged names and routing (`benchmarks.realistic.shared_history.ARMS`):

| Arm | Ranking |
| --- | --- |
| `cosine` | `precomputed_weights=semantic_only_weights()` (`[1,0,0,0,0,0]`) |
| `gated` | cosine when `is_query_affect_neutral(valence, τ)`; otherwise full query-affect scoring |
| `always_on` | full query-affect scoring on every query |

Matched across arms: event order, memory IDs, top-k, τ, seed, clock, dataset,
model, revision, device, engine config, write-back **policy per condition**.
Cosine uses the same engine and write-back policy; only ranking weights differ.

### 3.2 Write-back conditions

| Condition | Flag | Role |
| --- | --- | --- |
| Normal write-backs | `read_only_retrieves=false` | **Primary** pilot condition |
| Read-only retrieves | `--read-only-retrieves` | Mechanism diagnostic only |

Read-only restores a deep-copied snapshot after each retrieve. It is **not** an
additional independent sample. Do not pool the two conditions into N.

Primary `b`, `c`, and ICC are reported for the write-back condition, and also
with/without carry-over-dependent pairs as a sensitivity analysis (§4.3).
Read-only figures are diagnostic.

### 3.3 Affect strata

Using recorded query valence and frozen τ = 0.2:

| Stratum | Rule | Queries in this file |
| --- | --- | --- |
| Near-neutral | `abs(v) < 0.2` | 14 |
| Non-neutral | `abs(v) >= 0.2` | 22 (includes 4 boundary) |

Near-neutral gated-versus-cosine discordance is identically zero **by
mechanism** under tag-only write-back on `InMemoryStore` with a deterministic
embedder (ADR 0002). That cell is a wiring check, not evidence of benefit.
Its ICC is not estimable (§5).

### 3.4 Commands (not executed here)

Primary (write-back):

```bash
uv run python -m benchmarks.realistic.runner \
  --shared-history --embedder sbert-bge \
  --embedder-model BAAI/bge-small-en-v1.5 \
  --seed 7 \
  --dataset benchmarks/datasets/shared_history_recall_v1.json
```

Diagnostic (read-only): same command plus `--read-only-retrieves`.

Do not omit `--embedder sbert-bge` in a way that would load MiniLM. Do not pass
`--hash-salt` / `--hash-salts` on the semantic run. Preserve the unsalted hash
runner as a separate mechanism/regression control; do not mix those outputs
into this pilot’s N.

---

## 4. Discordance D, signed effect Z, ranking disagreement

Issue #193 box 4.

For each method ∈ {`gated`, `always_on`} versus `cosine`, each write-back
condition, and each affect stratum, align on query identity
`(scenario_id, session_id, query_id)`.

Let `top1_correct_*` be 1 iff the rank-1 retrieved memory id is in the query’s
`expected_memory_ids`, else 0.

### 4.1 Definitions

**Correctness discordance (incidence):**

```text
D_i = 1[top1_correct_method != top1_correct_cosine]
```

**Signed paired correctness difference:**

```text
Z_i = top1_correct_method - top1_correct_cosine
```

`Z_i ∈ {-1, 0, +1}`. Correlation of `D_i` does not specify the variance of
`Z_i`.

**Ranking disagreement** (diagnostic only): the ordered retrieved-id lists
differ. Two wrong IDs can disagree in rank without contributing to `b` or `c`.
Ranking disagreement is **not** `D_i` and **not** `Z_i`. Inspect ties and
candidate sets. Report zero discordance as an observation, not as equivalence.

### 4.2 Paired cells

Retain aligned query IDs and top-1 correctness. For each comparison ×
condition × stratum:

| Cell | Meaning |
| --- | --- |
| both-correct | method 1, cosine 1 (`Z_i = 0`, `D_i = 0`) |
| both-wrong | method 0, cosine 0 (`Z_i = 0`, `D_i = 0`) |
| method-only-correct (`b`) | method 1, cosine 0 (`Z_i = +1`, `D_i = 1`) |
| cosine-only-correct (`c`) | method 0, cosine 1 (`Z_i = −1`, `D_i = 1`) |

Also report `b+c`, `(b+c)/N_cell`, and `(b-c)/N_cell`, where `N_cell` is the
query count **in that cell** (stratum × condition), not a pooled N.

Any query-level exact paired test (for example McNemar on `b` vs `c`) is
**descriptive** and must disclose the independence assumption it violates
(queries share a history). It does not gate v2.

### 4.3 Carry-over labels (frozen from ADR 0002)

Hash-embedder observation (not a semantic result): gated-versus-cosine top-1
discordant pairs on this dataset at commit
`441207d13f83b3dbefe4f17039ff68707072085f` differed by write-back mode.
Write-back: `b=1`, `c=0` on bakery `bk_q_inspection_not_gasket`.
Read-only: `b=1`, `c=0` on ferry `fs_q_no_sailing`. Both are query 3 of 3.
A follow-up tag-field ablation attributed both to carry-over (reconsolidated
`core_affect` for the ferry; `core_affect` plus `retrieval_count` for the
bakery).

Apply the same labels to **semantic** discordant pairs:

| Label | Rule |
| --- | --- |
| `carry-over-dependent` | Discordant under write-back only |
| `carry-over-suppressed` | Discordant under read-only only |
| `both` | Discordant under both conditions |

This is a sensitivity analysis, not an exclusion. Report primary `b`, `c`, and
ICC both with and without carry-over-dependent pairs. Dropping only
write-back-only pairs would bias toward read-only.

### 4.4 Scenario table (required)

Publish one table per method-versus-cosine comparison, with a row per
scenario, including:

- scenario id
- query count (always 3 in this file)
- near-neutral count and non-neutral count
- `b`, `c`, discordance rate (`mean D_i`), signed effect (`mean Z_i`)
- a column marking each discordant pair as write-back only, read-only only, or
  both, **per arm and stratum**, listing discordant query IDs per mode

Do not omit scenarios with zero discordance.

---

## 5. Observed-scale ICC estimator for `D_i`

Issue #193 box 5. Predeclared **before** execution. Exploratory. Not a
population ICC for v2.

### 5.1 Estimator

For each comparison × write-back condition × affect stratum, fit a one-way
random-effects ANOVA on the observed 0/1 `D_i` clustered by `scenario_id`
(Shrout & Fleiss, 1979, ICC(1); unbalanced form with average cluster size
`n0`).

Let `j = 1…k` index scenarios that have at least one query in the cell,
`n_j` the query count in that cell, `N = Σ n_j`, `Y_{ij} = D_i` in scenario
`j`, `Ȳ_j` the scenario mean, `Ȳ` the grand mean:

```text
SSB = Σ_j n_j (Ȳ_j − Ȳ)²
SSW = Σ_j Σ_i (Y_{ij} − Ȳ_j)²
MSB = SSB / (k − 1)
MSW = SSW / (N − k)
n0  = (1 / (k − 1)) * (N − (Σ_j n_j²) / N)
ICC = (MSB − MSW) / (MSB + (n0 − 1) * MSW)
```

Use only numpy (this repository’s statistics stack has no scipy dependency).
Do not treat this ICC as the variance of `Z_i`. Optionally report the same
estimator on `Z_i` as a **diagnostic**; v2 planning uses the joint paired
outcomes of `Z`, not ICC(`D`) alone (§8).

### 5.2 Uncertainty

Scenario-clustered nonparametric bootstrap: resample `k` scenarios with
replacement, recompute ICC on the queries those scenarios contribute, 2000
replicates, seed 7, percentile 95% interval. This is not a query-level
bootstrap and is not an independent-N claim.

If a replicate is degenerate under §5.3, drop it from the interval and report
how many replicates were informative. If fewer than 100 informative
replicates remain, report the interval as **not estimable** with that reason.

### 5.3 Sparse, unequal, and degenerate handling

Report **not estimable** with the reason; **never substitute zero**.

| Condition | Action |
| --- | --- |
| Outcomes constant (`SSB = 0` and `SSW = 0`) | Not estimable (no variation) |
| Near-neutral gated vs cosine | Not estimable (discordance identically zero by mechanism) |
| `k < 2` | Not estimable |
| Fewer than two scenarios with `n_j ≥ 2` | Not estimable (no within-scenario information) |
| `N = k` (all `n_j = 1`) | Not estimable (`MSW` undefined / no repeats) |
| Fit unidentified (`N − k = 0`, or `MSB + (n0−1)MSW = 0`) | Not estimable |
| Point estimate outside `[−1, 1]` from sampling | Not estimable (boundary/degenerate) |

Unequal cluster sizes are handled by `n0`; do not drop small clusters to force
balance. Do not pool arms, strata, or write-back conditions to manufacture
replication or to make an ICC estimable.

Publish the number of informative clusters (scenarios with `n_j ≥ 2` in that
cell) beside any estimate.

---

## 6. Twelve constructed histories

Issue #193 box 6.

This file has 12 author-constructed scenarios, 3 queries each (verified:
every scenario contributes 3 queries). Queries inside a history are
dependent. Twelve histories may permit a **descriptive** ICC; they do not
justify a precise population ICC for v2.

Stratum-specific cluster sizes in this file (not measured outcomes):

| Stratum | Cluster-size histogram | Note |
| --- | --- | --- |
| Near-neutral | 10 scenarios with 1 query, 2 with 2 | Most near-neutral clusters carry no within-scenario information |
| Non-neutral | 10 with 2, 2 with 1 | Still 1–2 queries per history, not 3 |

Do not pool:

- cosine / gated / always-on
- near-neutral / non-neutral
- write-back / read-only
- hash salts or hash vs semantic embedders

into an independent N. Additional queries in an existing history are not
additional independent histories.

---

## 7. Rho sensitivity cases (illustrative only)

Issue #193 box 7.

For equal cluster size `m` and exchangeable observed-scale correlation `rho`,
the usual mean-variance design effect is

```text
DE = 1 + (m − 1) * rho
```

([Campbell et al., 2012, CONSORT cluster extension](https://doi.org/10.1136/bmj.e5661)).

With three queries per history, `DE = 1 + 2 * rho`. The ADR example `rho = 0.25`
gives `DE = 1.5` and `36 / 1.5 = 24`. **These are illustrative assumptions, not
measured pilot results and not a validated power calculation for paired
correctness.**

Declared sensitivity cases (not confidence bounds, not estimated ICCs):

| `rho` | `DE = 1 + 2 * rho` (m = 3) | `36 / DE` (illustrative) |
| ---: | ---: | ---: |
| 0 | 1.0 | 36 |
| 0.10 | 1.20 | 30 |
| 0.25 | 1.50 | 24 |
| 0.50 | 2.00 | 18 |
| 1.0 | 3.00 | 12 |

Stratum-specific `m` can differ from 3 (this file: 1–2). Repeating the same
arithmetic with `m = 2` is also illustrative (`DE = 1 + rho`) and is not a
second power analysis. Do not label either table as required N, N_eff for a
confirmatory study, or a sample-size promise.

---

## 8. Rules for a later v2 plan

Issue #193 box 8. This section freezes **planning rules**. It does not size v2
and does not choose N from this unrun pilot.

After the pilot receipt is archived (§9), a **separate** v2 pre-registration
must exist before anyone authors or scores a confirmatory set. That later
document, not this one, will fix:

- target population
- independent scenario sampling
- primary contrast and stratum
- smallest useful effect
- alpha, power, multiplicity policy
- stopping rule and resource limit

Sample-size work for that document must:

- use **semantic** paired outcomes from this pilot, not hash discordance
- model the **joint** paired outcomes and the scenario dependence of **`Z`**,
  not inflate a query-level exact test using ICC(`D`) alone
- treat discordance incidence as insufficient by itself
- report required **independent histories** and total queries, with
  cluster-size assumptions
- show sensitivity to uncertainty in both discordance and effect size
- not pick the sizing assumption from whichever pilot ICC gives the smallest N
- keep these 12 pilot histories **out of** the confirmatory set
- freeze model id, revision, and configuration before confirmation

Additional queries stuffed into an existing history do not create additional
independent histories.

---

## 9. Archive, exclusion, and stop branch

Issue #193 box 9.

Order of operations:

1. Merge this protocol (no execution in the same change).
2. Satisfy §2 plumbing and §1 revision pinning.
3. Run the two conditions in §3 on CPU CI; write receipts (§1.7).
4. **Archive** those receipts and outcomes in-tree under the existing policy.
5. Only then write a **separate** v2 pre-registration. Do not start v2
   authoring or scoring before step 4.

If the pilot is too sparse, the ICC/`Z` structure is not estimable in the
primary cell, the projected budget is infeasible, or no useful effect is
supported, publish an inconclusive or negative pilot and either **stop** or
pre-register additional **development** data. Do not:

- assume future discordant pairs will favor gated
- switch models after seeing scores without labelling the new analysis
  exploratory
- reduce confirmatory N after looking at confirmatory outcomes
- recycle these 12 histories as confirmation by changing only the embedder

ADR 0001’s human-grounded MSC study remains a separate evidence track and is
not replaced by this pilot.

---

## 10. Issue #193 checklist mapping

| #193 box | Covered in |
| --- | --- |
| Freeze immutable semantic model revision, preprocessing, CPU configuration, dataset digest, runner revision, seeds, thresholds, top-k and receipt settings | §1 |
| Verify injection/reset, fresh stores per arm/scenario, non-64D vectors, mixed-dimension rejection, snapshot restoration and same-checkpoint neutral equivalence; real-model integration coverage and reproducing CI | §2 |
| Preregister cosine/gated/always-on under normal write-back (primary) and read-only (diagnostic), matched event order/settings, prespecified affect strata | §3 |
| Define D and Z separately from ranking disagreement; publish per-scenario query/stratum counts, b, c, discordance rates and signed effects | §4 |
| Predeclare observed-scale ICC estimator for D, uncertainty method, sparse/unequal/degenerate handling; report exploratory ICC when estimable, else not estimable with reasons, never zero | §5 |
| Report informative scenario counts and the limits of 12 constructed histories; do not pool arms, strata, modes or salts into independent N | §6 |
| Publish rho sensitivity cases 0, 0.10, 0.25, 0.50, 1.0; label DE=1+(m−1)rho and the m=3 / rho=0.25 example as illustrative, not measured power | §7 |
| Base later v2 planning on joint paired outcomes and scenario dependence of Z, not ICC(D) alone; report independent scenario counts and total queries | §8 |
| Archive receipts and outcomes before separate v2 preregistration; exclude pilot histories from confirmation; retain an inconclusive/stop branch | §9 |

---

## References (estimators and design-effect illustration only)

- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
  assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428.
- Campbell, M. K., Piaggio, G., Elbourne, D. R., & Altman, D. G. (2012).
  Consort 2010 statement: extension to cluster randomised trials.
  *BMJ*, 345, e5661. https://doi.org/10.1136/bmj.e5661
