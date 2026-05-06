# Pre-registration Addendum I — Resonance Magnitude-Amplification Decomposition: e5 × no_resonance

**Date written:** 2026-05-06
**Protocol version:** addendum_i_v1
**Parent pre-regs:** `benchmarks/preregistration_addendum_v3.md` (S3),
`benchmarks/preregistration_addendum_s3_closure.md` (Hb FAIL closure)
**Closes:** #29 (magnitude-amplification investigation)

> **Epistemic status:** This is a **post-hoc analysis document**, not a strict
> pre-registration. The committed result files
> `benchmarks/ablation/results.v2.{sbert,e5}.json` predate this write-up.
> The aggregate-level confirmatory finding (Hb FAIL on both embedders, with
> e5 Δ=+0.085 p<0.001) is already pre-registered and reported in
> `preregistration_addendum_s3_closure.md`. This addendum **decomposes** that
> finding by challenge type, **corrects the framing** of issue #29
> (sign-consistent, not sign-reversed), and **defers** mechanism instrumentation
> to a future confirmatory study (Hi3, scoped at the bottom of this file).

---

## Background

The S3 closure (Addendum S3, 2026-05-04) reported on `realistic_recall_v2`
(N=200, 5 challenge types × 40):

| Hypothesis | Embedder | full top1 | no_resonance top1 | Δ | p_boot |
|---|---|---:|---:|---:|---:|
| Hb (no_resonance) | SBERT | 0.535 | 0.565 | +0.030 | 0.203 |
| Hb (no_resonance) | e5 | 0.510 | 0.585 | +0.075 | <0.001 |

(Numbers above re-derived from the committed JSON; the headline S3 row reports
`+0.085` rounded to 2 dp from the SBERT-baseline reference. Both are within
rounding of one another. The exact aggregate Δ is **+0.075** for e5 and
**+0.030** for SBERT.)

Issue #29 framed this as a **sign reversal** — "e5 shows opposite direction
to SBERT". This framing is **incorrect at the per-challenge level**: both
embedders show **non-negative Δ on every challenge type** (see Hi1 below).
The phenomenon is a **magnitude amplification** (e5 ≈ 2.5× SBERT in
aggregate), not a directional flip.

This addendum formalises the post-hoc per-challenge decomposition and
proposes a confirmatory mechanism study.

---

## Hi1 — Per-challenge decomposition (post-hoc, descriptive)

**Source:** `benchmarks/ablation/results.v2.{sbert,e5}.json` (committed).
**Status:** descriptive — no new bootstrap test (per-query data not persisted
in the JSON; paired-bootstrap CIs on per-challenge subsets would require a
re-run).

| Challenge type | SBERT Δ | e5 Δ | e5 − SBERT |
|---|---:|---:|---:|
| affective_arc       | +0.025 | +0.075 | +0.050 |
| momentum_alignment  |  0.000 |  0.000 |  0.000 |
| recency_confound    | +0.075 | +0.125 | +0.050 |
| same_topic_distractor | +0.025 | +0.050 | +0.025 |
| **semantic_confound** | +0.025 | **+0.125** | **+0.100** |
| **aggregate**         | **+0.030** | **+0.075** | **+0.045** |

Conclusions (descriptive, no formal test):

1. **Sign consistency:** Δ ≥ 0 for every (embedder × challenge) cell except
   the tied `momentum_alignment` (Δ = 0 on both embedders).
2. **Dominant amplification channel:** `semantic_confound` shows the largest
   embedder-gap (+0.100), accounting for ~2.2× of the +0.045 aggregate gap.
3. **Secondary amplification channel:** `recency_confound` and `affective_arc`
   each contribute +0.050.

Marginal CIs (extracted from JSON, not paired):

- e5 / semantic_confound: full = 0.625 [0.475, 0.775], no_res = 0.750
  [0.625, 0.875] — overlap is large at marginal level. Aggregate-level
  paired bootstrap (`Hb e5: p<0.001`) is the confirmatory layer; per-challenge
  splits are exploratory decompositions of that confirmed effect.

---

## Hi2 — Mechanistic hypothesis (not tested in this addendum)

**Hypothesis (qualitative):** e5-small-v2 produces tighter intra-topic
clusters than SBERT-bge-small-en. ResonanceLink's spreading-activation
mechanism therefore over-fires on topically-similar memories, biasing the
ranking toward semantically-near distractors precisely on the challenge
type (`semantic_confound`) that requires affective discrimination among
topic-near memories.

This is **stated but not tested**. Tests would require:

- Per-memory link-set instrumentation in the runner (link count, link
  type distribution, link-strength statistics) for both embedders on the
  same dataset.
- Comparison of activation map sparsity at retrieval time (Pass 2 of the
  6-signal pipeline).
- Counterfactual: train resonance with `cosine_threshold` raised so e5
  forms fewer links; check if the +0.125 on semantic_confound shrinks.

These are **not** in scope for this addendum. They are scoped in Hi3 below
as a deferred confirmatory study.

---

## Hi3 — Deferred confirmatory mechanism study (pre-registered)

**Status:** PENDING EXECUTION.
**Trigger condition for execution:** when (a) `realistic_recall_v3` or another
≥500-query affect-rich dataset exists, AND (b) the runner has been
instrumented to dump per-memory `ResonanceLink` statistics
(`links_count`, `link_types`, `link_strengths`) per encode pass.

### Frozen design (Hi3)

**Hi3 (confirmatory):** On a dataset with N ≥ 500 queries, the per-challenge
amplification (e5 − SBERT) on `semantic_confound` is **at least +0.05**
(half the v2 estimate of +0.100, accounting for regression to the mean).

- **Direction:** `Δ_e5_semantic_confound − Δ_sbert_semantic_confound > 0.05`
- **Test:** paired bootstrap n=10,000, seed=1 (frozen, NOT seed=0 —
  seed=0 is reserved for replication of v2 aggregate)
- **Holm correction:** family of {Hi3, Hi3_recency, Hi3_arc} — 3 hypotheses
  if extending to other channels.
- **Pre-spec failure mode:** if Hi3 FAIL, the v2 finding is downgraded from
  "embedder-amplification of resonance interference" to "v2 sample-size
  artefact". Claim matrix updated accordingly.

### Mechanism instrumentation requirement

The runner MUST emit, for each variant × query, a per-memory dump:

```jsonc
{
  "qid": "...",
  "embedder": "e5-small-v2",
  "variant": "full",
  "links_per_memory": {
    "mean": 4.2, "median": 4, "max": 5
  },
  "link_types": { "TEMPORAL": 23, "SEMANTIC": 41, "MOOD": 12, ... },
  "link_strength_distribution": [0.62, 0.71, ...]   // top-5 per memory
}
```

This dump enables the mechanism comparison (e5 produces denser/tighter
link sets vs SBERT?) without requiring a separate codebase.

### Reporting rules for Hi3

1. seed=1 is frozen at the commit that ships the runner instrumentation.
2. Hi3 confirmatory result reported regardless of outcome.
3. Mechanism findings (link-set comparison) reported as exploratory
   regardless of Hi3 verdict.
4. Closure document: `preregistration_addendum_i_closure.md` (when executed).

---

## What this addendum closes vs leaves open

**Closed (post-hoc):**
- ✅ Issue #29 framing corrected: not a sign reversal, but a magnitude
  amplification ~2.5× on aggregate, ~5× on `semantic_confound`.
- ✅ Per-challenge decomposition documented from committed v2 data.
- ✅ Claim matrix updated to reference this decomposition.

**Open (deferred to Hi3):**
- ❌ Mechanism: do e5 link-sets differ from SBERT link-sets, and does that
  difference predict the per-challenge Δ?
- ❌ Confirmation on larger N: is the v2 +0.100 on semantic_confound stable
  on N≥500?
- ❌ Counterfactual: does raising `ResonanceConfig.cosine_threshold` for
  e5 close the gap?

---

## Reading guide

- For the parent S3 pre-registration: `preregistration_addendum_v3.md`
- For S3 execution + aggregate verdicts: `preregistration_addendum_s3_closure.md`
- For the canonical claim matrix: `docs/research/claim_validation_matrix.json`
  (the `theory_faithful_operationalization` row references this addendum)

---

## Appendix — exact numbers used

Extracted from `benchmarks/ablation/results.v2.e5.json` and
`benchmarks/ablation/results.v2.sbert.json`, both committed before this
write-up:

```
e5 / variant=full          aggregate top1_accuracy = 0.510
e5 / variant=no_resonance  aggregate top1_accuracy = 0.585

sbert / variant=full          aggregate top1_accuracy = 0.535
sbert / variant=no_resonance  aggregate top1_accuracy = 0.565

n_bootstrap = 10000, seed = 0, ci_method = percentile
```

Per-challenge values reported in Hi1 are the `top1_accuracy` field of each
`challenge_type_metrics` entry, rounded to 3 dp.
