# Pre-Registration Addendum G — Dual-Path Ablation with LLM Appraisal

**Status:** PENDING EXECUTION
**Date written:** 2026-05-05
**Protocol version:** addendum_g_v1
**Parent pre-regs:** `benchmarks/preregistration_addendum_v2.md` (Add. A),
`benchmarks/preregistration_addendum_v3.md` (Add. D, Hd1/Hd2),
`benchmarks/preregistration_addendum_f.md` (Hf1)

> **Epistemic status:** This file establishes the protocol for Addendum G.
> No results have been collected. This document must not be modified after
> execution begins. The hypothesis, metric, test, and dataset requirements
> are frozen at this commit.

---

## Motivation

Three findings motivate this study:

1. **Oracle-affect circularity (Add. A, Add. D):** The Hd1/Hd2 studies show
   AFT outperforms naive cosine when affect is *preset* by the author. This
   does not test AFT under real inference conditions, where affect must be
   inferred by an appraisal engine.

2. **Keyword appraisal is destructive (Ha2/Hb2 FAIL):** `KeywordAppraisalEngine`
   reduces top1 by Δ=−0.39 vs preset-affect AFT. A non-destructive appraisal
   signal is required to demonstrate a full pipeline advantage.

3. **Dual-path advantage over synchronous appraisal (Hf1 PASS):** Deferring
   appraisal to the slow path (`elaborate()`) partially mitigates synchronous
   keyword override. The question is whether dual-path encoding with a *good*
   appraisal engine (LLM) provides a net positive over no-appraisal on
   scenarios without oracle affect injection.

**Central question:** Does AFT with `LLMAppraisalEngine` in dual-path mode
outperform `naive_cosine` on a benchmark where affect values are NOT preset
by the author?

---

## Dataset requirement (pre-specified)

This study requires a dataset that satisfies all of:

1. **No preset affect injection:** scenario memories must NOT include
   `valence`/`arousal` fields pre-filled by the author. The only affect
   signal available to AFT must come from the appraisal engine.
2. **Affective challenge types:** scenarios must include at least one
   challenge type where emotional context discriminates the target from
   distractors (otherwise the study is underpowered against a no-affect null).
3. **N ≥ 100 queries** for ≥80% power to detect Δ = 0.10 at α = 0.05.
4. **Author-blind to system output during construction:** the dataset must be
   finalized and committed before any benchmark run begins.

The existing `realistic_recall_v*.json` datasets do NOT satisfy criterion 1
(they include preset `valence`/`arousal` fields). A new dataset
`realistic_recall_v3_noAF.json` (AF = affect-free) must be constructed.

---

## Systems

- **`aft_llm_dual`**: `EmotionalMemory` with `LLMAppraisalEngine`
  (`dual_path_encoding=True`), `elaborate()` called after each encode.
  Affect is inferred by the LLM, not preset.
- **`naive_cosine`**: pure cosine retrieval, no affective state. Same as
  prior studies.

No `aft_noAppraisal` condition in this study — that is the oracle baseline
already tested in Add. A/D. The control is naive cosine (no affect at all).

---

## Primary hypothesis (Hg1)

**Hg1 (confirmatory, one-tailed):**
`aft_llm_dual.top1_accuracy > naive_cosine.top1_accuracy`
on `realistic_recall_v3_noAF` (N ≥ 100 queries).

- Direction: `aft_llm_dual_top1 − naive_cosine_top1 > 0`
- Metric: top1_accuracy (proportion of queries where rank-1 = target)
- Test: paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05
- Minimum effect threshold: Δ > 0.05 (5 pp) for practical significance
- Holm correction: Hg1 is the sole confirmatory hypothesis; no family correction required

### Exploratory hypotheses (Hg2, Hg3)

**Hg2 (exploratory):** `aft_llm_dual.top1 > aft_noAppraisal.top1`
(where `aft_noAppraisal` uses neutral CoreAffect(0.0, 0.5, 0.5) — NOT oracle
preset values). Tests whether LLM appraisal provides positive signal vs. no
affect signal at all.

**Hg3 (exploratory):** `aft_llm_dual.top1 > aft_llm_sync.top1`
where `aft_llm_sync` runs LLM appraisal synchronously at encode time
(not deferred). Tests whether the dual-path temporal architecture provides
an advantage over synchronous LLM appraisal.

Exploratory hypotheses are analyzed post-hoc and cannot be promoted to
confirmatory claims.

---

## Statistical analysis plan

- Paired bootstrap: n=10,000, seed=0, percentile method, one-tailed p.
- Report: mean top1, 95% CI, Δ (aft_llm_dual − naive), 95% CI of Δ,
  p (bootstrap one-tailed), Cohen's d.
- No Holm correction (single primary hypothesis).
- Power note: at N=100, to detect Δ=0.10 with d≈0.25 (estimated from
  prior studies), approximate one-tailed power ≈ 75%. N=150 recommended
  for 85% power.

---

## Execution (to be filled at run time)

```bash
# Dataset must be committed first
make bench-addendum-g    # to be added to Makefile when dataset is ready
```

Output: `benchmarks/appraisal_confound/results.hg1.{json,md,protocol.json}`

LLM dependency: requires `EMOTIONAL_MEMORY_LLM_API_KEY` and a compatible
model. Estimated cost: ~$2–5 at gpt-5-mini pricing for N=100 dual-path
encodes (each encode triggers one LLM call via `elaborate()`).

---

## Reporting rules

Per `benchmarks/preregistration.md` §Reporting rules:

1. Hg1 is confirmatory; result is reported regardless of outcome.
2. Hg2 and Hg3 are exploratory; labeled as such.
3. seed=0 is frozen; no re-seeding after results are known.
4. If Hg1 FAIL: interpret as "LLM appraisal does not provide a net positive
   over naive cosine on N=100 affect-free scenarios at this effect size".
5. If Hg1 PASS: the oracle-affect limitation (Add. D) is partially resolved
   for the dual-path LLM configuration.

---

## Prerequisites before execution

- [ ] Construct `benchmarks/datasets/realistic_recall_v3_noAF.json`
      (N ≥ 100 queries, no preset affect fields, author-blind to system output)
- [ ] Commit dataset before any benchmark run
- [ ] Configure LLM environment (`EMOTIONAL_MEMORY_LLM_API_KEY`, model)
- [ ] Add `bench-addendum-g` target to Makefile
- [ ] Write result files to `benchmarks/appraisal_confound/results.hg1.*`
- [ ] Write closure document `benchmarks/preregistration_addendum_g_closure.md`
