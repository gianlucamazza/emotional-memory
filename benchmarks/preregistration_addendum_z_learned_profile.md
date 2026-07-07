# Pre-registration Addendum Z — Hz: learned retrieval profile (held-out learning-to-rank)

**Status:** PRE-REGISTERED (not yet executed). Committed before the harness runs a
single scored evaluation.
**Date (pre-reg):** 2026-07-07
**Mechanism:** replace the fixed retrieval weight vector
`base_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]` (`retrieval.py`) with a **linear
weight vector learned per corpus** by pairwise learning-to-rank over the 6 AFT retrieval
signals, evaluated strictly out-of-sample via **k-fold cross-fitting**.
**Parent closures / boundary being tested:**
`preregistration_addendum_q_affect_gating_closure.md` (state-based gating: safe wrapper,
no gain — the query is never appraised) ·
`preregistration_addendum_t_query_appraisal_closure.md` (Ht1 PASS, retrieve-time query
appraisal, +0.115 curated) · `..._t2a_..._closure.md` (naturalistic FAIL) ·
`..._x_madialbench_..._closure.md` (X, counter-congruent FAIL) ·
`..._x2_esmemeval_..._closure.md` (X2, affect-orthogonal FAIL) ·
`preregistration_addendum_y_query_affect_gate_closure.md` (Y, gate = partial safe wrapper).

---

## Motivation

Every third-party FAIL (K/T2A DailyDialog, X MADial-Bench, X2 ES-MemEval) was run with a
**single fixed weight vector**. Addendum J swept 10 _hand-authored_ LoCoMo weight configs and
found none closed the gap — but that is a discrete grid of author-chosen vectors, not a
weight vector **fit to the data**. No addendum has ever tested a **learned** profile, and
the theory residual named in X ("support-mode / counter-congruent retrieval profile") was
left unscheduled precisely because tuning weights _on_ MADial would be circular.

**Cross-fitting resolves that circularity.** We do not tune on the test queries: weights are
fit on k−1 folds and evaluated on the held-out fold; only pooled held-out metrics gate the
verdict. The question is not "can we fit MADial" (trivially yes) but "does a learned affect
profile **generalize** out-of-sample" — which is a legitimate, non-circular test of whether
the third-party failures are weight-misspecification (fixable) or construct-level (fundamental).

**The question.** Does a per-corpus **learned** linear combination of the 6 AFT signals,
evaluated held-out, (a) **beat cosine** on any third-party corpus where the fixed profile
failed, and (b) **preserve** the curated on-regime advantage?

**Honest expected outcome, declared ex-ante.** Most likely **Branch B**: even the held-out
optimal linear weighting will not beat cosine on X/X2, because the gold relation there is
affect-orthogonal (X2) or counter-congruent to mood-congruence in a way a _fixed-per-corpus_
linear profile cannot capture per-query (X). If so, Z **hardens** the boundary from "these
fixed weights fail" to its strongest form: _no learned linear affect profile generalizes
here_. A **Branch A** (held-out learned beats cosine on ≥1 third-party corpus) would be the
first external positive and would break the provenance bound — reported with full skepticism
(replication across fold seeds required before any strong claim).

**Design discovery (why this is cheap and exact).** The 6 signals per (query, candidate) are
exactly what `retrieval.retrieval_breakdown(...)` already computes (`RetrievalSignals`
s1..s6). Scoring a candidate under a weight vector `w` is the dot product `w · s`. So the
`aft_learned` arm needs **no new LLM calls** beyond the appraisals the `aft_query_appraised`
arm already runs (reused from the committed corpus `results.json`), and no `src/` change —
the harness computes the feature matrix once per corpus and re-ranks under learned `w`.

---

## Protocol

- **Features.** Per (query, candidate) the 6 AFT retrieval signals as produced by the AFT
  arm's scoring: s1 semantic cosine, s2 mood congruence, s3 affect proximity, s4 momentum
  alignment, s5 recency/decay, s6 resonance. Extracted via `retrieval_breakdown(...).signals`
  with the same query affect (direct-VAD, `query_affect=`) the `aft_query_appraised` arm uses.
  Features are standardised (z-scored) per corpus using **train-fold statistics only** (the
  scaler is fit on train folds and applied to the held-out fold — no test leakage).
- **Model.** A **linear** ranker `score = w · s`, `w ∈ ℝ^6`, learned by **pairwise logistic
  learning-to-rank** (RankNet-style): for each query, form (relevant, non-relevant) candidate
  pairs and minimise the logistic loss on score differences by full-batch gradient descent
  (numpy only — no new dependency). L2 regularisation `λ` fixed ex-ante (see below). Linear
  by design: it is the exact, interpretable counterfactual to the fixed weight vector, and
  keeps the model capacity at 6 parameters (negligible overfitting risk).
- **s2 sign is free.** `w` is unconstrained, so `w[1]` (mood congruence) may go **negative** —
  representing counter-congruent / support-mode recall (the X theory residual). The learned
  sign of s2 is a pre-declared interpretability readout.
- **Cross-fitting (anti-circularity core).** Per corpus, `k`-fold over queries
  (`k = 5` primary; deterministic fold assignment `query_index % k`, no shuffle — seed-free).
  For each fold: fit scaler + `w` on the other k−1 folds, predict the held-out fold, compute
  the held-out per-query metric. Pool held-out per-query metrics across folds → the
  `aft_learned` arm. **No query is ever scored by a `w` that saw it.**
- **Fixed hyperparameters (ex-ante, not tuned on test):** `k = 5`, learning rate `0.1`,
  `2000` gradient steps, L2 `λ = 1.0`, z-score standardisation. Reported as a fixed protocol;
  a `k ∈ {5, 10}` sensitivity is secondary and non-gating.
- **No tuning** of any weight, schema, prompt, or embedder on any corpus beyond the pairwise
  loss on **train folds**.

## Corpora and arms

Four corpora, each on its **own pre-registered metric** (no cross-corpus pooling). Arms:
`naive_cosine` (baseline), `aft_fixed` (the current `base_weights`), `aft_learned`
(held-out cross-fit weights — this study).

| Corpus                | Regime tested                  | Runner                                 | Metric                   | Role     |
| --------------------- | ------------------------------ | -------------------------------------- | ------------------------ | -------- |
| MADial-Bench (X)      | counter-congruent              | `benchmarks/madialbench/runner.py`     | nDCG@5                   | break    |
| ES-MemEval (X2)       | affect-orthogonal              | `benchmarks/esmemeval/runner.py`       | upstream-verbatim nDCG@4 | break    |
| DailyDialog (T2A)     | naturalistic affect-sparse     | `benchmarks/dailydialog/t2a_runner.py` | top1                     | break    |
| `realistic_recall_v2` | curated, affect-discriminative | `benchmarks/query_appraisal/runner.py` | top1                     | preserve |

## Hypotheses / quantities

- **Hz1 (break, primary).** On the third-party family {MADial-Bench, ES-MemEval, DailyDialog},
  `aft_learned` (held-out) > `naive_cosine` on the corpus metric (Δ>0, one-tailed) for **≥1**
  corpus, surviving Holm–Bonferroni across the family (m=3). PASS → the fixed-weight failure
  was (at least partly) mis-specification → provenance bound broken (Branch A).
- **Hz2 (preserve, primary).** On `realistic_recall_v2`, `aft_learned` (held-out) ≥
  `aft_fixed` − ε with ε=0.02 (non-inferiority, one-tailed): learning does not sacrifice the
  curated on-regime advantage.
- **Secondary / descriptive (non-gating — the scientific content):**
    - `aft_learned vs naive_cosine` and `aft_learned vs aft_fixed` on **all four** corpora
      (Δ, 95% CI, one-tailed p);
    - the **learned weight vector** `w̄` (mean across folds) per corpus, **especially the sign
      and magnitude of s2** on MADial-Bench (counter-congruent hypothesis: expect `w[1] < 0`);
    - the **train→test generalization gap** per corpus (in-sample vs held-out metric) — a large
      gap flags overfitting / non-generalizing profile;
    - `k ∈ {5, 10}` sensitivity of Hz1/Hz2 (secondary, non-gating).

## Statistical analysis plan (pre-declared)

- **Primary metrics:** per-query, each corpus's own (curated/DailyDialog top1;
  ES-MemEval upstream-verbatim nDCG@4; MADial-Bench nDCG@5).
- **Test:** paired bootstrap difference on held-out per-query scores, n=10,000, seed=0,
  one-tailed (`benchmarks/common/statistics.paired_bootstrap_diff`); Cohen's d on paired diffs.
- **Confirmatory family:** Hz1 = {MADial, ES-MemEval, DailyDialog} `aft_learned vs cosine`
  under **Holm–Bonferroni m=3**; Hz2 = curated non-inferiority. α=0.05. All secondary
  quantities cannot gate the verdict.
- **N per corpus** reported in the closure (queries per corpus; held-out pooled N = full N).
  Power/MDE note per corpus (observed SD + implied MDE at 80% power).

## Decision rule (pre-declared, ex-ante)

- **Hz1 PASS (≥1 third-party corpus, Holm-adjusted p<0.05, held-out Δ>0) → Branch A.** First
  external positive: the fixed-weight failure was mis-specification on that corpus. **Report
  with maximal skepticism**: before any public claim, replicate the PASS under an alternative
  deterministic fold partition (`(query_index * 7 + 3) % k`) and confirm the sign holds; report
  both. Follow-up (separate pre-registered PR, NOT this study): promote to a `src/`
  `fit_retrieval_weights()` utility applying learned `w` via the existing
  `build_retrieval_plan(..., precomputed_weights=)` / `retrieve(..., precomputed_weights=)`
  path (bypassing `adaptive_weights` — the Addendum-Q Amendment-1 trap). New claim
  `learned_retrieval_profile` = `controlled_evidence`, scoped to the corpus/corpora that PASS.
- **Hz1 FAIL → Branch B.** Boundary CONFIRMED at its strongest form: **no held-out learned
  linear affect profile beats cosine** on the third-party corpora. Upgrades `08_limitations.md`
  §2.4 from "the fixed weights fail" to "an optimally-weighted linear combination of the affect
  signals does not generalize out-of-sample here" — the provenance/construct bound is not a
  weight-tuning artifact.
- **Hz2 FAIL (curated non-inferiority violated).** Learning sacrifices the on-regime gain
  (overfitting, or the affect signals are not linearly separable even in the good regime) →
  flag as a deployability caveat; does not by itself close the line if Hz1 holds.

Marginal handling: `0.04 < p_holm < 0.05` → "PASS marginal", flagged. No post-hoc threshold
or metric switching. The learned `w` interpretability is reported as descriptive, never used
to re-derive a hypothesis after the fact.

## Circularity guardrails (declared)

- **No test leakage:** scaler + `w` fit on train folds only; held-out fold scored by an unseen
  `w`. Deterministic seed-free fold assignment declared ex-ante.
- **Hyperparameters fixed ex-ante** (`k`, lr, steps, `λ`), not tuned on any corpus's metric.
- **Model capacity capped** at 6 linear parameters — the smallest model that is the exact
  counterfactual to the fixed weight vector; no non-linearity, no per-query features beyond
  the 6 signals.
- **Branch A requires replication** under a second fold partition before any external claim.
- Appraisals reused verbatim from committed corpus `results.json` (deterministic direct-VAD),
  so `aft_learned` differs from `aft_query_appraised` **only** in the weight vector.

## Scope (explicit)

**In scope:** the `aft_learned` arm on the four corpora, features = the 6 AFT signals, linear
pairwise-LTR with cross-fitting, `k=5` primary + `k=10` sensitivity, interpretability readout
of `w̄` (esp. s2 sign), generalization-gap diagnostic.

**Out of scope:** any `src/` change (the production `fit_retrieval_weights()` API is a
Branch-A follow-up, separately pre-registered/PR'd); non-linear rankers (trees, MLPs);
per-query adaptive weighting; tuning any hyperparameter on a corpus metric; new appraisal
calls beyond the arms already run; pooling queries across corpora; push to `origin/main`
without authorisation.

## Execution (planned harness, committed before the run in a separate PR)

```bash
make bench-z-profile                                        # scored, 4 corpora (needs API key)
make bench-z-profile-dry                                    # smoke, keyword appraiser, no LLM
```

Harness: a shared `benchmarks/common/ltr.py` (`fit_pairwise`, `cross_fit`, feature
standardisation; unit-tested against hand-computed examples incl. a synthetic
counter-congruent set where the recovered `w[1]` must be negative), plus a
`benchmarks/learned_profile/runner.py` that, per corpus, extracts the per-query
(features[n×6], gold[n], metric_fn) triple, cross-fits, computes the 3 arms, and applies the
Holm m=3 / non-inferiority verdict. Reuses `benchmarks/common/statistics` (paired bootstrap,
Holm) and the appraisal-cache pattern of `benchmarks/gate/runner.py`. Dry-run writes
`results.dry.*` and never clobbers scored artifacts.

**Pre-registration integrity:** this document is committed before the harness executes a
single scored run; the closure reports per-corpus held-out Δ/CI/p for Hz1 (Holm m=3) and Hz2
(non-inferiority), the learned `w̄` with the s2-sign readout, the generalization gap, the
`k` sensitivity, and the Branch-A/B verdict with the replication requirement.
