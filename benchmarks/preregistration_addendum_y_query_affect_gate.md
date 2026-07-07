# Pre-registration Addendum Y — Hy: query-affect-conditioned gate (safe wrapper + penalty decomposition)

**Status:** PRE-REGISTERED (2026-07-07) — committed before any scored run.
**Date (pre-reg):** 2026-07-07
**Mechanism:** a per-query router. Appraise the query with direct-VAD; if the appraised
query is affectively **neutral** (`|valence| < τ`), retrieve with **pure cosine**;
otherwise retrieve with the Addendum-T affect-conditioned path (`retrieve(..., query_affect=)`).
**Embedder / LLM:** each corpus keeps its own pre-registered embedder and the direct-VAD
appraiser (`DIRECT_VAD_SCHEMA`, gpt-5-mini) already used by its `aft_query_appraised` arm.
**Parent closures:** `preregistration_addendum_q_affect_gating_closure.md` (state-based
gating: Hq2 PASS = safe wrapper recovers the always-on penalty, but the query is never
appraised — "state-injection boundary") · `preregistration_addendum_t_query_appraisal_closure.md`
(Ht1 PASS, retrieve-time query appraisal, +0.115 curated, ~59% recovery / ~82% on the
affect-discriminative subset) · `preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`
(T2A FAIL, naturalistic) · X (`..._x_madialbench_..._closure.md`, counter-congruent FAIL) ·
X2 (`..._x2_esmemeval_..._closure.md`, affect-orthogonal FAIL — which named this study).

---

## Motivation

The X2 closure named the untested variant verbatim: _"affect-gating: engage the affect
channel only when the query carries affect (45.2% here do not)… a query-affect-conditioned
gate is the untested variant."_ Addendum Q closed **state-based** gating (safe wrapper, no
gain, because the query is never appraised). Addendum T then made query appraisal available
at retrieve time. This study is their composition: gate the affect channel **on the
appraised query's own affect**.

**The question.** Does a query-affect gate (a) **preserve** Addendum T's gain in the
affect-discriminative regime, and (b) **recover** the always-on affect penalty on
affect-sparse / affect-orthogonal corpora — without tuning anything on the test data?

**Honest expected outcome, declared ex-ante.** The gate will NOT make AFT beat cosine on
third-party corpora. It is a _partial_ wrapper: it recovers the penalty on **neutral**
queries (routed to cosine) but not on **affect-carrying-yet-misaligned** queries (X2's gold
is affect-orthogonal even for affect-carrying queries; X's is counter-congruent). The real
contribution is to **decompose the off-regime penalty** into a gateable neutral-query
component and a non-gateable affect-carrying component — a decomposition that should map
onto the two measured third-party failure modes (T2A affect-sparse vs X2 affect-orthogonal).
This is the closest available result to a positive (a safer-to-deploy AFT), not "AFT wins".

**Design discovery (why this is cheap and exact).** The gate routes each query _wholly_ to
one arm, and every harness already computes per-query scores for both `naive_cosine` and
`aft_query_appraised`, plus the per-query appraised affect. So the `gated` arm is an **exact
per-query selection** between two already-computed arms — no new retrieval logic, no new LLM
calls beyond the arms already running, and cosine-equivalence on neutral queries is exact by
construction (it _is_ the cosine arm's result). This avoids the Addendum-Q Amendment-1 trap
(a routed `[1,0,0,0,0,0]` re-passed through `adaptive_weights()` is not pure cosine under
non-neutral mood). No `src/` change; harness-only, as Q.

---

## Protocol

- **Gate variable:** `abs(appraised query valence)`. Valence is the calibrated direct-VAD
  axis (r≈0.79 vs EmoBank human gold, near-zero bias; r≈0.80 vs oracle state in Addendum T),
  and the schema's declared neutral is exactly `valence=0.0`. Arousal (r≈0.56) is NOT used
  as a gate variable — it would require Addendum-W recalibration.
- **Gated arm construction (per corpus, per query in file order):**
  `gated[i] = cosine[i]` if `abs(valence_i) < τ` else `aft_query_appraised[i]`, where
  `cosine[i]` / `aft_query_appraised[i]` are the per-query scores the two existing arms
  already produce, and `valence_i` is the appraised query valence already stored by the AFT
  arm (`appraised_query_affect` / the curated `appraised` dict).
- **Threshold τ = 0.2 (primary).** A symmetric neutral band ±0.2 around the schema neutral
  (0), i.e. a 10%-of-range deadzone on [-1,1], with ex-ante pedigree from X/X2 (the D2
  valence-displacement threshold). **Honest disclosure:** 0.2 had prior exposure via X2's
  _post-hoc_ observation that 45.2% of its queries fall in `|valence|<0.2`; it was not chosen
  to optimize the gate. To keep the verdict from resting on a single seen value, the
  **sensitivity across τ ∈ {0.1, 0.2, 0.3}** is reported as a secondary, non-gating quantity,
  and the primary hypotheses are directional (their sign does not hinge on τ's exact value).
- **No tuning** of any weight, schema, prompt, or embedder on any corpus. The gate adds no
  parameter beyond τ, which is pre-declared.

## Corpora and arms

Four corpora, each scored on its **own pre-registered metric** (no cross-corpus pooling);
each already ships a `naive_cosine` and an `aft_query_appraised` arm:

| Corpus                | Regime tested                                 | Runner                                 | Metric                   |
| --------------------- | --------------------------------------------- | -------------------------------------- | ------------------------ |
| `realistic_recall_v2` | curated, affect-discriminative (**preserve**) | `benchmarks/query_appraisal/runner.py` | top1                     |
| DailyDialog (T2A)     | naturalistic affect-sparse (**recover**)      | `benchmarks/dailydialog/t2a_runner.py` | top1                     |
| ES-MemEval (X2)       | affect-orthogonal (**recover**)               | `benchmarks/esmemeval/runner.py`       | upstream-verbatim nDCG@4 |
| MADial-Bench (X)      | counter-congruent (**recover**)               | `benchmarks/madialbench/runner.py`     | nDCG@5                   |

Arms per corpus: `naive_cosine` (baseline), `aft_query_appraised` (Addendum-T always-on),
`gated` (this study). Oracle upper bound: none — the gate signal is the appraised valence
itself; there is no separate ground-truth neutrality label.

## Hypotheses / quantities

- **Hg1 (recover, primary, on X2).** `gated` nDCG@4 > `aft_query_appraised` nDCG@4 (Δ>0,
  one-tailed). The gate improves over always-on AFT by routing neutral queries to cosine.
  X2 is the corpus that generated this study.
- **Hg2 (preserve, primary, on curated v2).** `gated` top1 > `naive_cosine` top1 (Δ>0,
  one-tailed). The gate retains the Addendum-T advantage in the affect-discriminative regime.
- **Secondary / descriptive (non-gating — the scientific content):**
    - `gated vs cosine` and `gated vs aft_query_appraised` on **all four** corpora (Δ, 95% CI);
    - **recovery fraction** per corpus = `(gated − aft) / (cosine − aft)` on the off-regime
      corpora — how much of the always-on penalty the gate removes (the sparse-vs-orthogonal
      decomposition);
    - **routing rate** per corpus = fraction of queries routed to cosine (should ≈ the neutral
      fraction, ~45% on X2) — a gate sanity diagnostic;
    - **best-of-both check** — does `gated` beat _both_ pure arms on any corpus? (upside
      scenario, pre-declared as exploratory);
    - **τ-sensitivity** — Hg1/Hg2 contrasts recomputed at τ ∈ {0.1, 0.2, 0.3}, all corpora.

## Statistical analysis plan (pre-declared)

- **Primary metrics:** per-query, each corpus's own (curated top1; X2 upstream-verbatim
  nDCG@4).
- **Test:** paired bootstrap difference, n=10,000, seed=0, one-tailed
  (`benchmarks/common/statistics.paired_bootstrap_diff`); Cohen's d on paired diffs.
- **Confirmatory family:** {Hg1, Hg2}, **Holm–Bonferroni m=2**, α=0.05. All secondary /
  descriptive quantities cannot gate the verdict.
- **N:** X2 = 1,133 in-family queries; curated v2 = 200. Power notes reported per corpus in
  the closure (observed SD + implied MDE at 80% power).

## Decision rule (pre-declared, ex-ante)

- **Hg1 ∧ Hg2 both PASS (Holm-adjusted p<0.05, Δ>0) → Branch A.** The query-affect gate is a
  validated _partial safe wrapper_: it preserves the on-regime gain and directionally
  recovers the off-regime penalty. **Follow-up, NOT part of this study:** promote the gate to
  a production `retrieve_query_gated()` API in `src/` (forcing the weight vector
  `[1,0,0,0,0,0]` via `precomputed_weights` in `engine._effective_retrieval_weights`,
  bypassing `adaptive_weights`), in its own PR. `claim_validation_matrix.json`: new claim
  `query_affect_gate` = `controlled_evidence`, scoped to "harmlessness off-regime + preserved
  on-regime gain", with the explicit ceiling that it cannot neutralize the affect-orthogonal
  penalty.
- **Hg1 FAIL.** The off-regime penalty is not concentrated on neutral queries; the gate does
  not even recover it → the affect-gating line closes.
- **Hg2 FAIL.** The gate sacrifices the curated gain (gate false-negatives on affect-carrying
  queries, or τ mis-set) → the gate is not deployable as specified; report and close.

Marginal handling: `0.04 < p_holm < 0.05` → "PASS marginal", flagged. No post-hoc threshold
or metric switching.

## Scope (explicit)

**In scope:** the `gated` arm on the four corpora above, gate variable `abs(query valence)`,
τ=0.2 primary + {0.1,0.2,0.3} sensitivity, diagnostics (routing rate, recovery fraction,
best-of-both).

**Out of scope:** any `src/` change (the production `retrieve_query_gated()` API is a
Branch-A follow-up, separately pre-registered/PR'd); tuning τ or any weight on the test
corpora; arousal-based gating (needs Addendum-W recalibration); a PAD-norm gate variant;
new appraisal calls beyond the arms already run; push to `origin/main` without authorisation.

## Execution (planned harness, committed before the run in a separate PR)

```bash
make bench-y-gate                                          # scored, 4 corpora (needs API key)
make bench-y-gate-dry                                      # smoke, keyword appraiser, no LLM
```

Harness: a shared `benchmarks/common/gate.py` (`gated_scores`, `recovery_fraction`,
routing stats; unit-tested against hand-computed examples), plus a `gated`-arm section added
to each of the four runners, computed from the two arms' per-query vectors + the stored
appraised valence. Dry-run writes `results.dry.*` and never clobbers scored artifacts.

**Pre-registration integrity:** this document is committed before the harness executes a
single scored run; the closure reports per-corpus Δ/CI/p for Hg1/Hg2 (Holm m=2), the
recovery-fraction decomposition, routing rates, best-of-both, τ-sensitivity, and the verdict.
