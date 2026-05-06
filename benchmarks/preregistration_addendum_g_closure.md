# Pre-Registration Addendum G — Hg1 Closure (LLM Appraisal Dual-Path)

**Status:** PENDING_EXECUTION
**Template written:** 2026-05-06
**Date executed:** {{date_executed}}
**Protocol version:** addendum_g_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_g.md`

> **Epistemic status:** This is a pre-specified closure template. Sections marked
> `{{placeholder}}` must be filled verbatim from `results.hg1.{json,md}` after
> execution. The two interpretive branches below (§ Interpretation) are pre-registered
> and must not be modified after results are known. All values in the Results section
> are PENDING — no data has been collected at the time of this commit.

---

## Background

Addendum G tests whether AFT with `LLMAppraisalEngine` in dual-path mode provides a
net positive over `naive_cosine` on a benchmark where affect is **not preset** by the
author (`realistic_recall_v3_noAF.json`).

Three prior findings motivate this study: (1) the Hd1/Hd2 oracle-affect circularity —
AFT's advantage was demonstrated with author-preset valence/arousal, not inferred
affect; (2) `KeywordAppraisalEngine` is destructive (Ha2/Hb2 FAIL, Δ=−0.39 vs oracle
AFT); (3) dual-path deferred appraisal mitigates synchronous override (Hf1 PASS). The
open question is whether a high-quality LLM appraisal engine closes the gap between
oracle-preset and no-affect conditions.

Infrastructure committed prior to execution: `runner_hg1.py` (600 LoC),
`realistic_recall_v3_noAF.json` (50 scenarios × 4 queries = N=200, no preset
valence/arousal), `bench-addendum-g` Makefile target. Dataset committed author-blind
at `055a78a` before any run.

---

## Execution

```bash
export EMOTIONAL_MEMORY_LLM_API_KEY=<key>
export EMOTIONAL_MEMORY_LLM_MODEL={{llm_model}}          # e.g. gpt-5-mini
make bench-addendum-g
```

Parameters (frozen in `runner_hg1.py`):
- seed: 0
- n_bootstrap: 10,000
- α: 0.05 (one-tailed)
- Δ threshold: 0.05 (5 percentage points)
- Dataset: `benchmarks/datasets/realistic_recall_v3_noAF.json` (N={{n_queries}} queries)
- Systems: `aft_llm_dual` (LLMAppraisalEngine + dual_path_encoding=True + elaborate())
  vs `naive_cosine` (primary); `aft_neutral`, `aft_llm_sync` (exploratory only)
- No Holm correction: Hg1 is the sole confirmatory hypothesis

Output files:
- `benchmarks/appraisal_confound/results.hg1.json`
- `benchmarks/appraisal_confound/results.hg1.md`
- `benchmarks/appraisal_confound/results.hg1.protocol.json`

---

## Results

*To be filled verbatim from `results.hg1.md` after execution.*

### System accuracy

| System | N | top1_acc | 95% CI |
|---|---|---|---|
| aft_llm_dual | {{n_queries}} | {{aft_llm_dual_top1}} | [{{aft_llm_dual_ci_lo}}, {{aft_llm_dual_ci_hi}}] |
| naive_cosine | {{n_queries}} | {{naive_cosine_top1}} | [{{naive_cosine_ci_lo}}, {{naive_cosine_ci_hi}}] |

### Hg1 (confirmatory) — aft_llm_dual.top1 > naive_cosine.top1

| Metric | Value |
|---|---|
| Δ (aft_llm_dual − naive_cosine) | {{delta}} |
| 95% CI of Δ | [{{ci_lo}}, {{ci_hi}}] |
| p_one_sided (bootstrap) | {{p_one}} |
| Cohen's d (paired) | {{cohens_d}} |
| Δ > 0.05 threshold | {{threshold_met}} |
| **Verdict** | **{{verdict}}** |

### Hg2 (exploratory) — aft_llm_dual.top1 > aft_neutral.top1

| Δ | p_one_sided | Result |
|---|---|---|
| {{hg2_delta}} | {{hg2_p}} | {{hg2_result}} |

### Hg3 (exploratory) — aft_llm_dual.top1 > aft_llm_sync.top1

| Δ | p_one_sided | Result |
|---|---|---|
| {{hg3_delta}} | {{hg3_p}} | {{hg3_result}} |

---

## Interpretation

*Pre-specified branches — select the applicable branch post-execution and delete the
other. Do not modify the selected branch text.*

### Branch A — If Hg1 PASS (Δ > 0.05 AND p_one < 0.05)

**Hg1 PASS.** AFT with `LLMAppraisalEngine` in dual-path mode outperforms naive cosine
by Δ = {{delta}} pp (d = {{cohens_d}}) on N = {{n_queries}} affect-free queries. This
partially resolves the oracle-affect circularity limitation identified in §7: the
architecture advantage is not solely an artefact of author-preset affect fields, but
holds under LLM-inferred appraisal.

**Scope and caveats:** The result is conditional on the specific LLM model used
({{llm_model}}). Appraisal quality is embedder- and language-dependent; the finding
extends the Hd1/Hd2 claims to the LLM-dual-path configuration on EN scenarios only.
The §7 paragraph "appraisal-engine dependency" should be updated to reflect that LLM
dual-path shows net positive over naive cosine, while noting model-dependence.

**Exploratory signal:** Hg2 and Hg3 results ({{hg2_result}}, {{hg3_result}}) are
descriptive — they cannot be promoted to confirmatory claims.

**`claim_validation_matrix.json` update:** row `appraisal_llm_real_dual_path` →
`status: PASS`, `delta: {{delta}}`, `p: {{p_one}}`, `d: {{cohens_d}}`.

**§7 Limitations action:** Remove or qualify the paragraph beginning "the oracle-affect
circularity..." — replace with a scoped statement acknowledging the LLM-dual-path
positive result and its model-dependence. Keep the language-scope limitation (EN only
for the LLM appraisal prompt).

---

### Branch B — If Hg1 FAIL (Δ ≤ 0.05 OR p_one ≥ 0.05)

**Hg1 FAIL.** AFT with `LLMAppraisalEngine` in dual-path mode does not provide a
reliably detectable advantage over naive cosine on N = {{n_queries}} affect-free
queries at the pre-specified threshold (Δ > 0.05, p < 0.05). The observed Δ =
{{delta}} (d = {{cohens_d}}, p_one = {{p_one}}).

**Interpretation:** The oracle-affect circularity (§7 Limitations) is not resolved by
the LLM dual-path configuration at this sample size and model ({{llm_model}}). Two
non-exclusive explanations are plausible: (1) LLM appraisal quality on noAF scenarios
is insufficient to generate reliable affective signal; (2) the N=200 dataset has
insufficient power to detect a smaller-than-expected effect. The §7 paragraph on
"appraisal-engine dependency" stands as written.

**What is NOT implied:** This FAIL does not invalidate Hd1/Hd2 (which use oracle
preset affect) nor Hi1/Hi2 (resonance magnitude amplification). The architecture
claims remain valid in their pre-registered scope (oracle affect). The Hg1 FAIL
narrows the claim: the advantage does not extend to LLM-inferred affect at d ≈ {{cohens_d}}.

**`claim_validation_matrix.json` update:** row `appraisal_llm_real_dual_path` →
`status: FAIL`, `delta: {{delta}}`, `p: {{p_one}}`, `d: {{cohens_d}}`.

**§7 Limitations action:** Keep the paragraph on oracle-affect circularity. Add one
sentence: "Addendum G (Hg1 FAIL, Δ={{delta}}, d={{cohens_d}}) confirms that the
advantage does not extend to LLM dual-path appraisal at this sample size."

---

## Coherence with prior closures

- **Hd1 PASS (Add. D)**: architecture advantage established with oracle preset affect.
  Hg1 tests whether LLM appraisal is a viable substitute for oracle. If PASS: convergent
  validity. If FAIL: scope limitation confirmed empirically.
- **Ha2/Hb2 FAIL (Add. A)**: `KeywordAppraisalEngine` is destructive. Hg1 tests
  whether the LLM engine avoids this destruction. The dual-path + elaborate() design
  (Hf1 mechanism) is the architectural difference.
- **Hf1 PASS (Add. F)**: dual-path deferral mitigates synchronous override. Hg3
  (exploratory) operationalizes this: `aft_llm_dual > aft_llm_sync` would confirm
  the dual-path temporal advantage under real LLM appraisal, not just keyword.
- **Hi1/Hi2 PASS**: resonance amplification is independent of the appraisal path —
  Hi1/Hi2 used oracle affect. Hg1 result does not affect Hi1/Hi2 validity.

---

## Update checklist (to complete at execution time)

- [ ] Replace all `{{placeholder}}` values with verbatim numbers from `results.hg1.json`
- [ ] Select and delete the inapplicable interpretive branch (A or B)
- [ ] Set `Status: PASS` or `Status: FAIL` in the header
- [ ] Set `Date executed: <actual date>` in the header
- [ ] Update `docs/research/claim_validation_matrix.json` row `appraisal_llm_real_dual_path`
- [ ] Update `paper/main.tex §7 Limitations` per the selected branch action
- [ ] Run `make reproduce-paper-check` and confirm zero diff
- [ ] Run `make check` and confirm suite passes
- [ ] Commit atomically: closure doc + claim matrix + §7 edit in one commit
