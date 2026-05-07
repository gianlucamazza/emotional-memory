# Pre-Registration Addendum G — Hg1 Closure (LLM Appraisal Dual-Path)

**Status:** FAIL
**Template written:** 2026-05-06
**Date executed:** 2026-05-07
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
export EMOTIONAL_MEMORY_LLM_MODEL=gpt-5-mini
make bench-addendum-g
```

Parameters (frozen in `runner_hg1.py`):
- seed: 0
- n_bootstrap: 10,000
- α: 0.05 (one-tailed)
- Δ threshold: 0.05 (5 percentage points)
- Dataset: `benchmarks/datasets/realistic_recall_v3_noAF.json` (N=200 queries)
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
| aft_llm_dual | 200 | 0.315 | [0.250, 0.380] |
| naive_cosine | 200 | 0.325 | [0.260, 0.390] |

### Hg1 (confirmatory) — aft_llm_dual.top1 > naive_cosine.top1

| Metric | Value |
|---|---|
| Δ (aft_llm_dual − naive_cosine) | -0.010 |
| 95% CI of Δ | [-0.055, 0.035] |
| p_one_sided (bootstrap) | 0.3669 |
| Cohen's d (paired) | -0.032 |
| Δ > 0.05 threshold | No |
| **Verdict** | **FAIL** |

### Hg2 (exploratory) — aft_llm_dual.top1 > aft_neutral.top1

| Δ | p_one_sided | Result |
|---|---|---|
| 0.000 | 0.5000 | FAIL |

### Hg3 (exploratory) — aft_llm_dual.top1 > aft_llm_sync.top1

| Δ | p_one_sided | Result |
|---|---|---|
| 0.185 | 0.0000 | PASS |

---

## Interpretation

*Pre-specified branches — Branch A deleted (Hg1 FAIL). Branch B selected.*

### Branch B — Hg1 FAIL (Δ ≤ 0.05 OR p_one ≥ 0.05)

**Hg1 FAIL.** AFT with `LLMAppraisalEngine` in dual-path mode does not provide a
reliably detectable advantage over naive cosine on N = 200 affect-free
queries at the pre-specified threshold (Δ > 0.05, p < 0.05). The observed Δ =
-0.010 (d = -0.032, p_one = 0.3669).

**Interpretation:** The oracle-affect circularity (§7 Limitations) is not resolved by
the LLM dual-path configuration at this sample size and model (gpt-5-mini). Two
non-exclusive explanations are plausible: (1) LLM appraisal quality on noAF scenarios
is insufficient to generate reliable affective signal; (2) the N=200 dataset has
insufficient power to detect a smaller-than-expected effect. The §7 paragraph on
"appraisal-engine dependency" stands as written.

**What is NOT implied:** This FAIL does not invalidate Hd1/Hd2 (which use oracle
preset affect) nor Hi1/Hi2 (resonance magnitude amplification). The architecture
claims remain valid in their pre-registered scope (oracle affect). The Hg1 FAIL
narrows the claim: the advantage does not extend to LLM-inferred affect at d ≈ -0.032.

**`claim_validation_matrix.json` update:** row `appraisal_llm_real_dual_path` →
`status: FAIL`, `delta: -0.010`, `p: 0.3669`, `d: -0.032`.

**§7 Limitations action:** Keep the paragraph on oracle-affect circularity. Add one
sentence: "Addendum G (Hg1 FAIL, Δ=-0.010, d=-0.032) confirms that the
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

- [x] Replace all `{{placeholder}}` values with verbatim numbers from `results.hg1.json`
- [x] Select and delete the inapplicable interpretive branch (A or B) — Branch A deleted
- [x] Set `Status: FAIL` in the header
- [x] Set `Date executed: 2026-05-07` in the header
- [x] Update `docs/research/claim_validation_matrix.json` row `appraisal_llm_real_dual_path`
- [x] Update `paper/main.tex §7 Limitations` per the selected branch action
- [ ] Run `make reproduce-paper-check` and confirm zero diff
- [ ] Run `make check` and confirm suite passes
- [ ] Commit atomically: closure doc + claim matrix + §7 edit in one commit
