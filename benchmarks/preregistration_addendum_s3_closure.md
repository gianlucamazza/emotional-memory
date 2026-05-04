# Pre-registration Addendum — Study S3 Closure

**Date executed:** 2026-05-04
**Protocol version:** addendum_s3_closure_v1
**Parent pre-reg:** `benchmarks/preregistration.md` (Study S3 — Layer Ablation v2 Powered, lines 193–237)

> **Epistemic status:** This document records the execution and results of the
> pre-registered Study S3. The canonical result files were written at execution time;
> this document provides the interpretive closure. All numbers match the committed
> JSON files exactly.

---

## Background

Study S3 pre-registered a powered (N=200) layer ablation on `realistic_recall_v2`,
replicating the v1 ablation (N=20, not powered) at sufficient sample size to detect
`Δ ≥ 0.10` with pre-specified Holm-corrected bootstrap tests.

Pre-registered hypotheses (confirmatory):
- **Ha:** removing `MoodField` (no_mood) reduces `top1_accuracy` vs full AFT.
- **Hb:** removing `ResonanceLink` (no_resonance) reduces `top1_accuracy` vs full AFT.
- **Hc (invariant):** `no_appraisal` is functionally equivalent to `full` (no appraisal engine configured on this benchmark).
- **Hd (exploratory):** `no_momentum` direction and magnitude.

---

## Execution

```bash
make bench-s3-sbert   # BAAI/bge-small-en-v1.5
make bench-s3-e5      # intfloat/e5-small-v2
```

Canonical result files:
- `benchmarks/ablation/results.v2.sbert.json`
- `benchmarks/ablation/results.v2.e5.json`

Parameters: seed=0, n_bootstrap=2000, N=200 (realistic_recall_v2), Holm correction on Ha+Hb family.

---

## Results summary

| Hypothesis | Embedder | Verdict | full top1 | variant top1 | Δ | p_boot | p_adj_holm |
|---|---|---|---|---|---|---|---|
| Ha (no_mood) | SBERT | **FAIL** | 0.54 | 0.52 | -0.02 | 0.264 | 1.000 |
| Ha (no_mood) | e5 | **FAIL** | 0.51 | 0.50 | -0.005 | 0.915 | 1.000 |
| Hb (no_resonance) | SBERT | **FAIL** | 0.54 | 0.56 | +0.02 | 0.203 | 1.000 |
| Hb (no_resonance) | e5 | **FAIL** | 0.51 | 0.59 | +0.085 | 0.000 | 0.000 |
| Hc (no_appraisal) | SBERT | **PASS** | 0.54 | 0.53 | -0.01 | 0.283 | — |
| Hc (no_appraisal) | e5 | **PASS** | 0.51 | 0.51 | +0.005 | 0.880 | — |
| Hd (no_momentum, expl.) | SBERT | NS | 0.54 | 0.56 | +0.02 | 0.067 | — |
| Hd (no_momentum, expl.) | e5 | NS | 0.51 | 0.51 | 0.00 | 1.000 | — |

Destructive variants (replication of prior results, not S3 hypotheses):

| Variant | SBERT Δ vs full | e5 Δ vs full | Verdict |
|---|---|---|---|
| dual_path | -0.20 | -0.27 | Destructive (He1 replicated) |
| aft_keyword_synchronous | -0.45 | -0.45 | Destructive (Hf1 base replicated) |
| no_reconsolidation | +0.01 | +0.03 | Neutral (He2 null replicated) |

**Hf1 direct comparison (Add. F, secondary on v2):**
dual_path vs aft_keyword_synchronous direct paired bootstrap (n=10,000, seed=0):

| Embedder | Δ_Hf1 [95% CI] | p (one-tailed) | CI above 0 | Verdict |
|---|---|---|---|---|
| SBERT bge-small-en | +0.255 [0.190, 0.320] | <0.001 | ✓ | **Hf1 PASS** |
| e5-small-v2 | +0.165 [0.110, 0.225] | <0.001 | ✓ | **Hf1 PASS** |

Primary Hf1 (v1.4, N=100, n=10,000, seed=0): Δ=+0.290 [0.190, 0.390], p<0.001, CI above 0 → **PASS**. See `benchmarks/preregistration_addendum_f_closure.md`.

---

## Interpretation

**Ha FAIL, Hb FAIL — both embedders.** Removing the MoodField or ResonanceLink
layer in isolation does not reduce `top1_accuracy` at N=200 on realistic_recall_v2.
For the SBERT embedder, both Ha and Hb are non-significant. For e5, the resonance
removal produces a statistically significant *improvement* (Δ=+0.085, p<0.001),
suggesting that the spreading-activation mechanism may interfere with e5's
distance geometry on this benchmark.

This does not imply the layers are inert. The Hd2 study (Addendum D closure)
shows that the full AFT architecture maintains a system-level advantage over
naive cosine (Δ=0.125–0.163), which cannot be explained by any single ablated
component. The per-layer contribution is not isolatable with this study design;
joint effects and interaction terms would require a factorial design.

**Hc PASS — both embedders.** The no_appraisal variant is correctly inert,
confirming the benchmark's methodological invariant holds.

**Pre-registered negative results per Reporting rule 5:** Ha and Hb FAIL are
reported as publishable findings — "insufficient evidence that mood or resonance
individually contribute to top1 accuracy at N=200, Δ_min=0.10."
