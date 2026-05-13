# Addendum I — Hi3 Confirmatory Analysis

Resonance amplification on affect-rich queries: e5-small-v2 vs SBERT.

Pre-registration: `benchmarks/preregistration_addendum_i.md`
Dataset: `benchmarks/datasets/realistic_recall_v3.json`
Bootstrap: n=10000, seed=1, CI=95%, one_sided_directional, Holm m=3

Statistic: `delta = mean(amp_e5) - mean(amp_sbert)` where `amp[i] = no_resonance_top1_hit[i] - full_top1_hit[i]` (positive = resonance hurts; pre-reg sign convention).
PASS iff `delta > 0.05` AND `p_adj < 0.05`.

## Confirmatory Results

| Hypothesis | Challenge | Δ [95% CI] | p_one | p_adj (Holm) | Cohen's d | Verdict |
|---|---|---|---|---|---|---|
| Hi3 (**primary**) | semantic_confound | 0.090 [0.030, 0.160] | 0.0078 | 0.0234 | 0.257 | **PASS** |
| Hi3_recency (secondary) | recency_confound | 0.070 [0.020, 0.130] | 0.0112 | 0.0234 | 0.239 | **PASS** |
| Hi3_arc (secondary) | affective_arc | 0.010 [-0.020, 0.050] | 0.3795 | 0.3795 | 0.058 | FAIL |

## Mechanism Analysis (Exploratory — no formal test)

| Embedder | links/memory (mean) | links/memory (max) | N memories |
|---|---|---|---|
| SBERT (bge-small) | 4.9999 | 5 | 94125 |
| e5-small-v2 | 4.9999 | 5 | 94125 |

Exploratory — no formal test. Differences in link density may explain differential resonance amplification between embedders.

*Run: 2026-05-06T20:13:17.947713+00:00*
