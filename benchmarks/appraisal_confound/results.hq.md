# Addendum Q — Affect-Aware Gating (Hq1-Hq3)

Dataset: `realistic_recall_v5_gate` v1.0.0  (50 scenarios, 200 queries)
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 0

## System Results (top1 accuracy)

| System | full | affective half | affect-free half |
|---|---:|---:|---:|
| `naive_cosine` | 0.495 [0.425, 0.565] | 0.140 [0.080, 0.210] | 0.850 [0.780, 0.920] |
| `aft_llm_dual` | 0.390 [0.325, 0.460] | 0.090 [0.040, 0.150] | 0.690 [0.600, 0.780] |
| `aft_gated_oracle` | 0.450 [0.380, 0.520] | 0.050 [0.010, 0.100] | 0.850 [0.780, 0.920] |
| `aft_gated_llm` | 0.470 [0.400, 0.540] | 0.090 [0.040, 0.150] | 0.850 [0.780, 0.920] |

## Hypothesis Tests

### Hq1 (confirmatory) — FAIL

**aft_llm_dual.top1 > naive_cosine.top1 on the affective subset (Δ > 0.05, one-tailed, Holm m=3)**  (N = 100)

Δ = -0.05 [-0.11, 0.00]  p_one_sided = 0.0597  p_holm = 0.0854  Cohen's d = -0.168

### Hq2 (confirmatory) — PASS

**aft_gated_llm.top1 > aft_llm_dual.top1 on the full corpus (Δ > 0.05, one-tailed, Holm m=3)**  (N = 200)

Δ = 0.08 [0.04, 0.12]  p_one_sided = 0.0003  p_holm = 0.0009  Cohen's d = 0.248

### Hq3 (confirmatory) — FAIL

**aft_gated_llm.top1 > naive_cosine.top1 on the full corpus (Δ > 0.05, one-tailed, Holm m=3)**  (N = 200)

Δ = -0.03 [-0.05, 0.00]  p_one_sided = 0.0427  p_holm = 0.0854  Cohen's d = -0.135

### Hq4 (exploratory) — FAIL

**aft_gated_oracle vs naive_cosine, full corpus (exploratory upper bound)**  (N = 200)

Δ = -0.04 [-0.07, -0.02]  p_one_sided = 0.0024  Cohen's d = -0.216

### Hq5 (exploratory) — REPORTED

**aft_gated_llm vs naive_cosine on the affect-free subset (exploratory: gate false-positive cost; expected ≈ 0)**  (N = 100)

Δ = 0.00 [0.00, 0.00]  p_one_sided = 0.5000  Cohen's d = nan

## Gate Classifier (Hq6)

### `aft_gated_oracle` — accuracy 1.0

Confusion (gt -> predicted): `{"affect_free": {"affect_free": 100}, "affective": {"affective": 99}}`

### `aft_gated_llm` — accuracy 0.8442

Confusion (gt -> predicted): `{"affect_free": {"affect_free": 99, "affective": 1}, "affective": {"affect_free": 30, "affective": 69}}`
