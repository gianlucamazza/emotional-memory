# Addendum Y — Query-affect-conditioned gate (Hg1 recover / Hg2 preserve)

**τ (primary):** 0.2  **Bootstrap:** n=10000, seed=0

## Per-corpus (primary τ)

| Corpus | metric | n | route→cos | gated | cosine | aft | recov | gated-cos | gated-aft |
|---|---|---|---|---|---|---|---|---|---|
| realistic_recall_v2 | top1 | 200 | 40.0% | 0.420 | 0.325 | 0.420 | -0.00 | Δ=+0.095 [+0.050, +0.140] p1=0.0000 d=0.291 | Δ=+0.000 [-0.035, +0.035] p1=0.5000 d=0.000 |
| dailydialog_t2a | top1 | 396 | 19.9% | 0.237 | 0.220 | 0.222 | -6.00 | Δ=+0.018 [-0.023, +0.061] p1=0.2135 d=0.043 | Δ=+0.015 [-0.005, +0.035] p1=0.0834 d=0.075 |
| esmemeval | u_ndcg@4 | 1133 | 45.5% | 0.209 | 0.284 | 0.133 | 0.50 | Δ=-0.075 [-0.086, -0.064] p1=1.0000 d=-0.393 | Δ=+0.076 [+0.065, +0.087] p1=0.0000 d=0.412 |
| madialbench | ndcg@5 | 160 | 0.0% | 0.245 | 0.304 | 0.245 | 0.00 | Δ=-0.059 [-0.109, -0.008] p1=0.9887 d=-0.176 | Δ=+0.000 [+0.000, +0.000] p1=0.5000 d=nan |

## Verdict (Holm m=2)

- **hg1_recover** (esmemeval gated_vs_aft): Δ=+0.076 [+0.065, +0.087] p_holm=0.0000 → PASS
- **hg2_preserve** (realistic_recall_v2 gated_vs_cosine): Δ=+0.095 [+0.050, +0.140] p_holm=0.0000 → PASS

**Branch A (both PASS): True**

Decision rule: `benchmarks/preregistration_addendum_y_query_affect_gate.md`.
