# Addendum Z — Held-out learned retrieval profile

**k:** 5  **Bootstrap:** n=10000, seed=0

| Corpus | metric | role | n | cosine | aft_fixed | aft_learned | gap | s2 | learned-cosine |
|---|---|---|---|---|---|---|---|---|---|
| madialbench | ndcg@5 | break | 160 | 0.304 | 0.166 | 0.319 | +0.008 | negative | Δ=+0.015 [-0.008,+0.040] p=0.1046 |
| esmemeval | u_ndcg@4 | break | 1133 | 0.284 | 0.083 | 0.269 | -0.002 | negative | Δ=-0.015 [-0.023,-0.006] p=0.9997 |
| dailydialog_t2a | top1 | break | 396 | 0.220 | 0.237 | 0.227 | +0.025 | non-negative | Δ=+0.008 [-0.045,+0.061] p=0.4056 |
| realistic_recall_v2 | top1 | preserve | 200 | 0.325 | 0.240 | 0.425 | +0.000 | negative | Δ=+0.100 [+0.060,+0.145] p=0.0000 |

## Verdict

- Hz1 madialbench: Δ=+0.015 p_holm=0.3138 → FAIL
- Hz1 esmemeval: Δ=-0.015 p_holm=0.9997 → FAIL
- Hz1 dailydialog_t2a: Δ=+0.008 p_holm=0.8111 → FAIL
- Hz2 preserve: Δ_vs_fixed=+0.185

**Branch A (≥1 break PASS): False**

Decision rule: `benchmarks/preregistration_addendum_z_learned_profile.md`.
