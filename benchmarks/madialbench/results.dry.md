# Addendum X — Third-party Retrieval on MADial-Bench EN (Hx1)

**Queries:** 10  **Memories:** 160  **Embedder:** `bge-small-en-v1.5`  **Bootstrap:** n=10000, seed=0  **[DRY RUN — not a scored result]**

## Metric grid (per-arm means)

| Metric | naive_cosine | aft_query_appraised |
|---|---|---|
| map@1 | 0.100 | 0.000 |
| map@10 | 0.037 | 0.034 |
| map@3 | 0.050 | 0.011 |
| map@5 | 0.034 | 0.024 |
| mrr@1 | 0.100 | 0.000 |
| mrr@10 | 0.187 | 0.095 |
| mrr@3 | 0.150 | 0.033 |
| mrr@5 | 0.170 | 0.078 |
| ndcg@1 | 0.100 | 0.000 |
| ndcg@10 | 0.237 | 0.177 |
| ndcg@3 | 0.163 | 0.050 |
| ndcg@5 | 0.202 | 0.139 |
| precision@1 | 0.100 | 0.000 |
| precision@10 | 0.040 | 0.060 |
| precision@3 | 0.067 | 0.033 |
| precision@5 | 0.060 | 0.080 |
| recall@1 | 0.020 | 0.000 |
| recall@10 | 0.080 | 0.120 |
| recall@3 | 0.040 | 0.020 |
| recall@5 | 0.060 | 0.080 |

## Hx1 — aft_query_appraised vs naive_cosine

Metric: **ndcg@5**  Δ=-0.063 [-0.205, +0.077]  p_one=0.7974  d=-0.261
MDE (80% power): 0.190 (sd of paired diffs 0.242, N=10)

**Hx1 verdict: FAIL** *(dry run — no verdict)*

## Diagnostics

D1 (appraisal vs third-party labels): AUC(Happy vs negative) = 0.516 (n=109/46; mean valence +0.049 vs +0.036)
D2 (corpus affect-discriminativeness): 20.0% of queries have |gold-set mean valence - bank mean| > 0.2

Decision rule: `benchmarks/preregistration_addendum_x_madialbench_third_party.md`.
