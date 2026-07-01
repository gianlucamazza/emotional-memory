# Addendum X — Third-party Retrieval on MADial-Bench EN (Hx1)

**Queries:** 160  **Memories:** 160  **Embedder:** `bge-small-en-v1.5`  **Bootstrap:** n=10000, seed=0

## Metric grid (per-arm means)

| Metric | naive_cosine | aft_query_appraised |
|---|---|---|
| map@1 | 0.219 | 0.119 |
| map@10 | 0.171 | 0.123 |
| map@3 | 0.147 | 0.103 |
| map@5 | 0.152 | 0.109 |
| mrr@1 | 0.219 | 0.119 |
| mrr@10 | 0.299 | 0.205 |
| mrr@3 | 0.266 | 0.172 |
| mrr@5 | 0.278 | 0.188 |
| ndcg@1 | 0.219 | 0.119 |
| ndcg@10 | 0.348 | 0.261 |
| ndcg@3 | 0.283 | 0.191 |
| ndcg@5 | 0.304 | 0.221 |
| precision@1 | 0.219 | 0.119 |
| precision@10 | 0.067 | 0.058 |
| precision@3 | 0.119 | 0.090 |
| precision@5 | 0.087 | 0.075 |
| recall@1 | 0.111 | 0.068 |
| recall@10 | 0.334 | 0.264 |
| recall@3 | 0.171 | 0.131 |
| recall@5 | 0.220 | 0.173 |

## Hx1 — aft_query_appraised vs naive_cosine

Metric: **ndcg@5**  Δ=-0.083 [-0.123, -0.043]  p_one=0.9998  d=-0.317
MDE (80% power): 0.051 (sd of paired diffs 0.262, N=160)

**Hx1 verdict: FAIL**

## Diagnostics

D1 (appraisal vs third-party labels): AUC(Happy vs negative) = 0.996 (n=109/46; mean valence +0.890 vs -0.116)
D2 (corpus affect-discriminativeness): 76.9% of queries have |gold-set mean valence - bank mean| > 0.2

Decision rule: `benchmarks/preregistration_addendum_x_madialbench_third_party.md`.
