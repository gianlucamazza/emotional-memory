# Addendum X2 — Third-party Retrieval on ES-MemEval/EvoEmo (Hx2)

**Queries:** 1133 in-family (294 zero-gold excluded)  **Sessions:** 401  **Pool:** 50  **Embedder:** `bge-small-en-v1.5`  **Bootstrap:** n=10000, seed=0

## Metric grid (per-arm means; `u_*` = upstream-verbatim formulas)

| Metric | naive_cosine | aft_query_appraised |
|---|---|---|
| map@1 | 0.378 | 0.151 |
| map@10 | 0.478 | 0.236 |
| map@3 | 0.426 | 0.175 |
| map@4 | 0.443 | 0.187 |
| map@5 | 0.453 | 0.199 |
| mrr@1 | 0.378 | 0.151 |
| mrr@10 | 0.514 | 0.265 |
| mrr@3 | 0.474 | 0.206 |
| mrr@4 | 0.488 | 0.221 |
| mrr@5 | 0.496 | 0.233 |
| ndcg@1 | 0.378 | 0.151 |
| ndcg@10 | 0.585 | 0.352 |
| ndcg@3 | 0.505 | 0.225 |
| ndcg@4 | 0.528 | 0.250 |
| ndcg@5 | 0.545 | 0.273 |
| precision@1 | 0.378 | 0.151 |
| precision@10 | 0.101 | 0.077 |
| precision@3 | 0.215 | 0.100 |
| precision@4 | 0.184 | 0.092 |
| precision@5 | 0.159 | 0.087 |
| recall@1 | 0.327 | 0.123 |
| recall@10 | 0.785 | 0.589 |
| recall@3 | 0.535 | 0.238 |
| recall@4 | 0.595 | 0.285 |
| recall@5 | 0.637 | 0.340 |
| u_ndcg@2 | 0.227 | 0.095 |
| u_ndcg@4 | 0.284 | 0.133 |
| u_ndcg@6 | 0.314 | 0.168 |
| u_recall@2 | 0.444 | 0.184 |
| u_recall@4 | 0.595 | 0.285 |
| u_recall@6 | 0.684 | 0.386 |

## Hx2 — aft_query_appraised vs naive_cosine

Metric: **u_ndcg@4** (upstream-verbatim formula)  Δ=-0.150 [-0.165, -0.136]  p_one=1.0000  d=-0.622
MDE (80% power): 0.018 (sd of paired diffs 0.242, N=1133)

**Hx2 verdict: FAIL**

## Per-capability (primary metric, descriptive)

| Capability | n | aft | cosine | Δ |
|---|---|---|---|---|
| abstention | 3 | 0.000 | 0.333 | -0.333 |
| conflict detection | 259 | 0.120 | 0.287 | -0.166 |
| information extraction | 301 | 0.155 | 0.316 | -0.161 |
| temporal reasoning | 273 | 0.153 | 0.285 | -0.131 |
| user modeling | 297 | 0.104 | 0.247 | -0.142 |

## Diagnostics

D1 (appraisal vs third-party labels): AUC(positive vs negative) = 0.950 [0.901, 0.988] (n=7/376; mean valence +0.633 vs -0.160)
D2 (corpus affect-discriminativeness, per-seeker banks): 63.0% of queries have |gold-set mean valence - seeker bank mean| > 0.2

Decision rule: `benchmarks/preregistration_addendum_x2_esmemeval_third_party.md`.
