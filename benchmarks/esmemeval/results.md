# Addendum X2 — Third-party Retrieval on ES-MemEval/EvoEmo (Hx2)

**Queries:** 1133 in-family (294 zero-gold excluded)  **Sessions:** 401  **Pool:** 50  **Embedder:** `bge-small-en-v1.5`  **Bootstrap:** n=10000, seed=0

## Metric grid (per-arm means; `u_*` = upstream-verbatim formulas)

| Metric | naive_cosine | aft_query_appraised |
|---|---|---|
| map@1 | 0.378 | 0.142 |
| map@10 | 0.478 | 0.216 |
| map@3 | 0.426 | 0.159 |
| map@4 | 0.443 | 0.170 |
| map@5 | 0.453 | 0.182 |
| mrr@1 | 0.378 | 0.142 |
| mrr@10 | 0.514 | 0.249 |
| mrr@3 | 0.474 | 0.192 |
| mrr@4 | 0.488 | 0.204 |
| mrr@5 | 0.496 | 0.217 |
| ndcg@1 | 0.378 | 0.142 |
| ndcg@10 | 0.585 | 0.329 |
| ndcg@3 | 0.505 | 0.209 |
| ndcg@4 | 0.528 | 0.229 |
| ndcg@5 | 0.545 | 0.254 |
| precision@1 | 0.378 | 0.142 |
| precision@10 | 0.101 | 0.072 |
| precision@3 | 0.215 | 0.090 |
| precision@4 | 0.184 | 0.082 |
| precision@5 | 0.159 | 0.080 |
| recall@1 | 0.327 | 0.114 |
| recall@10 | 0.785 | 0.548 |
| recall@3 | 0.535 | 0.212 |
| recall@4 | 0.595 | 0.255 |
| recall@5 | 0.637 | 0.312 |
| u_ndcg@2 | 0.227 | 0.088 |
| u_ndcg@4 | 0.284 | 0.120 |
| u_ndcg@6 | 0.314 | 0.155 |
| u_recall@2 | 0.444 | 0.169 |
| u_recall@4 | 0.595 | 0.255 |
| u_recall@6 | 0.684 | 0.357 |

## Hx2 — aft_query_appraised vs naive_cosine

Metric: **u_ndcg@4** (upstream-verbatim formula)  Δ=-0.164 [-0.178, -0.149]  p_one=1.0000  d=-0.652
MDE (80% power): 0.019 (sd of paired diffs 0.251, N=1133)

**Hx2 verdict: FAIL**

## Per-capability (primary metric, descriptive)

| Capability | n | aft | cosine | Δ |
|---|---|---|---|---|
| abstention | 3 | 0.000 | 0.333 | -0.333 |
| conflict detection | 259 | 0.135 | 0.287 | -0.152 |
| information extraction | 301 | 0.133 | 0.316 | -0.183 |
| temporal reasoning | 273 | 0.120 | 0.285 | -0.165 |
| user modeling | 297 | 0.094 | 0.247 | -0.153 |

## Diagnostics

D1 (appraisal vs third-party labels): AUC(positive vs negative) = 0.971 [0.938, 0.994] (n=7/376; mean valence +0.700 vs -0.175)
D2 (corpus affect-discriminativeness, per-seeker banks): 68.2% of queries have |gold-set mean valence - seeker bank mean| > 0.2

Decision rule: `benchmarks/preregistration_addendum_x2_esmemeval_third_party.md`.
