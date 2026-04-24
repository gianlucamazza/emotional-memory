# Comparative benchmark results

These numbers come from a **controlled synthetic benchmark** on
`affect_reference_v1`. They measure mood-congruent retrieval behavior under a
small public probe, not general downstream answer quality or production
superiority across memory systems.

Interpretation guardrails:

- `Recall@k` here means quadrant-level affect congruence, not QA accuracy.
- AFT receives explicit query affect (`valence`, `arousal`) in this protocol.
- General-purpose systems such as Mem0 and LangMem are being evaluated on a task
  narrower than their intended product surface.
- CI note: Item-level CI over top_k x N_queries items (N=4 queries x top_k items). Bootstrap percentile, n=2000, seed=0.

| System | Recall@k [95% CI] | Encode ms/item | Retrieve p50 ms | Retrieve p95 ms | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| aft | 0.45 [0.25, 0.65] | 0.92 | 3.37 | 3.81 | ok |
| naive_cosine | 0.45 [0.25, 0.65] | 0.02 | 2.82 | 2.98 | ok |
| recency | 0.25 [0.10, 0.45] | 0.01 | 0.01 | 0.02 | ok |

## Pairwise vs naive_cosine

Two-sided tests: paired bootstrap p-value and exact McNemar p-value.
H0: no difference. CI excludes 0 ↔ difference is credible at 95% level.

| System | Δ [95% CI] | p (bootstrap) | p (McNemar) | N items | Discordant | Padded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aft | 0.00 [-0.20, 0.20] | 1.0000 | 1.0000 | 20 | 4 | 0 |
| recency | -0.20 [-0.45, 0.00] | 0.1235 | 0.2188 | 20 | 6 | 0 |
