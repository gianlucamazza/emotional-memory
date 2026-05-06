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
| aft | 0.85 [0.65, 1.00] | 41.39 | 53.51 | 56.22 | ok |
| naive_cosine | 0.80 [0.60, 0.95] | 31.33 | 73.79 | 75.05 | ok |
| recency | 0.25 [0.10, 0.45] | 0.01 | 0.01 | 0.03 | ok |

## Pairwise vs naive_cosine

Two-sided tests: paired bootstrap p-value and exact McNemar p-value.
H0: no difference. CI excludes 0 ↔ difference is credible at 95% level.

| System | Δ [95% CI] | p (bootstrap) | p (McNemar) | N items | Discordant | Padded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aft | 0.05 [-0.15, 0.30] | 0.8250 | 1.0000 | 20 | 5 | 0 |
| recency | -0.55 [-0.75, -0.35] | 0.0000 | 0.0010 | 20 | 11 | 0 |
