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

| System | Recall@k | Encode ms/item | Retrieve p50 ms | Retrieve p95 ms | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| aft | 0.85 | 41.26 | 51.28 | 58.29 | ok |
| naive_cosine | 0.8 | 31.71 | 69.05 | 82.18 | ok |
| recency | 0.25 | 0.01 | 0.02 | 0.04 | ok |
| mem0 | 0.95 | 1363.53 | 161.25 | 178.06 | ok |
| letta | — | — | — | — | not_evaluated |
| langmem | 0.9 | 143.24 | 170.37 | 183.59 | ok |
