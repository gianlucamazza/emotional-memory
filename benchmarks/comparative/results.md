# Comparative benchmark results

| System | Recall@k | Encode ms/item | Retrieve p50 ms | Retrieve p95 ms | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| aft | 0.85 | 41.26 | 51.28 | 58.29 | ok |
| naive_cosine | 0.8 | 31.71 | 69.05 | 82.18 | ok |
| recency | 0.25 | 0.01 | 0.02 | 0.04 | ok |
| mem0 | 0.95 | 1363.53 | 161.25 | 178.06 | ok |
| letta | — | — | — | — | not_evaluated |
| langmem | 0.9 | 143.24 | 170.37 | 183.59 | ok |
