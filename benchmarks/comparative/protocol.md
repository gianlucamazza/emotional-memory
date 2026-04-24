# Comparative Benchmark Protocol

This document fixes the interpretation of the comparative benchmark in this
repository.

## Question being answered

> Does the system surface memories from the intended affective region under a
> controlled query-state setup?

This protocol is intentionally narrow. It is not a leaderboard for general agent
memory quality.

## Benchmark shape

- Dataset: `affect_reference_v1.jsonl`
- Size: 258 synthetic affect-labeled examples
- Query set: 4 representative queries, one per Russell quadrant
- Primary metric: `recall@k` as quadrant-level affect congruence
- Secondary metrics: encode latency, retrieve p50, retrieve p95

## Adapter behavior

- Affect-aware adapters may receive explicit query affect (`valence`, `arousal`)
  so their retrieval logic is actually exercised.
- General-purpose baselines may ignore those fields entirely.
- Semantic-only and recency-only baselines remain useful controls because they
  show what happens when affective state is not part of ranking.

## What the benchmark supports claiming

- AFT changes retrieval behavior in the intended affect-aware direction (validated
  by fidelity tests and the recency-baseline delta on this probe).
- AFT can be compared to semantic-only and recency-only controls on a shared
  synthetic retrieval probe.
- **Ceiling-effect caveat**: with a high-quality SBERT embedder (all-MiniLM-L6-v2)
  and only N = 4 queries × top_k items, both AFT and naive_cosine saturate near
  the same recall level. The probe has insufficient statistical power to detect
  AFT's architecture benefit at this scale; see the realistic replay benchmark
  for per-session ranking-shift evidence.
- Optional systems such as Mem0 and LangMem can be inspected as exploratory
  reference points under this specific protocol.

## What the benchmark does not support claiming

- General superiority over production memory systems
- Best downstream answer quality
- Best overall latency across heterogeneous backends
- Human-like emotional memory
- Human-perceived usefulness or coherence

## Required reporting

Every generated result set should report:

- dataset name and size
- systems included
- embedder used
- `top_k`
- query labels / quadrants
- the interpretive caveats above
