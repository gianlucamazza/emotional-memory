# Realistic Replay — Multi-Seed Robustness Sweep

Dataset: `realistic_recall_v2` v2.0 · embedder: `hash` · seeds: [0, 1, 7, 42, 123] · bootstrap n=2000 · isolation: subprocess-per-seed.

**Retrieval determinism (this run):** ✅ per-query top-1 outcomes were identical across all seeds in this sweep — point estimates did not move; only bootstrap CI bounds jittered.

**Caveat — near-deterministic, not bit-stable.** The RNG seed moves nothing, but the engine stamps encode/retrieve with real wall-clock time and ACT-R decay tracks `now - encoded_at`. Each seed runs in its own subprocess launched at a slightly different instant, so a query at a numerical tie can flip *even with subprocess isolation*. A fresh `make bench-multiseed` therefore reports `retrieval_deterministic=True` most of the time but occasionally `False` (observed cross-seed stdev up to ~0.0025), and the absolute mean drifts a little across sweeps. This variance is timing-driven, sits well inside the bootstrap CIs, and does not change the AFT-vs-baseline conclusion. See `docs/research/08_limitations.md` §2.9.

## Cross-seed `top1_accuracy`

| System | mean | stdev | min | max | spread |
|---|---:|---:|---:|---:|---:|
| `aft` | 0.1200 | 0.0000 | 0.1200 | 0.1200 | 0.0000 |
| `naive_cosine` | 0.0450 | 0.0000 | 0.0450 | 0.0450 | 0.0000 |

## Cross-seed top-1 Δ (vs baseline)

| Comparison | mean Δ | stdev | min | max | spread |
|---|---:|---:|---:|---:|---:|
| `aft_vs_naive_cosine` | +0.0750 | 0.0000 | +0.0750 | +0.0750 | 0.0000 |
