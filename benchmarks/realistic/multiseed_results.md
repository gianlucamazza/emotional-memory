# Realistic Replay — Multi-Seed Robustness Sweep

Dataset: `realistic_recall_v2` v2.0 · embedder: `hash` · seeds: [0, 1, 7, 42, 123] · bootstrap n=2000 · isolation: subprocess-per-seed.

**Retrieval determinism:** ✅ per-query top-1 outcomes are identical across all seeds — point estimates do not move; only bootstrap CI bounds jitter.

## Cross-seed `top1_accuracy`

| System | mean | stdev | min | max | spread |
|---|---:|---:|---:|---:|---:|
| `aft` | 0.1250 | 0.0000 | 0.1250 | 0.1250 | 0.0000 |
| `naive_cosine` | 0.0450 | 0.0000 | 0.0450 | 0.0450 | 0.0000 |

## Cross-seed top-1 Δ (vs baseline)

| Comparison | mean Δ | stdev | min | max | spread |
|---|---:|---:|---:|---:|---:|
| `aft_vs_naive_cosine` | +0.0800 | 0.0000 | +0.0800 | +0.0800 | 0.0000 |
