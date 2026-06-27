# Addendum V — direct-VAD appraisal vs SEC->projection

Dataset: `emobank_v1` v1.0 (N=300, EmoBank human VAD) · bootstrap n=2000 · seed=42.

## Per-dimension r / bias / MAE

| Method | Dimension | Pearson r [95% CI] | bias | MAE |
|---|---|---|---:|---:|
| `scherer_m1` | valence | +0.695 [+0.644, +0.740] | +0.157 | 0.324 |
| `scherer_m1` | arousal | +0.228 [+0.129, +0.324] | -0.060 | 0.112 |
| `scherer_m1` | dominance | +0.307 [+0.192, +0.411] | +0.135 | 0.260 |
| `direct_vad` | valence | +0.790 [+0.750, +0.824] | -0.013 | 0.329 |
| `direct_vad` | arousal | +0.582 [+0.510, +0.652] | -0.093 | 0.193 |
| `direct_vad` | dominance | +0.428 [+0.325, +0.521] | -0.018 | 0.142 |

## Paired Δr (direct_vad minus scherer_m1)

| Dimension | Δr [95% CI] | improves? |
|---|---|:---:|
| valence | +0.095 [+0.052, +0.141] | ✅ |
| arousal | +0.354 [+0.251, +0.457] | ✅ |
| dominance | +0.122 [-0.008, +0.243] | — |

## Verdicts

- **Hv1** (arousal r improved): ✅ PASS
- **Hv2** (dominance r improved): ✗ FAIL
- **Hv3** (valence |bias| reduced): ✅ PASS
- **Gv** (valence not regressed): ✅ OK
- **Decision — adopt direct-VAD:** ✅ YES
