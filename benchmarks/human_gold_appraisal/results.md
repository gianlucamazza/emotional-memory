# A5 — Appraisal vs human-gold affect (EmoBank, Addendum S)

Dataset: `emobank_v1` v1.0 (N=300, CC-BY-SA 4.0) · bootstrap n=2000 · seed=42 · llm=on.

Source: EmoBank (JULIELab/EmoBank), texts from MASC. Buechel & Hahn 2017, EACL.

## `keyword` engine

| Dimension | N | Pearson r [95% CI] | bias [95% CI] | MAE | human-validated |
|---|---:|---|---|---:|:---:|
| valence | 300 | +0.070 [-0.010, +0.148] | +0.018 [-0.005, +0.041] | 0.143 | ✗ |
| arousal | 300 | +0.106 [+0.012, +0.200] | -0.288 [-0.296, -0.280] | 0.288 | ✅ |
| dominance | 300 | -0.011 [-0.144, +0.110] | -0.023 [-0.033, -0.014] | 0.052 | ✗ |

## `llm` engine

| Dimension | N | Pearson r [95% CI] | bias [95% CI] | MAE | human-validated |
|---|---:|---|---|---:|:---:|
| valence | 300 | +0.703 [+0.655, +0.746] | +0.152 [+0.115, +0.190] | 0.319 | ✅ |
| arousal | 300 | +0.276 [+0.172, +0.373] | -0.058 [-0.071, -0.044] | 0.108 | ✅ |
| dominance | 300 | +0.333 [+0.224, +0.432] | +0.130 [+0.102, +0.158] | 0.256 | ✅ |
