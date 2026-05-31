# Addendum O — SEC→Affect Mapping Recalibration

Train events: 522 | Test events: 228 | seed: 42

## Valence (held-out test)

| Model | bias | MAE | Pearson r |
|---|---|---|---|
| baseline | +0.200 | 0.306 | 0.807 |
| M0 | +0.200 | 0.306 | 0.807 |
| M1 | +0.072 | 0.298 | 0.789 |
| M2 | +0.020 | 0.280 | 0.808 |

## Arousal (held-out test)

| Model | bias | MAE | Pearson r |
|---|---|---|---|
| baseline | -0.144 | 0.182 | 0.466 |
| M0 | -0.019 | 0.132 | 0.466 |
| M1 | -0.023 | 0.118 | 0.454 |
| M2 | -0.018 | 0.111 | 0.520 |
