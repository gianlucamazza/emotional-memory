# Addendum W — affine calibration of direct-VAD arousal

Dataset: `emobank_v1` v1.0 · dimension: **arousal** · N=300 · bootstrap n=2000 · seed=42.

Deployable affine coefficients (full-sample fit): `arousal_cal = +0.1627·arousal_direct +0.4491`.

## Protocol `native_split` — fit `a=+0.1696, b=+0.4465` (n_fit=250, n_eval=50)

| Estimator | MAE | bias | Pearson r |
|---|---:|---:|---:|
| direct_vad (raw) | 0.1998 | -0.0938 | +0.5226 |
| direct_vad (calibrated) | 0.0396 | +0.0015 | +0.5226 |
| scherer_m1 | 0.1054 | -0.0493 | +0.3492 |

- **Hw1** (calibrated < raw MAE): ΔMAE=+0.1603 CI=[+0.1254, +0.1941] p=0.0000 → ✅ improves
- **Hw2** (calibrated < scherer MAE): ΔMAE=+0.0658 CI=[+0.0474, +0.0857] p=0.0000 → ✅ dominates
- **Gw** (r preserved): ✅

## Protocol `kfold_cv` — 5-fold out-of-fold (n_eval=300)

| Estimator | MAE | bias | Pearson r |
|---|---:|---:|---:|
| direct_vad (raw) | 0.1972 | -0.1027 | +0.5724 |
| direct_vad (calibrated) | 0.0407 | +0.0002 | +0.5651 |
| scherer_m1 | 0.1156 | -0.0598 | +0.2072 |

- **Hw1** (calibrated < raw MAE): ΔMAE=+0.1565 CI=[+0.1438, +0.1687] p=0.0000 → ✅ improves
- **Hw2** (calibrated < scherer MAE): ΔMAE=+0.0750 CI=[+0.0661, +0.0841] p=0.0000 → ✅ dominates
- **Gw** (r preserved): ✅

## Decision

- Hw1 (reduces MAE): ✅ PASS
- Hw2 (beats scherer MAE): ✅ PASS
- Gw (slope>0 & r preserved): ✅ OK
- **Adopt affine calibration:** ✅ YES
