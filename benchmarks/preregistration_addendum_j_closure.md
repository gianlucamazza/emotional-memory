# Pre-registration Addendum J — Closure: LoCoMo Per-Task Pareto Study

**Date executed:** 2026-05-06
**Pre-registration:** `benchmarks/preregistration_addendum_j.md` (commit `7bcd663`)
**Runner:** `benchmarks/locomo/pareto_runner.py`
**Verdict (Hj1):** **FAIL**

---

## Configuration verification (no post-hoc deviation)

| Parameter | Pre-registered | Executed |
|---|---|---|
| Weight configs | W0–W9 (10 configs, frozen grid) | ✓ unchanged |
| Subsample | 50 QA × 4 categories = 200 QA | ✓ 200 QA across 10 conversations |
| Sample seed | 42 | ✓ 42 |
| naive_rag | 1 run on same subsample | ✓ 1 run |
| Embedder | bge-small-en-v1.5 (KeywordAppraisalEngine) | ✓ unchanged |

No weight configs added after first execution (prohibited by Add. J).

---

## Hj1 Verdict: FAIL

**Hj1 (pre-registered):** For at least one weight config W ∈ {W1…W9} and at
least one category C, `aft_config_W.F1(C) ≥ naive_rag.F1(C)` on the
50-QA-per-category subsample.

**Result:** No AFT weight configuration achieves F1 ≥ naive_rag on any of the
four LoCoMo categories. The gap to naive_rag persists under all tested
`base_weights` configurations.

---

## Aggregate results (200 QA, all 4 categories pooled)

| Config | F1 | judge_acc | Δ F1 vs W0 | Δ F1 vs naive_rag | Pareto vs W0? |
|---|---:|---:|---:|---:|---:|
| W0 | 0.1323 | 0.1950 | — | −0.0769 | |
| W1 | 0.1438 | 0.2150 | +0.0115 | −0.0654 | ✓ |
| **W2** | **0.1765** | **0.2800** | **+0.0442** | −0.0327 | ✓ |
| W3 | 0.1215 | 0.2000 | −0.0108 | −0.0877 | ✓ |
| W4 | 0.1150 | 0.2050 | −0.0173 | −0.0942 | |
| W5 | 0.1318 | 0.2450 | −0.0005 | −0.0774 | ✓ |
| W6 | 0.1216 | 0.2050 | −0.0107 | −0.0876 | ✓ |
| W7 | 0.1587 | 0.2500 | +0.0264 | −0.0505 | ✓ |
| W8 | 0.0996 | 0.1850 | −0.0327 | −0.1096 | |
| W9 | 0.1318 | 0.2200 | −0.0005 | −0.0774 | ✓ |
| **naive_rag** | **0.2092** | **0.3300** | +0.0769 | — | |

*Pareto vs W0: config improves ≥+0.01 on ≥1 category vs W0 without aggregate
regression >0.05 (Add. J §Pareto definition). Note: "Pareto vs W0" is a
separate condition from Hj1 — it measures intra-AFT improvement, not
matching naive_rag.*

---

## Per-category F1 matrix

| Config | multi_hop | temporal | open_domain | single_hop |
|---|---:|---:|---:|---:|
| W0 | 0.1699 | 0.0569 | 0.0945 | 0.2078 |
| W1 | 0.1632 | 0.0498 | 0.0849 | 0.2774 |
| **W2** | **0.2012** | **0.0623** | **0.0991** | **0.3432** |
| W3 | 0.1269 | 0.0346 | 0.0907 | 0.2337 |
| W4 | 0.1249 | 0.0428 | 0.0920 | 0.2003 |
| W5 | 0.1350 | 0.0387 | 0.1153 | 0.2381 |
| W6 | 0.1434 | 0.0467 | 0.0743 | 0.2219 |
| W7 | 0.1700 | 0.0289 | 0.0856 | 0.3502 |
| W8 | 0.1506 | 0.0533 | 0.0910 | 0.1037 |
| W9 | 0.1495 | 0.0461 | 0.0828 | 0.2488 |
| **naive_rag** | **0.2122** | **0.0725** | **0.1310** | **0.4212** |

---

## Per-category Δ F1 vs naive_rag

| Config | multi_hop | temporal | open_domain | single_hop |
|---|---:|---:|---:|---:|
| W0 | −0.0423 | −0.0156 | −0.0365 | −0.2134 |
| W1 | −0.0490 | −0.0227 | −0.0461 | −0.1438 |
| **W2** | **−0.0110** | −0.0102 | −0.0319 | −0.0780 |
| W3 | −0.0853 | −0.0379 | −0.0403 | −0.1875 |
| W4 | −0.0873 | −0.0297 | −0.0390 | −0.2209 |
| W5 | −0.0772 | −0.0338 | −0.0157 | −0.1831 |
| W6 | −0.0688 | −0.0258 | −0.0567 | −0.1993 |
| W7 | −0.0422 | −0.0436 | −0.0454 | −0.0710 |
| W8 | −0.0616 | −0.0192 | −0.0400 | −0.3175 |
| W9 | −0.0627 | −0.0264 | −0.0482 | −0.1724 |

Closest AFT cell: **W2 × multi_hop** (Δ = −0.011). Still below naive_rag.

---

## Secondary finding: intra-AFT improvement

Although Hj1 fails (no config matches naive_rag), tuning `base_weights`
significantly improves over the S1 default (W0):

- **Best overall**: W2 (`[0.50, 0.30, 0.10, 0.05, 0.05, 0.00]`) — high
  semantic + elevated mood, resonance disabled. Aggregate F1 +0.044 vs W0,
  judge_acc +0.085. Closest AFT config to naive_rag across all categories.
- **Best on single_hop**: W7 (`[0.70, 0.10, 0.08, 0.04, 0.04, 0.04]`) —
  very high semantic. single_hop F1 = 0.3502 vs W0 = 0.2078 (+0.135).
- **Pattern**: reducing or eliminating `resonance_boost` and increasing
  `semantic` weight consistently improves factual QA. This is directionally
  coherent with S1's finding (resonance aids affective scenarios but adds
  noise on literal content recall).

These are **exploratory descriptive findings**, not confirmatory claims
(Add. J §Statistical analysis plan: no bootstrap CI at per-category level;
no Holm correction).

---

## Conclusion

**Hj1 FAIL.** `base_weights` tuning alone cannot close the AFT vs naive_rag
gap on LoCoMo factual QA. The per-task tuning line for `base_weights` is
closed as an honest negative.

Per Add. J §Reporting rules for Hj1 FAIL: the inability to match naive_rag
via weight tuning reinforces the S1 negative result interpretation —
**AFT's affective machinery introduces noise on literal content recall
tasks regardless of how the retrieval weights are configured** within the
tested range.

Future architectural directions that remain open (outside the scope of
`base_weights` tuning):
- Per-category weight routing (adaptive configuration at retrieval time)
- Selective affect suppression on queries classified as factual/literal
- Dual-path architecture with factual/affective branching at the query level

A full-N (1986 QA) replication of W2 is **not recommended** per Add. J §Hj1:
the full-N replication was conditional on Hj1 PASS. Since Hj1 FAIL, this
line is closed.

---

## Files

| File | Description |
|---|---|
| `benchmarks/locomo/pareto_results.json` | Full results: predictions + scores per config + Pareto table |
| `benchmarks/locomo/pareto_results.md` | Human-readable Pareto table |
| `benchmarks/locomo/pareto_runner.py` | Runner implementing this protocol |
| `benchmarks/preregistration_addendum_j.md` | Frozen pre-registration (commit `7bcd663`) |
