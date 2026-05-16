# Multilingual Follow-up: Cross-Embedder Robustness (Italian Corpus)

> **Update 2026-05-16:** A pre-registered French slice (Hm1, Addendum M) confirms
> the cross-language effect on me5 at N=120 (Δ=+0.18, p<0.0001, g=0.424, Branch A PASS).
> Sister closure: [`benchmarks/preregistration_addendum_m_fr_closure.md`](../../benchmarks/preregistration_addendum_m_fr_closure.md).

This page documents the cross-embedder analysis of the Italian multilingual slice
(Addendum H). It complements the primary Italian results reported in §G6 of the
paper and in `preregistration_addendum_v2.md` (Addendum B).

---

## Summary finding

The AFT ranking advantage on `hit@k` in the Italian corpus **persists across
embedder choices** — from an EN-centric SBERT backbone (`all-MiniLM-L6-v2`) to a
genuinely multilingual model (`intfloat/multilingual-e5-small`). The delta is
consistent at ≈+0.15–0.16; both results are statistically significant.

Low absolute accuracy in the Italian SBERT condition is a backbone artefact. Once
a multilingual embedder is used, absolute accuracy rises for both AFT and
`naive_cosine` by similar margins, confirming that the improvement comes from
the embedder — and the AFT delta is not erased by it.

---

## Cross-embedder results table

| Embedder | System | top1 [95% CI] | hit@k [95% CI] |
|---|---|---|---|
| SBERT `all-MiniLM-L6-v2` (EN-centric) | `aft` | 0.24 [0.15, 0.34] | 0.34 [0.24, 0.44] |
| SBERT `all-MiniLM-L6-v2` | `naive_cosine` | 0.15 [0.07, 0.24] | 0.19 [0.11, 0.28] |
| `multilingual-e5-small` | `aft` | **0.29** [0.20, 0.39] | **0.42** [0.31, 0.54] |
| `multilingual-e5-small` | `naive_cosine` | 0.21 [0.12, 0.30] | 0.26 [0.17, 0.36] |

N = 80 queries, 20 Italian scenarios, bootstrap n = 2000, seed = 0.

### AFT vs naive_cosine delta

| Embedder | Metric | Δ | 95% CI | p | d |
|---|---|---|---|---|---|
| SBERT | top1 | +0.09 | [0.00, 0.18] | 0.074 (NS) | 0.22 |
| SBERT | hit@k | **+0.15** | **[0.08, 0.24]** | **0.0005** | 0.41 |
| me5-small | top1 | +0.08 | [-0.02, 0.18] | 0.154 (NS) | 0.17 |
| me5-small | hit@k | **+0.16** | **[0.06, 0.26]** | **0.001** | 0.35 |

---

## Interpretation

### 1. AFT advantage is not a SBERT artifact

Both embedder runs yield a positive and statistically significant AFT advantage on
`hit@k`. The delta (≈+0.15–0.16) and effect size (d ≈ 0.35–0.41) are consistent
across embedders. This rules out the possibility that the Italian SBERT hit@k
finding was specific to the EN-centric backbone amplifying or suppressing the
affective signal disproportionately.

### 2. SBERT all-MiniLM-L6-v2 is the Italian accuracy bottleneck

Switching to `multilingual-e5-small` lifts absolute `top1_accuracy` for both
systems (+0.05 for AFT, +0.06 for naive) and `hit@k` (+0.08 for AFT, +0.07 for
naive). The AFT delta remains flat, which means:

- The embedder provides better Italian semantic alignment (absolute lift)
- The affective layer adds value on top of whatever the embedder provides (delta stable)

### 3. top1 advantage remains non-significant across embedders

Neither embedder condition shows a significant `top1_accuracy` advantage. The
confidence intervals for the SBERT condition touch zero; for me5-small, the CI
crosses zero. This pattern is consistent with the G4/G5 English results, where
`top1` is harder to move than `hit@k` in the realistic replay paradigm.

### 4. Context: English vs Italian

The English v2 controlled results are substantially higher (top1 ≈ 0.70 for AFT
on N=200, SBERT). The Italian gap is expected: the Italian corpus is smaller
(N=80), the SBERT backbone is EN-centric, and the Italian keyword appraisal rules
have lower coverage than their English counterparts. The cross-embedder analysis
isolates the backbone contribution and confirms the AFT layer generalises.

---

## Limitations

- N=80 gives moderate power. Effect sizes for `top1` (d ≈ 0.17–0.22) are below
  the threshold where N=80 is adequately powered at α=0.05. The top1 NS finding
  should be interpreted as "not enough evidence to reject H₀", not as absence of
  effect.
- `multilingual-e5-small` is still a relatively small model (118M params). Larger
  multilingual models (e.g., `multilingual-e5-large`, `BGE-M3`) may close the
  gap with the English baseline further.
- The Italian keyword appraisal rules in `make_multilingual()` cover common
  emotion categories but are narrower than the English set. An LLM-backed
  appraisal engine in Italian may change the picture.
- A Spanish slice (Hd2_ES, `realistic_recall_v2_es.json`, 20 scenarios / 80 queries)
  was added in v0.8.2 (commit `898e132`). SBERT PASS Δ=0.138 p=0.045 d=0.233;
  me5 FAIL Δ=0.113 p=0.110 d=0.189.
- A French slice (Hm1, Addendum M, 2026-05-16) confirms the effect on me5 at N=120
  (Δ=+0.18, p<0.0001, g=0.424, Branch A PASS). See
  `benchmarks/preregistration_addendum_m_fr_closure.md`.

---

## Branch C closure — Power top-up to N=120 (2026-05-07)

A pre-registered power top-up extended both IT and ES datasets to N=120 (30 scenarios)
and re-ran the me5 runner. Pre-registration committed before dataset generation or
benchmark execution to prevent data-snooping.

| Language | N=80 Δ | N=80 p | N=80 d | N=120 Δ | N=120 p | N=120 d | Branch |
|---|---|---|---|---|---|---|---|
| Italian (me5) | +0.163 | 0.012 | 0.290 | **+0.058** | **0.276** | **0.105** | C |
| Spanish (me5) | +0.113 | 0.110 | 0.189 | **0.000** | **1.000** | **0.000** | C |

**Decision: Branch C (FAIL-FAIL).** Neither language reaches significance at the
pre-declared N=120 power target. Cross-language evidence is limited to:

- Italian hit@k advantage (SBERT + me5, N=80, d≈0.35–0.41) — significant but a
  different metric than the headline top1 used for Hd2.
- Spanish-SBERT top1 (N=80, Δ=+0.138, p=0.045, d=0.233) — directional positive,
  single result, not power-replicated.

The headline EN advantage (SBERT d=0.49, e5 d=0.31, N=200) is unaffected.

**Update 2026-05-16 (Addendum M):** A pre-registered French slice at the same power
target (N=120, me5) confirms the cross-language effect: Δ=+0.18, p<0.0001, g=0.424
(Branch A PASS). Cross-language evidence is now **two-positive**: Spanish-SBERT
exploratory (N=80) and French-me5 confirmatory (N=120). Italian and Spanish me5 power
top-up results remain unchanged (FAIL at declared power, different dataset ecology).

Full closure: `benchmarks/preregistration_addendum_hd2_powertopup_closure.md`.

---

## Audit status

This analysis closes audit priority **(6) Multilingual slice (Addendum B)** from
[`audit_2026-04.md`](audit_2026-04.md). The slice has been executed on two
embedder configurations, and the power top-up has been run (Branch C closure, 2026-05-07).

**Claim update (2026-05-16):** Cross-language claims scoped to "English (robust,
multi-embedder); Spanish exploratory positive (SBERT N=80); French confirmatory
positive (me5 N=120, Addendum M, Branch A PASS); Italian and Spanish me5 FAIL at
declared power."

---

## Source files

- `benchmarks/preregistration_addendum_h.md` — Addendum H formal write-up
- `benchmarks/realistic/results.v2_it.sbert.{json,md}` — SBERT results
- `benchmarks/realistic/results.v2_it.me5.{json,md}` — me5 results
- `benchmarks/realistic/runner.py` — benchmark runner
- `benchmarks/datasets/realistic_recall_v2_it.json` — Italian dataset
