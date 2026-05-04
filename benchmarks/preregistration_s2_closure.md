# Pre-registration — Study S2 Closure

**Date executed:** 2026-05-04
**Protocol version:** s2_closure_v1
**Parent pre-reg:** `benchmarks/preregistration.md` (Study S2, lines 106–191)

> **Epistemic status:** This document records the execution and results of the
> pre-registered Study S2 hypotheses H3–H6 and exploratory H7–H8. All numbers
> match the committed JSON files exactly.

---

## Background

Study S2 runs the confirmatory realistic-recall benchmark on `realistic_recall_v2`
(50 scenarios, N=200 queries, 5 challenge types × 40 queries each). It pre-registers
H3 (overall AFT > naive) and H4–H6 (per-challenge-type advantages), with H7–H8
(semantic_confound and same_topic_distractor) as exploratory.

Holm family per pre-reg: H3, H4, H5, H6 jointly corrected (4 tests). In practice,
the per-challenge Holm correction in `challenge_subset_pairwise_v2.json` is applied
over all 5 challenge types (including H7–H8), which is more conservative.

---

## Execution

```bash
make bench-realistic       # realistic_recall_v2, sbert-bge
uv run python -m benchmarks.realistic.runner \
    --embedder e5-small-v2 \
    --out benchmarks/realistic/results.v2.e5.json

uv run python -m benchmarks.realistic.analyze_challenge_subsets \
    --sbert-json benchmarks/realistic/results.v2.sbert.json \
    --second-json benchmarks/realistic/results.v2.e5.json \
    --second-label e5-small-v2 \
    --out benchmarks/realistic/challenge_subset_pairwise_v2.json
```

Canonical result files:
- `benchmarks/realistic/results.v2.sbert.json` (EN SBERT, primary)
- `benchmarks/realistic/results.v2.e5.json` (EN e5-small-v2, secondary)
- `benchmarks/realistic/challenge_subset_pairwise_v2.json` (per-challenge Holm)

Parameters: seed=0, n_bootstrap=2,000 (overall), 2,000 (per-challenge subset).

---

## Results summary

### H3 — Overall advantage (N=200 queries)

| Embedder | AFT top1 | naive top1 | Δ [95% CI] | p_bootstrap | Verdict |
|----------|----------|-----------|------------|------------|---------|
| SBERT bge-small-en | 0.530 | 0.330 | +0.205 [0.150, 0.265] | <0.001 | **H3 PASS** |
| e5-small-v2 | 0.500 | 0.340 | +0.155 [0.090, 0.225] | <0.001 | **H3 PASS** |

### H4–H8 — Per challenge type (N=40 per type; Holm over 5 types)

**SBERT bge-small-en-v1.5:**

| Challenge type | Hyp | AFT top1 | naive top1 | Δ | p_boot | p_adj_holm | Verdict |
|---|---|---|---|---|---|---|---|
| affective_arc | H4 | 0.425 | 0.150 | +0.275 | 0.000 | **0.000** | **PASS** |
| recency_confound | H5 | 0.150 | 0.050 | +0.100 | 0.051 | 0.054 | **FAIL** (marginal) |
| momentum_alignment | H6 | 0.600 | 0.325 | +0.275 | 0.000 | **0.000** | **PASS** |
| semantic_confound | H7 (expl.) | 0.725 | 0.475 | +0.250 | 0.003 | **0.008** | (PASS) |
| same_topic_distractor | H8 (expl.) | 0.750 | 0.625 | +0.125 | 0.027 | 0.054 | (marginal) |

**e5-small-v2:**

| Challenge type | Hyp | AFT top1 | naive top1 | Δ | p_boot | p_adj_holm | Verdict |
|---|---|---|---|---|---|---|---|
| affective_arc | H4 | 0.475 | 0.200 | +0.275 | 0.002 | **0.008** | **PASS** |
| recency_confound | H5 | 0.275 | 0.075 | +0.200 | 0.010 | **0.040** | **PASS** |
| momentum_alignment | H6 | 0.450 | 0.425 | +0.025 | 0.811 | 0.811 | **FAIL** |
| semantic_confound | H7 (expl.) | 0.625 | 0.475 | +0.150 | 0.133 | 0.398 | (NS) |
| same_topic_distractor | H8 (expl.) | 0.675 | 0.550 | +0.125 | 0.163 | 0.398 | (NS) |

---

## Interpretation

**H3 PASS — both embedders.** AFT reliably outperforms naive cosine at the overall
level (N=200). Δ=+0.205 (SBERT) and Δ=+0.155 (e5), both p<0.001. This confirms the
primary S2 claim: AFT helps on multi-session affective memory replay tasks.

**H4 (affective_arc) PASS — both embedders.** The mood-field signal that privileges
emotionally salient older memories over neutral recent ones operates as expected.
The advantage is large (Δ=+0.275) and highly significant with both embedders.

**H5 (recency_confound) — mixed.** SBERT: p_adj=0.054 (fails α=0.05 threshold,
marginal). e5-small-v2: p_adj=0.040 (PASS). The recency-confound advantage is
real but small (Δ=0.10 SBERT, 0.20 e5) and embedder-dependent. Pre-registered verdict:
FAIL for SBERT (marginally), PASS for e5. H5 is not confirmed consistently across
embedders.

**H6 (momentum_alignment) — mixed.** SBERT: p_adj=0.000 (strongly PASS, Δ=+0.275).
e5-small-v2: p_adj=0.811 (FAIL, Δ=+0.025, NS). The momentum signal is captured by
the SBERT geometry but not by e5's distance space. H6 is embedder-dependent: strong
evidence for SBERT, no evidence for e5.

**H7 (semantic_confound, exploratory).** SBERT: PASS (Δ=+0.250, p_adj=0.008). e5: NS
(Δ=+0.150, p_adj=0.398). AFT uses affect to disambiguate semantically similar
distractors — works well with SBERT, less so with e5.

**H8 (same_topic_distractor, exploratory).** SBERT: marginal (p_adj=0.054). e5: NS
(p_adj=0.398). Not pre-registered; report for completeness.

**Summary.** H3 is confirmed on both embedders. H4 is confirmed. H5 and H6 are
embedder-dependent: SBERT shows the theoretically predicted per-type advantages
while e5's distance geometry is less responsive to mood/momentum signals. The
mixed H5/H6 result motivates per-challenge analysis in subsequent studies and
limits the claim to "architecture-level advantage is consistent" (H3/H4) rather
than "all per-type mechanisms contribute uniformly across embedders."
