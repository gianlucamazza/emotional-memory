# Pre-Registration Addendum P — Closure (Hg1 Re-run, Recalibrated Mapping)

**Status:** FAIL — Hp1 not supported; naive cosine significantly *beats* AFT dual-path
(Δ = −0.087, p = 0.0018)

**Date executed:** 2026-05-31
**Parent pre-reg:** `benchmarks/preregistration_addendum_p_hg1_rerun.md`

> All numbers below are verbatim from `benchmarks/appraisal_confound/results.hg1_v4.json`
> (embedder `sbert-bge`, seed 0, n_bootstrap 10000, dataset `realistic_recall_v4_noAF`
> v1.0.0, gpt-5-mini appraisal). The runner labels the hypotheses `Hg1/Hg2/Hg3`; these are
> the pre-registered `Hp1/Hp2/Hp3` of this addendum, run on the v4 leakage-free dataset.

---

## Background

Addendum G (Hg1) found AFT LLM dual-path appraisal did **not** beat naive cosine on
affect-free retrieval (Δ = −0.010, p = 0.367). WP-1a + Addendum N diagnosed the original
failure as **mis-calibration** of the SEC→valence/arousal mapping; Addendum O recalibrated
that mapping (model M1, merged `main` 9ec4752), cutting held-out affect bias (valence
+0.200 → +0.072, arousal −0.144 → −0.023).

Hp1 is the pre-registered `next_study`: does the recalibrated mapping convert into a retrieval
advantage? Because M1 was calibrated on scenarios drawn from `realistic_recall_v3`, the
original Hg1 set `realistic_recall_v3_noAF` (⊂ v3) cannot be reused without train/test
leakage — so a fresh affect-free set disjoint from v3 (`realistic_recall_v4_noAF`, `pNN_`
namespace, zero overlap on scenario ids / memory ids / event text) was authored and frozen
before the run (author-blind).

---

## Execution

```bash
# dataset frozen + committed first (ids/structure fixed in Python, LLM authors only text):
uv run python -m benchmarks.datasets.generate_v4_noAF --n-scenarios 40

# confirmatory run (recalibrated M1 mapping live in main):
EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS=300 uv run python -m \
  benchmarks.appraisal_confound.runner_hg1 --embedder sbert-bge \
  --dataset benchmarks/datasets/realistic_recall_v4_noAF.json \
  --out-json  benchmarks/appraisal_confound/results.hg1_v4.json \
  --out-md    benchmarks/appraisal_confound/results.hg1_v4.md \
  --out-protocol benchmarks/appraisal_confound/results.hg1_v4.protocol.json
```

Frozen parameters identical to Addendum G: embedder `sbert-bge`, seed 0, n_bootstrap 10000,
threshold Δ > 0.05, α 0.05. Only the mapping (M1) and the dataset (v4, disjoint from v3)
differ. gpt-5-mini appraisal (one LLM call per encoded event for the LLM systems).

---

## Results (N = 160 queries, 40 scenarios)

### System top-1 accuracy

| System | top1 | 95% CI |
|---|---:|---|
| `aft_llm_dual` | 0.800 | [0.737, 0.862] |
| `naive_cosine` | **0.887** | [0.837, 0.931] |
| `aft_neutral`  | 0.744 | [0.675, 0.812] |
| `aft_llm_sync` | 0.287 | [0.219, 0.362] |

### Hypotheses

| ID (pre-reg) | Comparison | Type | Result | Δ | 95% CI | p (1-sided) | Cohen's d |
|---|---|---|---|---:|---|---:|---:|
| Hp1 (Hg1) | dual > cosine | confirmatory | **FAIL** | −0.0875 | [−0.144, −0.031] | 0.0018 | −0.242 |
| Hp2 (Hg2) | dual > neutral | exploratory | PASS | +0.0563 | [0.000, 0.112] | 0.0295 | 0.157 |
| Hp3 (Hg3) | dual > sync | exploratory | PASS | +0.5125 | [0.425, 0.594] | 0.0000 | 0.953 |

---

## Interpretation (pre-specified Branch B — FAIL)

**Hp1 fails, and not as a null.** On the leakage-free affect-free set, naive cosine
(top1 0.887) **significantly outperforms** AFT LLM dual-path (0.800): Δ = −0.087, CI entirely
below zero, p = 0.0018, d = −0.24. The recalibrated mapping (Addendum O) did **not** rescue
affect-free retrieval — relative to Addendum G the gap actually widened (Δ −0.010 → −0.087) and
crossed into significance. On a dataset where semantics alone is highly discriminative (cosine
≈ 0.89), adding the affect channel *degrades* ranking rather than helping. The
architecture-vs-cosine boundary holds, more sharply than before.

**Yet the affect signal is real (Hp2, Hp3).** Two exploratory comparisons isolate *why*:

- **Hp2 (dual > neutral), PASS:** dual (0.800) beats a fixed-neutral affect (0.744),
  Δ = +0.056, p = 0.030. The LLM-inferred affect carries genuine signal — it is better than
  having no affect at all.
- **Hp3 (dual > sync), PASS, large:** deferred dual-path (0.800) massively beats synchronous
  appraisal (0.287), Δ = +0.512, d = 0.95. Synchronous LLM appraisal *collapses* on this set;
  the deferred (two-pass) schedule is essential for the affect channel not to corrupt encoding.

**Reconciling the three.** The affect channel is informative (beats neutral) and the dual-path
schedule is necessary to use it (beats sync by a mile), but the resulting full system still
falls short of pure cosine when the query is affect-free and semantics already suffice. The
honest reading: better-calibrated affect is a **net distractor on affect-free queries** even
though it is not noise. This does not contest oracle-affect results (Hd1/Hd2), where affect is
the discriminating dimension by construction; it bounds the claim to those conditions.

Note on `aft_llm_sync` = 0.287: the synchronous-appraisal collapse is the dominant story behind
Hp3's large effect. It is consistent with Addendum G's Hg3 (dual > sync) PASS and strengthens
it; the dual-path/deferred design choice (LeDoux 1996) is vindicated even as the overall
affect-free advantage is refuted.

---

## Coherence with prior closures

- Consistent with Addendum G (Hg1 FAIL) and Addendum N (prompt calibration FAIL): the
  affect-free architecture advantage over cosine is not established under any mapping; here it
  is significantly negative.
- Hp3 (dual > sync) reaffirms Addendum G's Hg3 PASS — dual-path schedule confirmed.
- Hp2 (dual > neutral) is new positive evidence that the recalibrated LLM affect carries signal
  (just not enough to overtake cosine on affect-free queries).
- Does not touch Hd1/Hd2 (oracle-affect PASS). Addendum O (mapping calibration PASS, held-out)
  stands — Hp1 shows it does not *also* deliver an affect-free retrieval win.

## Artifacts

- Pre-reg: `benchmarks/preregistration_addendum_p_hg1_rerun.md`
- Dataset (frozen, committed pre-run): `benchmarks/datasets/realistic_recall_v4_noAF.json`
- Generator: `benchmarks/datasets/generate_v4_noAF.py`
- Results: `benchmarks/appraisal_confound/results.hg1_v4.{json,md,protocol.json}`

## Claim-matrix + changelog updates

- `docs/research/claim_validation_matrix.json` → `appraisal_llm_real_dual_path`: record the
  leakage-free re-run (Hp1 FAIL, naive cosine significantly ahead; Hp2/Hp3 PASS); retire the
  `next_study` pointer.
- `CHANGELOG.md` `[Unreleased]`.
