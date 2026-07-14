# Closure Addendum Z — Hz: held-out learned retrieval profile (learning-to-rank)

**Status:** EXECUTED (2026-07-14) — **Branch B: Hz1 FAIL (0/3 break corpora), Hz2 PASS.** No held-out
learned linear affect profile beats cosine on any third-party corpus; the provenance/construct
boundary is hardened to its strongest measured form.
**Pre-registration:** `preregistration_addendum_z_learned_profile.md` (committed pre-run) · harness
`benchmarks/common/ltr.py` + `benchmarks/learned_profile/` (dry-validated on all 4 corpora before
the scored run; zero `src/` change).
**Run:** `make bench-z-profile`, 4 corpora, k=5 cross-fitting, paired bootstrap n=10,000 seed=0,
Holm m=3 (Hz1 break family) + non-inferiority ε=0.02 (Hz2 preserve). Appraisals reused from
committed corpus `results.json` (deterministic direct-VAD); `aft_learned` differs from
`aft_query_appraised` only in the weight vector. 1 transient LLM 503 on `realistic_recall_v2`
fell back to neutral appraisal (logged warning; not systematic).
**Artifacts:** `benchmarks/learned_profile/results.json` / `results.md`

---

## Verdict — Branch B (boundary confirmed at strongest form)

Confirmatory family: Hz1 = {MADial-Bench, ES-MemEval, DailyDialog} `aft_learned` (held-out) >
`naive_cosine`, Holm m=3, one-tailed; Hz2 = `realistic_recall_v2` non-inferiority
(`aft_learned` ≥ `aft_fixed` − 0.02).

| Corpus | Metric | n | cosine | aft_fixed | aft_learned | learned−cosine | p_holm | Hz1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MADial-Bench (X) | nDCG@5 | 160 | 0.304 | 0.166 | 0.319 | +0.015 [−0.008, +0.040] | 0.314 | FAIL |
| ES-MemEval (X2) | u_nDCG@4 | 1133 | 0.284 | 0.083 | 0.269 | −0.015 [−0.023, −0.006] | 0.9997 | FAIL |
| DailyDialog (T2A) | top1 | 396 | 0.220 | 0.237 | 0.227 | +0.008 [−0.045, +0.061] | 0.811 | FAIL |
| `realistic_recall_v2` | top1 | 200 | 0.325 | 0.240 | 0.425 | +0.100 [+0.060, +0.145] | — | Hz2 PASS |

**Hz1:** 0/3 break corpora PASS (none Holm-adjusted p<0.05 with held-out Δ>0) → **Branch B.**
**Hz2:** `aft_learned` vs `aft_fixed` Δ=**+0.185** [+0.120, +0.250], p=0.0000 → non-inferiority
PASS; learning **improves** the on-regime curated advantage rather than sacrificing it.

The pre-registered ex-ante expectation (Branch B) is confirmed: even the held-out optimal _linear_
combination of the 6 AFT retrieval signals does not beat cosine on third-party gold where the
fixed profile failed. The boundary upgrades from "these fixed weights fail" to "no learned linear
affect profile generalizes out-of-sample here."

## Learned weights and interpretability (descriptive, non-gating)

Mean fold weights `w̄` (s1 semantic, s2 mood congruence, s3 affect proximity, s4 momentum, s5
recency, s6 resonance):

| Corpus | w̄ (held-out) | s2 sign | gen. gap (in→held-out) |
| --- | --- | --- | ---: |
| MADial-Bench | [0.33, −0.01, 0.04, 0, 0.04, 0.04] | **negative** | +0.008 |
| ES-MemEval | [0.35, −0.01, 0.04, 0, 0.02, 0.06] | **negative** | −0.002 |
| DailyDialog | [0.01, 0.02, 0.01, 0, −0.005, −0.016] | non-negative | +0.025 |
| `realistic_recall_v2` | [0.38, −0.10, 0.19, 0, 0.12, −0.03] | **negative** | 0.000 |

On MADial and ES-MemEval the ranker recovers **negative s2** (counter-congruent / support-mode
direction), as the X theory residual predicted — yet held-out learned still does not beat cosine
(MADial: marginal +0.015, ns; ES-MemEval: significantly below cosine). On curated, semantic (s1)
and affect proximity (s3) dominate; mood congruence is negative but the profile strongly beats
both fixed and cosine.

**Learned vs fixed (held-out):** large gains on third-party (+0.153 MADial, +0.186 ES-MemEval, both
p=0.0000) show the _fixed_ weight vector was mis-specified — but mis-specification is not
sufficient: the optimally-weighted linear profile still cannot close the cosine gap.

## Interpretation

Addendum Z tests the last untested lever behind every third-party FAIL: not hand-authored weight
grids (Addendum J) but a **data-fit linear ranker** evaluated strictly out-of-sample via k-fold
cross-fitting. Cross-fitting resolves the circularity that kept a support-mode profile
unscheduled — this study asks whether generalization succeeds, not whether in-sample fit does.

**Branch B sharpens the operative boundary.** The third-party failures are not explained by a
single bad fixed vector alone. Where gold is counter-congruent (X) or affect-orthogonal (X2),
even a held-out learned linear combination of mood-congruence, affect-proximity, and the other
AFT signals cannot beat semantic-only retrieval. The on-regime curated advantage is not only
preserved but **amplified** under learning (+0.100 vs cosine, +0.185 vs fixed), confirming the
affect signals carry real discriminative information where the gold relation is
affect-conditioned — and that information is linearly usable there.

**What Branch B does not close.** Non-linear rankers, per-query adaptive weighting, and a
production `fit_retrieval_weights()` API remain out of scope (Branch-A follow-up only). The
support-mode retrieval profile named in Addendum X is partially reflected in negative learned s2
but did not translate into a held-out win.

## Follow-ups (not scheduled here)

- **`fit_retrieval_weights()` src API:** pre-registered as a Branch-A follow-up only; with
  Branch B there is no corpus-level PASS to promote.
- **k ∈ {5, 10} sensitivity:** secondary and non-gating per the prereg; the primary k=5 verdict
  stands.
- **Replication under alternate fold partition:** required only for Branch A; not applicable.

## Propagation

- `docs/research/08_limitations.md` §2.4 — Addendum Z update paragraph (boundary hardened:
  fixed-weight failure → no held-out learned linear profile generalizes on third-party corpora).
- `docs/research/claim_validation_matrix.json` — `cross_domain_affect_replication` wording +
  evidence extended with Addendum Z; mirrored in `docs/research/09_current_evidence.md`.
- `benchmarks/README.md` (addenda table row Z), `docs/research/index.md` (ladder),
  `CHANGELOG.md`, `ROADMAP.md`, `CLAUDE.md`.
