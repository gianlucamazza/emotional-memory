# Closure Addendum Y — Hy: query-affect-conditioned gate (safe wrapper + penalty decomposition)

**Status:** EXECUTED (2026-07-07) — **Branch A: both primary hypotheses PASS.** The
query-affect gate is a validated _partial safe wrapper_, and the run yields a clean
mechanistic decomposition of the off-regime penalty.
**Pre-registration:** `preregistration_addendum_y_query_affect_gate.md` (committed pre-run)
· harness PR #111 (merged before the scored run) · adapter-close fix PR #112 (the gate run
uses the corrected direct-VAD ES-MemEval per-query).
**Run:** `make bench-y-gate`, 4 corpora, τ=0.2 primary + {0.1,0.2,0.3} sensitivity, paired
bootstrap n=10,000 seed=0, Holm m=2. ES-MemEval per-query reused from the corrected scored
`benchmarks/esmemeval/results.json` (deterministic direct-VAD; the other three corpora ran
fresh). 1 transient LLM timeout fell back (1/~756 fresh appraisals; not systematic).
**Artifacts:** `benchmarks/gate/results.json` / `results.md` / `results.protocol.json`

---

## Verdict — Branch A (safe wrapper validated)

Confirmatory family {Hg1, Hg2}, Holm m=2, one-tailed:

- **Hg1 (recover, ES-MemEval): PASS.** `gated` > `aft_query_appraised` on u_nDCG@4,
  Δ=**+0.076** [+0.065, +0.087], p_holm=0.0000. The gate improves on always-on AFT by
  routing neutral queries to cosine.
- **Hg2 (preserve, realistic_recall_v2): PASS.** `gated` > `naive_cosine` on top1,
  Δ=**+0.095** [+0.050, +0.140], p_holm=0.0000. The gate retains the full Addendum-T gain
  in the affect-discriminative regime.

Both PASS → the query-affect gate preserves the on-regime advantage **and** directionally
recovers the off-regime penalty, with exact cosine-equivalence on neutral queries by
construction. This is the pre-registered Branch-A outcome.

## The decomposition (the durable result)

The gate routes each query wholly to cosine (if `|appraised valence| < 0.2`) or to the
Addendum-T affect-conditioned arm (otherwise). Per corpus, at τ=0.2:

| Corpus                | Regime              | route→cosine | gated | cosine |   aft | gated−cosine |  gated−aft |
| --------------------- | ------------------- | -----------: | ----: | -----: | ----: | -----------: | ---------: |
| `realistic_recall_v2` | curated (preserve)  |        40.0% | 0.420 |  0.325 | 0.420 |   **+0.095** |     +0.000 |
| DailyDialog (T2A)     | naturalistic sparse |        19.9% | 0.237 |  0.220 | 0.222 |       +0.018 |     +0.015 |
| ES-MemEval (X2)       | affect-orthogonal   |        45.5% | 0.209 |  0.284 | 0.133 |       −0.075 | **+0.076** |
| MADial-Bench (X)      | counter-congruent   |         0.0% | 0.245 |  0.304 | 0.245 |       −0.059 |     +0.000 |

(top1 for the first two; u_nDCG@4 / nDCG@5 for X2 / X.)

**The unifying law: the gate recovers exactly the neutral-query component of the penalty,
and nothing else.** Where the always-on affect penalty exists (cosine > aft), the fraction
recovered tracks the fraction of _harmful_ queries that are affectively neutral:

- **ES-MemEval (affect-orthogonal):** 45.5% of queries are neutral; the gate routes them to
  cosine and recovers **~50%** of the penalty (gated 0.209 sits halfway between aft 0.133 and
  cosine 0.284). It does **not** close the gap — the 54.5% affect-carrying queries still
  misfire on content-determined gold and are un-gateable (gated still −0.075 vs cosine).
- **MADial-Bench (counter-congruent):** **0%** of queries are neutral (distressed users →
  every query is affect-carrying, |valence|≥0.2), so the gate never fires (gated ≡ aft) and
  recovers **0%**. The counter-congruent penalty lives entirely on affect-carrying queries
  the query-affect gate cannot touch.
- **DailyDialog (sparse) & curated:** there is no penalty to recover (cosine ≈ aft, or aft
  ahead), so recovery is undefined; the gate is descriptively best-of-both on DailyDialog
  (+0.018 vs cosine, +0.015 vs aft, neither significant) and fully preserves the gain on
  curated (gated ≡ aft, +0.095 vs cosine).

## τ-sensitivity (robustness)

The verdict holds across τ ∈ {0.1, 0.2, 0.3}. On ES-MemEval, recovery is monotone in τ
(more routing → more recovery: gated−aft +0.057 / +0.076 / +0.095; gated−cosine −0.094 /
−0.075 / −0.056) — never closing the gap, consistent with the un-gateable affect-carrying
residual. On curated, gated−cosine is flat at +0.095 (robust preserve). On MADial, routing
stays ≈0 at every τ (no neutral queries exist to route). τ=0.2 is not a special value; the
mechanism's behavior is a smooth function of it.

## Interpretation (what the gate can and cannot do in production)

The query-affect gate is a **safe wrapper**: it makes AFT provably no-worse-than-cosine on
the neutral-query slice (exact cosine fallback there) while preserving the on-regime gain,
so shipping AFT+gate strictly dominates always-on AFT off-regime. But it is **not** a fix
for the third-party failures: it cannot rescue affect-carrying queries whose gold is
counter-congruent (X) or content-determined (X2's other half). The boundary from Addenda
X/X2 — _the gold relation itself must be affect-conditioned_ — is unchanged; the gate only
removes the self-inflicted penalty on queries that carry no affect to condition on.

## Follow-up (Branch A, NOT part of this study)

Promote the gate to a production API `retrieve_query_gated()` in `src/` (force the weight
vector `[1,0,0,0,0,0]` via `precomputed_weights` in `engine._effective_retrieval_weights`,
bypassing `adaptive_weights` — the Addendum-Q Amendment-1 trap), in its own pre-registered
PR. The harness gate is exact cosine-equivalence by construction; the production version
must replicate that (the audit in the prereg §Decision-rule Branch A confirms the hook).

## Propagation

- `docs/research/claim_validation_matrix.json` — new claim `query_affect_gate` =
  `controlled_evidence` (scoped: harmlessness off-regime + preserved on-regime gain, with
  the explicit ceiling that it recovers only the neutral-query penalty); mirrored verbatim
  in `docs/research/09_current_evidence.md`.
- `docs/research/08_limitations.md` — note that the query-affect gate is a safe wrapper, not
  a fix for the X/X2 boundary.
- `benchmarks/README.md` (addenda table row Y + `gate/` harness index), `docs/research/index.md`
  (ladder), `CHANGELOG.md`, `ROADMAP.md`.
