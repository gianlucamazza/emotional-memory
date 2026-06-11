# Closure — Addendum Q: Affect-Aware Gating (Hq1-Hq3)

**Status:** CLOSED — pre-specified **Branch C** (Hq1 FAIL); Hq2 PASS retained
**Pre-registration:** `benchmarks/preregistration_addendum_q_affect_gating.md` (frozen
2026-06-11, incl. pre-execution Amendment 1)
**Executed:** 2026-06-11, single run, seed=0, n_bootstrap=10,000, embedder `sbert-bge`,
gpt-5-mini appraisal (shared engine, M1 mapping) — no interim looks, no re-runs.
**Dataset:** `realistic_recall_v5_gate` v1.0.0 (50 scenarios / 200 queries, frozen and
committed pre-run, `aeb9613`); dry-run oracle-equivalence gate passed on all 100
affect-free queries before execution.

---

## Results (N = 200 queries; 100 affective / 100 affect-free)

### System top1 accuracy

| System             |                 full |       affective half |         affect-free half |
| ------------------ | -------------------: | -------------------: | -----------------------: |
| `naive_cosine`     | 0.495 [0.425, 0.565] | 0.140 [0.080, 0.210] | **0.850** [0.780, 0.920] |
| `aft_llm_dual`     | 0.390 [0.325, 0.460] | 0.090 [0.040, 0.150] |     0.690 [0.600, 0.780] |
| `aft_gated_oracle` | 0.450 [0.380, 0.520] | 0.050 [0.010, 0.100] | **0.850** [0.780, 0.920] |
| `aft_gated_llm`    | 0.470 [0.400, 0.540] | 0.090 [0.040, 0.150] | **0.850** [0.780, 0.920] |

### Hypotheses (confirmatory family: Holm m=3, α=0.05, threshold Δ>0.05)

| ID  | Comparison                       | Type         | Result      |      Δ | 95% CI         |    p_one | p_holm |      d |
| --- | -------------------------------- | ------------ | ----------- | -----: | -------------- | -------: | -----: | -----: |
| Hq1 | dual > cosine, affective subset  | confirmatory | **FAIL**    | −0.050 | [−0.11, 0.00]  |   0.0597 | 0.0854 | −0.168 |
| Hq2 | gated_llm > dual, full           | confirmatory | **PASS**    | +0.080 | [0.04, 0.12]   |   0.0003 | 0.0009 | +0.248 |
| Hq3 | gated_llm > cosine, full         | confirmatory | **FAIL**    | −0.025 | [−0.05, 0.00]  | 0.0427\* | 0.0854 | −0.135 |
| Hq4 | gated_oracle vs cosine, full     | exploratory  | FAIL        | −0.045 | [−0.07, −0.02] |   0.0024 |      — | −0.216 |
| Hq5 | gated_llm vs cosine, affect-free | exploratory  | Δ = 0 exact |  0.000 | [0.00, 0.00]   |      0.5 |      — |    n/a |

\* Hq3's nominal p is in the **wrong direction** (Δ < 0): cosine is nominally ahead.

### Per-challenge decomposition (top1)

| System             | semantic_confound | recency_confound | same_topic | affect_tiebreak | arc_blind |
| ------------------ | ----------------: | ---------------: | ---------: | --------------: | --------: |
| `naive_cosine`     |             0.909 |            0.758 |      0.882 |       **0.280** |     0.000 |
| `aft_llm_dual`     |             0.970 |            0.515 |      0.588 |           0.160 |     0.020 |
| `aft_gated_oracle` |             0.909 |            0.758 |      0.882 |           0.100 |     0.000 |
| `aft_gated_llm`    |             0.909 |            0.758 |      0.882 |           0.160 |     0.020 |

### Gate classifier (Hq6, mandatory report)

- `aft_gated_oracle`: accuracy 1.0 (by construction).
- `aft_gated_llm`: accuracy **0.844** — affect_free recall 99/100; affective recall **69/99**
  (30 affective queries misrouted to the cosine path). Asymmetric: trajectory-phrased
  affective queries often read as factual to the gate.
- Dataset note: two `affective_arc_blind` queries from different scenarios are textually
  identical ("Which instant finally broke the long slog?", `q06_*_q4` / `q37_*_q4`) — the
  oracle text→label map therefore has 199 unique keys (labels agree; no effect on routing
  or scoring; symptomatic of the arc_blind under-determination below).

---

## Interpretation (pre-specified Branch C — Hq1 FAIL)

**Hq1 fails: LLM-inferred affect does not beat cosine even on affect-discriminative
queries under state-based signals** (0.090 vs 0.140 on the affective half; on the cleaner
`affect_congruent_tiebreak` type alone, 0.160 vs 0.280). Per the pre-registered Branch C,
the affect-routing line **closes**: no gating or routing scheme can rescue a channel that
underperforms in the very regime it is supposed to win.

**Construction limitation (disclosed, does not rescue the verdict).**
`affective_arc_blind` hit a floor for _every_ system (≤ 0.02 incl. cosine): under the
inherited cross-scenario accumulation protocol, 50 same-shape arcs make a trajectory-only
query ("the moment it finally turned") under-determined — it cannot identify _which_
scenario, and the global mood state cannot either. Hq1's informative evidence is the 50
anchored tiebreak queries, where the affect channel still loses by 12 points. The verdict
stands on those; the arc_blind floor only means the affective-half accuracies understate
everyone equally.

**Hq2 PASS is the salvaged, deployment-relevant positive.** Gating recovers the entire
always-on penalty: the gated arms score **exactly** cosine on the affect-free half
(Hq5 Δ = 0.000 — the Amendment-1 front-router equivalence held live, not just in the dry
run), lifting full-corpus top1 from 0.390 to 0.470 (+0.080, p_holm=0.0009). The
front-router is a _safe wrapper_: an agent that must run AFT should gate it. But it
cannot exceed cosine (Hq3 FAIL; Hq4: even the perfect-gate arm is significantly below,
−0.045), because the gated-on path inherits the channel's deficit.

**The bottleneck is the channel, not the classifier.** Two independent signatures:
(a) the oracle gate (accuracy 1.0) scores _worse_ than the imperfect LLM gate
(0.450 vs 0.470 full) — the LLM gate's 30 affective→cosine misroutes accidentally help
because cosine is better on affective queries too; (b) improving gate accuracy therefore
has negative expected value on this corpus.

**Exploratory observation — engine side effects.** On the affective half the
oracle-gated engine path (0.050) trails always-on dual (0.090) despite identical
configuration there. The Amendment-1 disclosed difference is that gated arms skip engine
retrieval side effects (Hebbian strengthening, reconsolidation) on affect-free queries;
their absence appears to _hurt_ later affective retrievals. CIs overlap — recorded as a
hypothesis, not a finding.

**Why this does not contradict Hd1/Hd2 (oracle-affect PASS).** In the Hd-family
datasets each query carries a `state` field that the runner injects into the engine
_before_ retrieval — the benchmark itself performs the query↔state alignment. Addendum Q
shows that without that injection, the trajectory does not supply the alignment for
free. The oracle-affect boundary is therefore also a **state-injection boundary**: the
documented fact that _the query is never appraised_ (docs/mental_model.md) is now
empirically load-bearing.

## Decision

- Claim `appraisal_llm_real_dual_path` stays **falsified**, scope extended: also on
  affect-discriminative queries under state-based signals (this study), and no
  routing/gating configuration closes the gap (Hq3/Hq4). Positive sub-result recorded:
  gating eliminates the always-on penalty exactly (Hq2 PASS, Hq5 Δ=0).
- The affect-aware **routing line is closed**. Residual hypothesis space (NOT scheduled):
  **retrieve-time query appraisal** — appraise the query itself and add a
  query-affect↔memory-affect proximity signal. That is the only mechanism that could
  answer query-driven affect questions; it is a new architectural signal and requires its
  own pre-registration (Addendum R) if ever pursued.

## Coherence with prior closures

- Consistent with Hg1/Hp1 (cosine ahead on affect-free); extends the falsification to the
  affect-discriminative regime under state-based signals.
- Hq2 echoes Hp3's lesson at a different level: _when_ affect is applied matters more
  than how well it is calibrated (deferral beats sync at encode; gating beats always-on
  at retrieve).
- Hd1/Hd2 (oracle affect + state injection) untouched; their scope is now sharper.

## Artifacts

- Pre-reg: `benchmarks/preregistration_addendum_q_affect_gating.md` (+ Amendment 1)
- Dataset (frozen pre-run): `benchmarks/datasets/realistic_recall_v5_gate.json`
- Generator: `benchmarks/datasets/generate_v5_gate.py`
- Runner: `benchmarks/appraisal_confound/runner_hq.py`
- Results: `benchmarks/appraisal_confound/results.hq.{json,md,protocol.json}`

## Cascade changes

| File                                         | Change                                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `docs/research/claim_validation_matrix.json` | `appraisal_llm_real_dual_path`: append Addendum Q (Hq1/Hq3/Hq4 FAIL, Hq2 PASS, gate 0.844); next_study → retrieve-time query appraisal (not scheduled) |
| `docs/research/09_current_evidence.md`       | Append Addendum Q to the appraisal section                                                                                                             |
| `docs/research/08_limitations.md`            | State-injection boundary paragraph                                                                                                                     |
| `ROADMAP.md`                                 | New closed entry; Addendum P "next angle" pointer resolved                                                                                             |
| `CHANGELOG.md`                               | `[Unreleased]` Research bullet                                                                                                                         |
