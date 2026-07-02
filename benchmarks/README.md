# Benchmarks — pre-registered evidence programme

This directory holds the pre-registered study documents (`preregistration_*.md`, one
pre-registration + one closure per executed study), the benchmark harnesses (one
subdirectory per benchmark family), and the vendored datasets (`datasets/`, with
licensing in `datasets/README.md`).

Every scored study is **committed before execution** (pre-registration integrity); the
closure records realized numbers, the pre-declared verdict, and propagation targets.
Canonical claim wording lives in
[`docs/research/claim_validation_matrix.json`](../docs/research/claim_validation_matrix.json);
the reader-friendly ladder is
[`docs/research/index.md`](../docs/research/index.md) and
[`docs/research/09_current_evidence.md`](../docs/research/09_current_evidence.md).

## Addenda index (chronological)

| Addendum             | Topic                                                         | Verdict                                                                                | Closure                                                                                                   |
| -------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| A | Appraisal-confound ablation (Ha2: keyword appraisal vs cosine) | **FAIL** (Δ=−0.39 — keyword appraisal collapses the advantage) | `preregistration_addendum_a_closure.md` |
| B/C | Cross-language replication slices (IT me5, ES SBERT) | PASS (IT Δ=+0.163 p=0.012; ES Δ=+0.138 exploratory) | `preregistration_addendum_bc_closure.md` |
| S1                   | LoCoMo external QA (Gate 1)                                   | **FAIL** (F1 0.168 vs 0.271)                                                           | `preregistration_s1_closure.md`                                                                           |
| S2 | Realistic replay v2 vs cosine (H3) | **PASS** (Δ=+0.205 [0.150, 0.265], SBERT) | `preregistration_s2_closure.md` |
| v2 / v2-SOTA         | Realistic replay v2 + Mem0/LangMem baselines                  | PASS (oracle-affect regime)                                                            | `preregistration_addendum_v2.md`, `preregistration_addendum_v2_sota.md`                                   |
| F | Dual-path vs synchronous keyword appraisal (Hf1) | **PASS** | `preregistration_addendum_f_closure.md` |
| G                    | Architecture vs cosine, LLM appraisal, affect-free data (Hg1) | **FAIL** (Δ=−0.010)                                                                    | `preregistration_addendum_g_closure.md`                                                                   |
| Hd2 (+ power top-up) | v2 generalization + N=200 top-up                              | PASS                                                                                   | `preregistration_addendum_hd2_closure.md`, `preregistration_addendum_hd2_powertopup_closure.md`           |
| H                    | Multilingual IT (SBERT/me5)                                   | mixed — see closure                                                                    | (doc: `docs/research/12_multilingual_followup.md`)                                                        |
| I                    | Resonance amplification / Hi3                                 | Hi3_recency PASS, Hi3_arc **falsified**                                                | `preregistration_addendum_i_closure.md`                                                                   |
| J                    | LoCoMo per-task weight Pareto sweep (Hj1)                     | **FAIL** (all 10 configs)                                                              | `preregistration_addendum_j_closure.md`                                                                   |
| K                    | DailyDialog ecological replication (Hk1)                      | **FAIL** (Δ=−0.008)                                                                    | `preregistration_addendum_k_dailydialog_closure.md` (retry prereg: `preregistration_addendum_k_retry.md`) |
| L                    | Query-type routing on LoCoMo (Hl1)                            | **FAIL**                                                                               | `preregistration_addendum_l_query_routing_closure.md`                                                     |
| M                    | French realistic replay (Hm1)                                 | **PASS** (Δ=+0.18, g=0.42)                                                             | `preregistration_addendum_m_fr_closure.md`                                                                |
| N                    | Appraisal prompt calibration                                  | **FAIL** (reverted)                                                                    | `preregistration_addendum_n_appraisal_calibration_closure.md`                                             |
| O                    | SEC→affect mapping recalibration (M1)                         | PASS (held-out)                                                                        | `preregistration_addendum_o_mapping_recalibration_closure.md`                                             |
| P                    | Hg1 re-run, leakage-free, recalibrated (Hp1)                  | **FAIL** (Δ=−0.087; claim falsified)                                                   | `preregistration_addendum_p_hg1_rerun_closure.md`                                                         |
| Q                    | Affect-aware gating / routing closure (Hq1–Hq5)               | **FAIL** (gating = safe wrapper only)                                                  | `preregistration_addendum_q_affect_gating_closure.md`                                                     |
| S3                   | Layer ablation at power                                       | per-layer not isolatable                                                               | `preregistration_addendum_s3_closure.md`                                                                  |
| R                    | Downstream encode→retrieve→generate→judge (Hr1/Hr2)           | **PASS** (judge-acc 0.595 vs 0.440; oracle-affect regime)                              | `preregistration_addendum_r_downstream_closure.md`                                                        |
| S                    | Human-gold appraisal vs EmoBank (A5)                          | valence human-validated (r=0.70); arousal/dominance weak                               | `preregistration_addendum_s_human_gold_appraisal_closure.md`                                              |
| U                    | Circularity audit of v2                                       | ~62.5% AFT-favorable by construction                                                   | `preregistration_addendum_u_circularity_audit_closure.md`                                                 |
| V                    | Direct-VAD appraisal estimator                                | **PASS** (r=0.79/0.58/0.43; shipped opt-in)                                            | `preregistration_addendum_v_direct_vad_closure.md`                                                        |
| T                    | Retrieve-time query appraisal (Ht1)                           | **PASS** (Δ=+0.115; ~59% oracle recovery, production-reachable)                        | `preregistration_addendum_t_query_appraisal_closure.md`                                                   |
| T2A                  | Naturalistic re-test on DailyDialog (Ht2a)                    | **FAIL** (regime-bound)                                                                | `preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`                                    |
| W                    | Affine arousal calibration                                    | ADOPTED (measurement-only; library integration declined)                               | `preregistration_addendum_w_arousal_calibration_closure.md`                                               |
| X                    | Third-party retrieval, MADial-Bench (Hx1)                     | **FAIL, inverted** (Δ=−0.083; construct boundary: counter-congruent supportive recall) | `preregistration_addendum_x_madialbench_third_party_closure.md`                                           |

Unexecuted pre-registrations on file: `preregistration_addendum_k_retry.md` (Hk1
affective-trajectory retry at N≥120), `preregistration_addendum_v3.md`.

## Harness directories

`fidelity/` (127 psychological-invariant tests, `make bench-fidelity`) · `perf/` ·
`comparative/` (+SOTA) · `realistic/` (v1–v5 datasets) · `ablation/` · `locomo/` ·
`dailydialog/` (Hk1 + T2A) · `downstream/` (Addendum R) · `human_gold_appraisal/`
(Addendum S) · `circularity_audit/` (U) · `appraisal_vad/` (V) · `arousal_calibration/`
(W) · `query_appraisal/` (T) · `madialbench/` (X) · `appraisal_quality/` ·
`appraisal_calibration/` · `appraisal_confound/` · `appraisal_diagnostics/` ·
`human_eval/` (Gate 2 kit, unrun — issue #27) · `common/` (shared statistics) ·
`datasets/` (licensing in its README).
