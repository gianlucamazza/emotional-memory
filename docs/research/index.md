# Research Foundations

The library's theoretical basis is documented in the following research papers:

- [Philosophical Foundations](01_foundations.md) — From Greek origins to enactivism
- [Neuroscience](02_neuroscience.md) — Emotional memory from a cognitive neuroscience perspective
- [Computational Models](03_computational_models.md) — Survey of computational emotion models
- [State of the Art](04_state_of_art.md) — LLM memory systems and affective computing
- [Design Principles](05_design_principles.md) — Affective Field Theory design rationale
- [Bibliography](06_bibliography.md) — Full reference list
- [Related Work](07_related_work.md) — 29-system comparative table positioning AFT
- [Limitations](08_limitations.md) — What is validated today, and what is not
- [Current Evidence](09_current_evidence.md) — Study ladder and claim/evidence matrix
- [Scientific Quality Bar](10_scientific_quality_bar.md) — Pre-committed gates before upgrading public claims
- [Dominance Design (G7)](11_dominance_design.md) — Option A: CoreAffect 3D design note (shipped v0.8.2)
- [Multilingual Follow-up (Addendum H + M)](12_multilingual_followup.md) — Cross-embedder robustness: Italian AFT advantage stable across SBERT and me5-small; French Hm1 PASS (Addendum M, 2026-05-16)
- [Addendum L — Query-Type Routing](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_l_query_routing_closure.md) — Closed-loop heuristic routing on LoCoMo: Hl1 Branch B FAIL (smoke test, 2026-05-19)
- [Addendum R — Downstream task (Hr1/Hr2)](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_r_downstream_closure.md) — encode→retrieve→generate→judge on realistic_recall_v2 (N=200, oracle affect): AFT judge-accuracy 0.595 vs cosine 0.440 (Δ=+0.155, p<0.001) — **PASS**, bounded to the oracle-affect regime (2026-06-26)
- [Addendum S — Human-gold appraisal](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_s_human_gold_appraisal_closure.md) — LLM appraisal vs EmoBank human VAD (N=300): valence human-validated r=0.70 [0.66, 0.75]; arousal/dominance weak; +0.15 bias persists; keyword engine not validated (2026-06-26)
- [Addendum U — Circularity audit](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_u_circularity_audit_closure.md) — realistic_recall_v2 decomposition: ~62.5% of queries are AFT-favorable by construction; the advantage concentrates there and is null on the affect-neutral remainder (2026-06-27)
- [Addendum V — Direct-VAD appraisal](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_v_direct_vad_closure.md) — direct-LLM valence/arousal/dominance beats the Scherer SEC→projection on EmoBank human gold (valence r=0.79, arousal r=0.58; Hv1/Hv3 PASS); shipped as opt-in `DIRECT_VAD_SCHEMA` (2026-06-27)
- [Addendum T — Retrieve-time query appraisal](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_t_query_appraisal_closure.md) — appraising the query at retrieve-time (no oracle) beats cosine on curated realistic_recall_v2 (Ht1 PASS, Δ=+0.115, ~59% of the oracle advantage recovered); production-reachable, bounded to the affect-discriminative regime (2026-06-27)
- [Addendum T2A — Naturalistic query-appraisal re-test](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md) — the same mechanism on naturalistic DailyDialog does **not** beat cosine (Ht2a FAIL, Δ=−0.008, p_holm=1.000); a faithful diagnostic (valence r=0.69, arousal r=0.74) shows this is a regime limit, not an appraisal-quality failure (2026-06-27)
- [Audit (2026-04)](audit_2026-04.md) — Critical self-review of strengths, gaps, and priority order
- `claim_validation_matrix.json` — Canonical machine-readable source for public scientific claims
