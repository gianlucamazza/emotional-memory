# Roadmap

This document describes planned work for upcoming releases. Items are ordered by priority within each milestone. Dates are targets, not commitments.

For already-shipped features see [CHANGELOG.md](CHANGELOG.md).

---

## v0.5.x — Stabilisation (shipped ✓)

Patch releases fixing regressions and improving developer experience. No new APIs.

- [x] Fix SQLiteStore cross-thread safety (`threading.RLock`)
- [x] `SentenceTransformerEmbedder` — first-class embedder, `[sentence-transformers]` extra
- [x] README quickstart with `pip install emotional-memory[sentence-transformers]`
- [x] `CITATION.cff` — Zenodo-ready, GitHub "Cite this repository" button
- [x] Fidelity benchmark table links to source test files
- [x] `docs/research/08_limitations.md` — documented known limits
- [x] Published to PyPI as `emotional-memory==0.5.2`
- [x] Zenodo DOI `10.5281/zenodo.19636356`
- [x] arXiv-style paper 10p (`paper/main.tex`) — 4 figures, comparative + perf tables

---

## v0.6.0 — Discovery & Integration (shipped ✓)

- [x] LangChain adapter (`EmotionalMemoryChatHistory`, `[langchain]` extra + CI job)
- [x] Comparative benchmark vs `naive_cosine`, `recency`, Mem0, LangMem (Letta cloud-only, availability-guarded)
- [x] Dataset `affect_reference_v1` (258 examples)
- [x] arXiv-style paper updated (Zenodo DOI `10.5281/zenodo.19640250`, PyPI `0.6.0`)
- [x] mkdocs-material site source (`docs/`) with async, LangChain, persistence tutorials
- [x] HuggingFace Space `homen3/emotional-memory-demo` live
- [x] `Development Status :: 4 - Beta`

---

## v0.7.0 — Scientific Evidence Push (shipped ✓ 2026-05-02)

The originally-scoped "production readiness" items (Qdrant, Chroma, OpenTelemetry, BYO appraisal) **did not ship in v0.7.0** — they have been moved to v0.9.0. v0.7.0 instead consolidated empirical evidence and architecture attribution.

### Pre-registered evidence programme

- [x] **Gate 1 (external benchmark, LoCoMo)** — runner, hypothesis tests, full N=1986 QA results. **Honest negative**: H1/H2 FAIL (`benchmarks/locomo/`). Claim `locomo_external_qa_negative` added to claim matrix.
- [x] **Gate 3 (architecture attribution)** — Hd1 PASS (Addendum D, seed=1): `aft_noAppraisal` Δ=+0.23 vs `naive_cosine`. Closes the appraisal-confound objection.
- [x] **G4/G5 cross-embedder at N=200** — `realistic_recall_v2` (50 scenarios × 4 challenge types). SBERT Δ=+0.205 (d=0.49), e5-small-v2 Δ=+0.155 (d=0.31). CLOSED.
- [x] **G6 multilingual (Italian)** — `realistic_recall_v2_it.json` on SBERT and `multilingual-e5-small`. hit@k significant on both (p=0.0005 / p=0.001). EN-centric SBERT confirmed as IT accuracy bottleneck.
- [x] **G9 ablations** — `no_reconsolidation` (He2: null), `dual_path` (He1: destructive — keyword-driven), `aft_keyword_synchronous` (Hf1: deferral mitigates synchronous override).
- [x] **`EmotionalMemoryConfig.enable_reconsolidation: bool = True`** — new public flag (sync + async engines).
- [x] **Pre-registration corpus** — addenda v2 (B), v3 (D/E), F (Hf1), H (G6 cross-embedder).
- [x] **`docs/research/audit_2026-04.md`** — running tracker; gates table, gap inventory, claim coherence.

### Paper / artefacts

- [x] arXiv submission bundle ready (`paper/arxiv-submission.tar.gz`, cs.LG)
- [x] Zenodo DOI `10.5281/zenodo.19972285`
- [x] PyPI `emotional-memory==0.7.0`
- [x] Six project-scoped Claude Code skills (`.claude/skills/`)

### Production-readiness items NOT in v0.7.0

Moved to **v0.9.0** below. The dot-release window remains open for paper polish only.

---

## v0.7.x — Paper polish (open — arXiv submission pending, decoupled from version line)

Dot release(s) for the paper bundle, no API changes:

- [x] Footnote linking Addendum H from §Limitations of `paper/main.tex`
- [x] arXiv submission bundle ready (`paper/arxiv-submission.tar.gz`, `make check-arxiv-bundle` enforces freshness)
- [ ] arXiv submission executed (cs.LG, no endorsement) — upload pending (user action)
- [ ] Post-submission: update `release.toml: arxiv_id`, run `sync_release_metadata --from-toml`, refresh README badges

---

## v0.8.x — Evidence programme, closed (2026-04 → 2026-05) ✓

All items in this milestone shipped across v0.8.0–v0.8.3 dot releases.

### Gate evidence — all closed

- [x] **S3 ablation @ N=200** — Ha (no_mood): FAIL; Hb (no_resonance): FAIL (e5 magnitude amplification — decomposed in Add. I, #29 closed); Hc: PASS; Hd2: PASS (EN Δ=+0.125). Closure docs in `benchmarks/`.
- [x] **P2-1 power top-up Hd2 IT/ES to N=120** — Branch C (FAIL-FAIL, 2026-05-07). me5: IT Δ=+0.058 p=0.276, ES Δ=0.000 p=1.00. Cross-language scoped to ES-SBERT N=80 exploratory positive. Closure: `benchmarks/preregistration_addendum_hd2_powertopup_closure.md`.
- [x] **G7 PAD dominance design** — design note `docs/research/11_dominance_design.md` shipped. CoreAffect 3D promotion deferred to v0.10 (back-compat migration required).
- [x] **SSOT automation** — `make bump VERSION=X.Y.Z` (atomic 3-file edit + propagation + preflight), `make check-all`, `sync_release_metadata` covers demo/README drift.

### Gate 2 — Human evaluation (kit shipped, execution deferred to v0.10)

Kit is ready in `benchmarks/human_eval/`. Execution (Prolific/MTurk distribution) moved to v0.10.

### Multilingual breadth — Italian + Spanish shipped; French deferred to v0.10

Italian (G6) and Spanish (Hd2_ES, v0.8.2) slices closed. French extension deferred to v0.10.

---

## v0.9.0 — Production Readiness (shipped, PR #24) ✓

Goal: make the library production-grade for teams running agents at scale.

### Enterprise vector stores

- [x] `QdrantStore` adapter + `[qdrant]` optional extra
- [x] `ChromaStore` adapter + `[chroma]` optional extra
- [x] Both implement the `MemoryStore` protocol; ANN behaviour documented in `docs/api/stores.md`

### Observability

- [x] Optional OpenTelemetry spans on `encode`, `retrieve`, `encode_batch`, `elaborate`, `observe`, `prune`
- [x] `[otel]` optional extra; spans no-op when extra is absent
- [x] (Structured `logging.DEBUG` events on pipeline operations already shipped in v0.2.0.)

### Post-v0.9 follow-ups (under [Unreleased] in CHANGELOG, scoped to v0.10)

BYO appraisal schema (#25) shipped post-v0.9 (commit `57ef515`). LoCoMo per-task Pareto study (#26) executed post-v0.9 (Add. J, Hj1 FAIL). See v0.10 below.

---

## v0.10.0 — Evidence + parametricity (shipped 2026-05-07 ✓)

Collecting items deferred from v0.7–v0.9.

### BYO appraisal schema (shipped, #25) ✓

- [x] `AppraisalSchema` config class — parameterise the Scherer CPM prompt so OCC, GRID, or custom taxonomies can be injected without forking (commit `57ef515`)
- [x] Schema-validated `AppraisalVector` (Pydantic) for non-Scherer outputs (commit `57ef515`)

### LoCoMo per-task Pareto study (Gate 1 follow-up) — CLOSED, #26

- [x] Pre-registration frozen — Addendum J (`benchmarks/preregistration_addendum_j.md`): 10-config weight grid, 200-QA stratified subsample, cost ~$1.
- [x] Implement `benchmarks/locomo/pareto_runner.py` (stratified sampling, 10-config sweep)
- [x] Execute sweep (200 QA × 11 configs including naive_rag baseline)
- [x] Pareto-frontier analysis: **Hj1 FAIL** — no AFT config ≥ naive_rag on any category. Best: W2 aggregate F1=0.1765 vs naive_rag=0.2092. Per-task `base_weights` tuning line closed. See `benchmarks/preregistration_addendum_j_closure.md`.

### Gate 2 — Human evaluation execution (deferred from v0.8)

- [ ] Distribute `benchmarks/human_eval/` packets to 20–30 raters (Prolific or MTurk)
- [ ] Collect `ratings.jsonl`, run `benchmarks/human_eval/pipeline.py`
- [ ] Update `claim_validation_matrix.json` and audit doc

### G7 — PAD dominance (CLOSED, #28 wontfix)

- [x] 3D `CoreAffect` shipped in v0.8.2 (`8b9ddbe`); fidelity test re-enabled. Design note: `docs/research/11_dominance_design.md`.
- Back-compat read for pre-v0.8.2 snapshots intentionally not implemented (see #28 closure comment).

### Sign-reversal experiment (CLOSED, #29)

- [x] Per-challenge decomposition: magnitude amplification, not sign reversal. `semantic_confound` is the dominant driver (e5 Δ=+0.125 vs SBERT Δ=+0.025). See `benchmarks/preregistration_addendum_i.md`.
- [x] **Hi3 confirmatory (CLOSED, 2026-05-06)**: N=500 confirms cross-embedder amplification on `semantic_confound` (Δ=+0.090 [0.030,0.160], d=0.257, Holm-adj p=0.0234, **PASS**). Secondary Hi3_recency PASS (Δ=+0.070); Hi3_arc FAIL (Δ=+0.010). Closure: `benchmarks/preregistration_addendum_i_closure.md`. Mechanism (Hi2: link-set density differentiation) remains exploratory — both embedders saturate top-5 link cap; link-type/strength instrumentation not yet measured.

### Multilingual breadth (deferred from v0.8)

- [x] One additional non-English slice — Spanish (`realistic_recall_v2_es.json`, shipped v0.8.2 as Hd2_ES). Closes #30.
- [x] Extends G6 cross-embedder claim to ≥3 language families — `_figure_multilingual` now shows IT + ES × SBERT + me5.

### P3-1 — DailyDialog ecological replication (CLOSED, Branch B, 2026-05-07)

- [x] Pre-registration: `benchmarks/preregistration_addendum_k_dailydialog.md` (Hk1).
- [x] DailyDialog benchmark infrastructure: persona builder, programmatic query generator (4 types), AFT + naive_cosine adapters, runner with paired bootstrap + Holm m=4.
- [x] **Hk1 FAIL (Branch B)**: N=120 personas, 396 queries, multilingual-e5-small. AFT top1=0.212 vs naive_cosine=0.220 (Δ=-0.008, p_holm=1.000, d=-0.015). Only `affective_trajectory` shows exploratory positive trend (Δ=+0.103, d=0.186, N=39; underpowered). Cross-domain ecological replication not established. Regime-specificity of AFT advantage confirmed consistent with LoCoMo FAIL. Closure: `benchmarks/preregistration_addendum_k_dailydialog_closure.md`.

---

## v0.10.x — Supply-chain & developer-experience hardening (in progress, 2026-05)

Post-v0.10.0 dot-releases shipping CI/CD hardening with no API changes.

### CI / Security (shipped 2026-05-12 ✓)

- [x] **`uv_build` backend** — switched from setuptools to `uv_build` for reproducible wheel builds.
- [x] **`basedpyright`** — secondary type-checker added to CI (`continue-on-error: true` during baseline cleanup).
- [x] **CodeQL SAST** (`codeql.yml`) — scans Python on every push/PR to `main`.
- [x] **Conventional PR-title enforcement** (`pr-title.yml`) — blocks merges that violate Conventional Commits.
- [x] **Codecov configuration** — informational coverage gate (target 90%, threshold 1%).
- [x] **zizmor workflow** (`zizmor.yml`) — static analysis of workflow files with SARIF upload to GitHub Advanced Security.
- [x] **SBOM + SLSA + PEP 740 attestations** in `release.yml` — CycloneDX SBOM, SLSA build provenance, and PEP 740 attestations generated and attested on every PyPI release.
- [x] **zizmor self-audit** — all six workflows SHA-pinned, `persist-credentials: false` everywhere, `permissions: contents: read` default in `ci.yml`, `release.yml` cache-poisoning and template-injection findings resolved. `uv run zizmor .github/workflows/` → clean.
- [x] **`scripts/resolve_version.py`** — version resolver extracted from `release.yml` to eliminate heredoc injection vector.
- [x] **Pre-commit hooks modernised** — upstream pinned hooks (`ruff-pre-commit v0.11.12`, `validate-pyproject v0.23`, `zizmorcore/zizmor-pre-commit v1.22.0`); adds `check-merge-conflict`, `check-case-conflict`, `detect-private-key`, `mixed-line-ending`.

### SSOT tooling (shipped 2026-05-07 ✓)

- [x] **`scripts/check_metadata_ssot.py`** — validates author/license/keywords across pyproject, CITATION.cff, codemeta.json, .zenodo.json; wired into CI `meta-integrity` job.
- [x] **`scripts/check_python_version_consistency.py`** — validates Python floor across ruff, mypy, basedpyright, classifiers, CI matrix.

### Open

- [ ] arXiv submission (cs.LG, no endorsement) — upload pending (user action); see v0.7.x.
- [ ] Gate 2 — Human evaluation execution (Prolific/MTurk distribution of `benchmarks/human_eval/packets.json`). Does **not** block v0.11.0; tracked on v1.0 roadmap.

---

## v0.11.0 — Feature release (shipped 2026-05-19 ✓)

### WS1 / WS2 — Debt closure (completed 2026-05-13)

- [x] **SECURITY.md**: supported-versions table updated (0.11.x current, 0.10.x security-only, <0.10 unsupported).
- [x] **`langmem` extra removed**: no `integrations/langmem.py` existed; mypy overrides and `install-langmem` target cleaned up.
- [x] **`letta_client` mypy override removed**: orphan override eliminated.
- [x] **`basedpyright` now gating**: `continue-on-error: true` removed from CI; type-checker blocks merges on error.
- [x] **Static `__all__` declarations**: optional exports declared upfront; no `reportUnsupportedDunderAll` warnings.
- [x] **`ChromaStore.__len__` cast**: `int(col.count())` satisfies mypy `no-any-return`.
- [x] **Makefile test runner**: `uv run python -m pytest` for correct venv resolution.

### WS4 — Research claim closure (completed 2026-05-13)

- [x] **Hg1 → `falsified`**: LLM dual-path vs cosine on affect-free data (Addendum G). No retry planned.
- [x] **Hi3_arc → `falsified`**: No embedder gap on `affective_arc` (Addendum I). Amplification scoped to semantic/recency channels.
- [x] **Hk1 → `retry_planned`**: `affective_trajectory` sub-claim (d=0.186, N=39) warrants N≥120 retry on an affect-richer corpus.
- [x] Status legend extended with `falsified` and `retry_planned` in `claim_validation_matrix.json`.

### WS3 — New features (WS3a+WS3b closed; WS3c closure pending Addendum L)

- [x] **WS3a** — `integrations/mem0.py`: `EmotionalMemoryMem0Backend` facade (mem0 API surface, no runtime mem0ai dep), `messages_to_content` helper, 49 tests, `docs/tutorials/mem0.md` tutorial. Exported from integrations subpackage and top-level `emotional_memory`.
- [x] **WS3b** — French multilingual slice. Addendum M FR me5 N=120 Branch A PASS (Δ=+0.18 top1 [0.11, 0.26], p<0.0001, Hedges g=0.424, 2026-05-16). `cross_domain_affect_replication` → `controlled_evidence`. See `benchmarks/preregistration_addendum_m_fr_closure.md`.
- [x] **WS3c** — `query_classifier.py` (`HeuristicQueryClassifier`, `LLMQueryClassifier`, `QueryClassifier` protocol, `LOCOMO_ROUTING`); `QueryClassifierConfig` in `retrieval.py`; routing injection in `engine.py` + `async_engine.py`. Addendum L closed (2026-05-19, 200-QA smoke test): **Hl1 Branch B FAIL** (Δ=−0.017 vs W2, below +0.02 threshold and in wrong direction). Hl2 FAIL (Δ=−0.081 vs naive_rag). Hl3 data-collection issue (classifier log bug). Routing ships as optional feature.

---

## v0.11.x+ — Appraisal-quality & boundary research (closed, 2026-05-30 → 2026-07-02)

Post-v0.11.0 dot-release research closing the automatic-vs-oracle appraisal gap. No API changes.

- [x] **Addendum N — prompt recalibration (FAIL, reverted).** Diagnosed the Hg1 null as
      mis-calibration, not blindness (valence Pearson r=0.81). A prompt-only recalibration zeroed
      the valence bias (+0.169→+0.044) but left arousal bias unchanged and regressed the gold set;
      Hn1/Hn2 FAIL, prompt reverted. See `benchmarks/preregistration_addendum_n_appraisal_calibration_closure.md`.
- [x] **Addendum O — mapping recalibration (PASS, calibration only).** Numerically refit the
      Scherer SEC→valence/arousal projection (`_scherer_project`, model M1) on a by-scenario 70/30
      split; held-out valence bias +0.200→+0.072, arousal −0.144→−0.023; Ho1/Ho2 PASS. M1 weights
      live in `main` (#46). A calibration result, not a retrieval result. See
      `benchmarks/preregistration_addendum_o_mapping_recalibration_closure.md`.
- [x] **Addendum P — Hg1 re-run with M1 (FAIL).** Re-ran Hg1 on a leakage-free affect-free
      dataset disjoint from v3 (`realistic_recall_v4_noAF`, 40 scenarios / 160 queries, frozen
      pre-run). Naive cosine _significantly_ ahead: dual-path AFT top1 0.800 vs 0.887 (Δ=−0.0875
      [−0.144,−0.031], p=0.0018, d=−0.242). Exploratory: Hp2 dual>neutral PASS (the affect signal
      is real); Hp3 dual>sync PASS, d=0.95 (deferred dual-path is essential). Claim
      `appraisal_llm_real_dual_path` stays **falsified**; the affect-free architecture-vs-cosine
      line is closed. The "next angle" (affect-aware routing) was executed as Addendum Q —
      see below. See `benchmarks/preregistration_addendum_p_hg1_rerun_closure.md`.
- [x] **Addendum Q — affect-aware gating (Branch C, 2026-06-11).** Pre-registered the
      routing synthesis (Hq1–Hq3, Holm m=3; front-router per Amendment 1) on
      `realistic_recall_v5_gate` (50 scenarios / 200 queries, 100/100 gate-labelled, frozen
      pre-run). **Hq1 FAIL**: LLM-inferred affect loses to cosine on the affective subset
      itself (tiebreak 0.160 vs 0.280); **Hq3 FAIL**, Hq4 — even the oracle-gate arm is
      significantly below cosine (Δ=−0.045). **Hq2 PASS** (+0.080, p_holm=0.0009): gating
      recovers the entire always-on penalty exactly (gated == cosine on affect-free queries,
      Hq5 Δ=0.000) — a safe wrapper, not an advantage. The affect-routing line is **closed**;
      residual hypothesis (not scheduled): retrieve-time query appraisal as a new signal.
      See `benchmarks/preregistration_addendum_q_affect_gating_closure.md`.
- [x] **Addenda R/S — downstream value + human-gold appraisal (2026-06-26).** Addendum R:
      encode→retrieve→generate→judge on realistic_recall_v2 (N=200, oracle affect) — **PASS**,
      AFT judge-accuracy 0.595 vs cosine 0.440 (Δ=+0.155, p<0.001), bounded to the oracle
      regime. Addendum S: LLM appraisal vs EmoBank human VAD (N=300) — valence human-validated
      (r=0.70), arousal/dominance weak, +0.15 bias stands, keyword engine not validated.
      See `benchmarks/preregistration_addendum_r_downstream_closure.md` and
      `preregistration_addendum_s_human_gold_appraisal_closure.md`.
- [x] **Addendum U — circularity audit of v2 (2026-06-27).** ~62.5% of queries are
      AFT-favorable by construction; the advantage concentrates there (Δ=+0.304) and is null
      on the neutral remainder (Δ=+0.013, p=0.63). Headline scoped accordingly.
      See `benchmarks/preregistration_addendum_u_circularity_audit_closure.md`.
- [x] **Addendum V — direct-VAD estimator (2026-06-27).** Direct LLM V/A/D rating beats
      the Scherer SEC→projection on every axis vs EmoBank (valence r=0.79, arousal r=0.58,
      dominance r=0.43); shipped opt-in as `DIRECT_VAD_SCHEMA`.
      See `benchmarks/preregistration_addendum_v_direct_vad_closure.md`.
- [x] **Addendum T — retrieve-time query appraisal (2026-06-27): Ht1 PASS.** Appraising
      the query (direct-VAD, no oracle) beats cosine on realistic_recall_v2 (Δ=+0.115,
      p<0.001; ~59% oracle recovery) — the production-reachable mechanism, shipped as the
      public `query_affect` API. See `benchmarks/preregistration_addendum_t_query_appraisal_closure.md`.
- [x] **Addendum T2A — naturalistic re-test (2026-06-27): FAIL.** The same mechanism on
      DailyDialog does not beat cosine (Δ=−0.008, p_holm=1.000) despite faithful appraisal
      (valence r=0.69, arousal r=0.74) — the T recovery is regime-bound.
      See `benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`.
- [x] **Addendum W — affine arousal calibration (2026-06-28): ADOPTED, measurement-only.**
      Held-out EmoBank fit cuts arousal MAE 0.20→0.04 preserving r; library integration
      evaluated and declined (would compress arousal into [0.45, 0.61], breaking the decay
      floor and s3). See `benchmarks/preregistration_addendum_w_arousal_calibration_closure.md`.
- [x] **Addendum X — third-party retrieval, MADial-Bench (2026-07-02).** First test of the
      query-appraisal mechanism (Addendum T) on a released third-party retrieval-native
      emotional corpus (NAACL 2025, MIT; N=160, oracle-free, harness merged pre-run).
      **Hx1 FAIL, inverted**: cosine significantly ahead (nDCG@5 0.304 vs 0.221, Δ=−0.083
      [−0.123, −0.043], powered negative, MDE 0.051) despite near-perfect appraisal
      (D1 AUC=0.996) and an affect-discriminative corpus (D2=76.9%). Post-hoc: the benchmark
      rewards **counter-congruent supportive recall** (emotion regulation) — a **construct**
      boundary on top of the regime (U/T2A) and provenance bounds. Residuals (not scheduled):
      Addendum X2 on ES-MemEval/EvoEmo (longitudinal QA replication); theory-level
      support-mode retrieval profile from the emotion-regulation literature.
      See `benchmarks/preregistration_addendum_x_madialbench_third_party_closure.md`.

- [x] **Addendum X2 — third-party retrieval, ES-MemEval/EvoEmo (2026-07-07).** The
      replication reserved by the X closure: same oracle-free mechanism on the second
      released third-party corpus (WWW 2026, CC-BY-4.0; N=1,133 in-family QA queries,
      401 session documents, 50-candidate pools, upstream-verbatim metrics; harness
      merged pre-run). **Hx2 FAIL, inverted**: cosine significantly ahead
      (u_nDCG@4 0.284 vs 0.133, Δ=−0.150 [−0.165, −0.136], powered, MDE 0.018) with
      faithful appraisal (D1 AUC=0.950) and an affect-discriminative corpus (D2=63.0%,
      ex-ante low-D2 prior wrong). Post-hoc: failure mode distinct from X — the gold is
      **affect-orthogonal** (content-determined QA; 45.2% of queries affectively
      neutral), not counter-congruent. Provenance bound hardened: positive retrieval
      evidence remains confined to author-crafted corpora; operative boundary = the gold
      relation itself must be affect-conditioned. Residual (not scheduled): a
      query-affect-conditioned gate (engage the affect channel only on affect-carrying
      queries — the untested variant left open by Addendum Q); no further released
      third-party corpora exist today.
      See `benchmarks/preregistration_addendum_x2_esmemeval_third_party_closure.md`.

- [x] **Addendum Y — query-affect-conditioned gate (2026-07-07).** The untested variant
      X2 reserved: appraise the query and route neutral queries (|valence|<0.2) to pure
      cosine, else to the Addendum-T affect-conditioned arm. 4 corpora, Holm m=2, harness
      merged pre-run; the gate arm is an exact per-query selection between the existing
      cosine and aft arms (zero src/ change). **Branch A — Hg1 (recover) + Hg2 (preserve)
      both PASS**: gated>aft +0.076 on ES-MemEval, gated>cosine +0.095 on curated. Durable
      result — the gate recovers **exactly the neutral-query component** of the off-regime
      penalty (~50% on ES-MemEval’s 45.5%-neutral queries; 0% on MADial, 0%-neutral) and
      preserves the on-regime gain: a validated safe wrapper, not a fix for the X/X2
      gold-relation boundary. Follow-up (not scheduled): promote to a production
      `retrieve_query_gated()` src API in its own pre-registered PR.
      See `benchmarks/preregistration_addendum_y_query_affect_gate_closure.md`.
- [x] **Addendum Z — held-out learned retrieval profile (Branch B, 2026-07-14).** The
      untested lever behind every third-party FAIL: they all used a _fixed_ weight vector.
      Z fits a linear pairwise learning-to-rank over the 6 retrieval signals and evaluates
      it strictly out-of-sample via k-fold cross-fitting. **Hz1 FAIL (0/3 break corpora):**
      no held-out learned linear profile beats cosine (MADial Δ=+0.015 p_holm=0.314; ES-MemEval
      Δ=−0.015 p_holm=0.9997; DailyDialog Δ=+0.008 p_holm=0.811). **Hz2 PASS:** curated
      learned top1=0.425 vs fixed 0.240 (+0.185). Negative learned s2 on MADial/ES-MemEval
      did not rescue third-party performance. Boundary hardened: fixed-weight failure → no
      held-out learned linear profile generalizes on third-party gold. Branch-A follow-up
      (`fit_retrieval_weights()` src API) not scheduled.
      See `benchmarks/preregistration_addendum_z_learned_profile_closure.md`.
- [x] **Algorithmic levers (P2) — evaluated 2026-07-08, mostly declined.** Three hand-tuned
      runtime enhancements were assessed against the code and priced honestly; none is opened
      as an engineering PR (same "evaluated & declined" disposition as Addendum W's library
      integration). The recorded verdicts, so a future pass does not silently re-open them:
    - **ACT-R per-trace spacing decay → DECLINED.** The current decay (`decay.py:69-89`) is a
      single-exposure power law with the spacing effect modelled as a scalar exponent damping
      `1/(1+retrieval_boost·n)`; a true multi-trace base-level sum `B_i = ln(Σ_j t_j^-d)` is
      **not** log-log linear and would break the declared fidelity invariant
      `benchmarks/fidelity/test_decay_power_law.py::test_log_log_linearity` (R²>0.99). It would
      add a per-presentation list to the frozen `EmotionalTag` (mirrored in both engines) to
      improve a minor signal (s5 weight 0.10) that is already monotone-correct, with **no**
      bearing on the open X/X2 boundary. Over-engineering of a documented simplification.
    - **Hebbian forgetting / LTD → DEFERRED (research addendum only, not an engineering PR).**
      The only lever with real intellectual merit — the graph today can only strengthen
      (`hebbian_strengthen` does `min(1.0, s+inc)`, `resonance.py:347`), so links grow
      monotonically (bloat). But no harness validates a link-decay rate → tuning it is a
      circularity risk the project deliberately avoids; and it breaks three pinned tests
      (non-co-retrieved link must stay exactly 0.5; fidelity monotonicity; exact spreading
      values 0.8/0.72), needs a new temporal field on the frozen `ResonanceLink` + a new config
      (`hebbian_increment` is floored at `ge=0.0`) + mirroring in both engines. Defensible
      **only** as a pre-registered, opt-in (default OFF) addendum with an explicit
      non-inferiority-on-curated + graph-bloat-reduction hypothesis. Behind Z in priority.
    - **Incremental adjacency cache → DECLINED.** `spreading_activation` rebuilds adjacency
      O(N·links) per retrieve (`resonance.py:280-290`), but the G2 prefilter already caps the
      candidate pool at `top_k·candidate_multiplier` = 5×3 = 15 in production (`engine.py:415`),
      so the rebuild is ~75 ops — not a bottleneck. A cache would speed up only the
      `bench_spreading_activation` microbench (which bypasses the prefilter), at the cost of
      invalidation correctness across encode/Hebbian/delete/prune × sync+async. Optimising a
      microbench is not value.

---

## v0.12.0 – v0.14.0 — Consolidation releases (2026-06-27 → )

- [x] **v0.12.0 shipped 2026-06-27** (DOI 10.5281/zenodo.20959964): `DIRECT_VAD_SCHEMA`
      public API + addenda R/S/U/V/T + `elaborate()` type-guard fix.
- [x] **v0.13.0 shipped 2026-06-27** (DOI 10.5281/zenodo.20962443): public `query_affect`
      API + `retrieve_with_query_appraisal()` + Addendum T2A boundary.
- [ ] **v0.14.0 — bumped on main (#88), release pending.** Metadata/docs/security snapshot
      (Addendum W + V/T-led paper reframe); no `src/` change vs v0.13.0 at bump time. Not yet
      tagged/published — `make release` is the gated user step (coins the Zenodo DOI, tags,
      publishes to PyPI, redeploys the HF Space).

## v1.0.0 — Stability commitment (target: when above is closed)

- [ ] Public-API freeze; semver-stability commitment for `EmotionalMemory`, `AsyncEmotionalMemory`, all `interfaces.py` protocols, the `EmotionalMemoryConfig` tree, and the persistence formats (`AffectiveState.snapshot`, `Memory.model_dump`).
- [ ] Migration guide for v0.x → v1.0.

---

## Contributing

Want to work on something on this roadmap? Open an issue first to discuss scope and approach. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

Items not on this roadmap but worth discussing:

- Persistent memory compression / summarisation
- Cross-agent emotional resonance (shared mood fields)
- Integration with more LLM frameworks (LlamaIndex, CrewAI, AutoGen)
- Real-time streaming encode (partial affective updates)
