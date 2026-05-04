# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2026-05-04

### Added

- **Multilingual ES — Spanish benchmark** (`realistic_recall_v2_es.json`, 20 scenarios,
  158 memories, 80 queries).  Hd2_ES: sbert-bge **PASS** Δ=0.138 p=0.045 d=0.233;
  me5 borderline FAIL Δ=0.113 p=0.110.  Architecture generalisation confirmed for
  a second non-English language (Spanish) with sbert; me5 marginal.

- **G7 — CoreAffect promoted to 3D PAD space** (valence × arousal × dominance).
  `CoreAffect` now carries a `dominance` field ([0, 1], default 0.5).  This
  makes perceived control a primary retrieval signal (`s3` affect-proximity now
  operates in full PAD space, `_MAX_AFFECT_DIST` → √6).  `AffectiveMomentum`
  gains `d_dominance` / `dd_dominance` derivatives; `AffectiveState` history
  tuples extend to 4-elements.  `AppraisalVector.to_core_affect()` maps
  `coping_potential → dominance`; `MoodField.update()` uses
  `core_affect.dominance` directly instead of the old valence×arousal heuristic.
  `test_dominance_retrieval_gap.py` promoted from xfail to passing.

- `scripts/generate_research_figures.py` — generates benchmark evidence figures
  from committed JSON artefacts without rerunning long studies. Outputs PNGs for
  docs and PDFs for paper use.
- `docs/images/research/` — five evidence figures covering realistic replay v2,
  challenge-type breakdown, S3 ablation, Italian multilingual slice, and the
  LoCoMo negative result.
- `make research-figures` / `make figures` — regenerate research-only or all
  project figures from repo data.
- `scripts/preflight.py` — gate G14 verifies the GitHub→Zenodo webhook is
  disabled before release.  Queries `gh api repos/.../hooks` and fails if any
  active hook points at `zenodo.org`.  Skipped gracefully if `gh` is
  unavailable.  Prevents shadow duplicate deposits like the v0.8.1 incident.
- `scripts/release.py` — Phase 1 anti-shadow guard.  After concept-DOI
  verification, queries the Zenodo public API for published records under the
  concept and aborts if any record matches the current version (other than the
  fresh draft).  Catches duplicate deposits even when both records sit under
  the same concept umbrella.

### Changed

- `README.md`, `docs/research/09_current_evidence.md`, and
  `docs/research/claim_validation_matrix.json` now surface the benchmark
  figures and align current evidence wording with closed S3/Hd2 results.

### Fixed

- `emotional_memory.visualization` now rejects invalid radar inputs and
  unsupported adaptive-heatmap axes instead of rendering misleading figures.

## [0.8.1] - 2026-05-04

Publishing-channel patch: include the paper PDF in the Zenodo deposit and
realign the canonical concept DOI to `10.5281/zenodo.19972258` (the umbrella
that contains the API-driven releases). No code changes.

### Changed

- `release.toml` — `concept_doi` corrected to `10.5281/zenodo.19972258`
  (the previous value `10.5281/zenodo.19972284` was the legacy webhook
  umbrella, frozen at v0.7.0)
- `pyproject.toml` — version `0.8.0` → `0.8.1`
- All SSOT-managed files re-synced: README badge, codemeta.json, paper/main.tex,
  demo/README.md, demo/app.py

### Added

- `scripts/check_release_metadata.py` — `--check-zenodo-remote` flag that
  queries the Zenodo REST API to verify `conceptdoi` matches release.toml
- `scripts/release.py` — Phase 1 guard that fails fast if the reserved
  draft's `conceptdoi` ≠ release.toml `concept_doi`

### Fixed

- v0.8.0 Zenodo deposit was source-only because the bucket was locked after
  the GitHub-webhook deposit; the paper PDF lived only on the GitHub Release
  asset. v0.8.1 ships the paper PDF directly in the Zenodo record.

## [0.8.0] - 2026-05-04

Research milestone: closes Study S3 (layer ablation at power) and Hd2
(Addendum D generalization). Refactors benchmark runners for multi-dataset
support and adds progress bars.

### Added

- `benchmarks/ablation/runner.py` — `--dataset PATH` flag; embedder choices
  extended to `e5-small-v2` + `multilingual-e5-small`; progress bar on variant loop
- `benchmarks/appraisal_confound/runner.py` — `--dataset PATH` flag; embedder
  `multilingual-e5-small`; dynamic hypothesis label (`Hd1`/`Hd2`/`Hd2_IT`)
  derived from dataset name at runtime; progress bars on system + scenario loops
- `Makefile` — `bench-s3-sbert`, `bench-s3-e5`, `bench-hd2-sbert`, `bench-hd2-it-me5`
- `benchmarks/ablation/results.v2.sbert.json` + `results.v2.e5.json` — Study S3 power results
- `benchmarks/appraisal_confound/results.hd2.sbert.json` + `results.hd2_it.me5.json` — Hd2 power results
- `benchmarks/preregistration_addendum_s3_closure.md` — S3 interpretive closure
- `benchmarks/preregistration_addendum_hd2_closure.md` — Hd2/Hd2_IT interpretive closure
- `tqdm>=4.65` added to `[project.optional-dependencies].bench`
- mypy override for `tqdm` / `tqdm.*` in `pyproject.toml`

### Changed

- `benchmarks/ablation/runner.py` — benchmark id now derived from `dataset.name`
  (was hardcoded `ablation_realistic_v1`; now `ablation_realistic_recall_v1`)
- `benchmarks/preregistration.md` — S3 closure footer appended
- `benchmarks/preregistration_addendum_v3.md` — Hd2 closure footer appended
- `docs/research/09_current_evidence.md` — S3 + Hd2 evidence tables appended
- `ROADMAP.md` — S3 + Hd2 milestone items marked `[x]` with date + verdicts
- `tests/test_ablation_runner.py` — updated assertion; added parametrized benchmark id test
- `pyproject.toml` — version `0.7.1` → `0.8.0`

### Research verdicts (S3 + Hd2)

| Study | Hypothesis | Verdict |
|---|---|---|
| S3 | Ha (no_mood < full) | FAIL — both embedders |
| S3 | Hb (no_resonance < full) | FAIL — both embedders (e5: opposite direction, SIG) |
| S3 | Hc (no_appraisal invariant) | PASS — both embedders |
| Hd2 | aft_noAppraisal > naive_cosine, Δ>0.10, v2 EN | PASS (Δ=0.125, p<0.001) |
| Hd2_IT | aft_noAppraisal > naive_cosine, Δ>0.10, v2 IT | PASS (Δ=0.163, p=0.012) |

## [0.7.1] - 2026-05-04

Paper-polish release. No API or behavioural changes; library code is identical
to v0.7.0. Bumps version to provide a canonical Zenodo archive of the corrected
paper bundle (§Limitations integrity fixes + complete pre-registered ablation
disclosure).

### Added

- `release.toml` — single source of truth for release-facing metadata
  (`concept_doi`, `version_doi`, `repo_url`). Edit here, propagate with
  `make sync-metadata`.
- `make sync-metadata` / `make sync-metadata-dry` — propagate `release.toml`
  values to all dependent files offline (`--from-toml`).
- `demo/app.py` — `_ZENODO_CONCEPT_DOI` and `_REPO_URL` module-level
  constants (managed by sync script) replace inline hardcoded strings in
  `_DESCRIPTION` f-string.
- `paper/main.tex` — `\zenodoconceptdoi`, `\zenodoversiondoi`, `\repourl`
  LaTeX commands (managed by sync script) centralise release metadata;
  used in body via `\href` and in the new Addendum H footnote.

### Changed

- `scripts/sync_release_metadata.py` — extended with `--from-toml` flag
  (reads `release.toml` offline); updated patterns for `paper/main.tex`
  `\newcommand` block and `demo/app.py` constants; added `CITATION.cff`
  `version:` field sync.
- `scripts/check_release_metadata.py` — ground truth moved from README
  badge / CITATION.cff to `release.toml`; added `repo_url`, `paper/main.tex`
  macro, and `demo/app.py` constant checks; removed fragile derived-value
  extraction.
- `paper/main.tex` — §Limitations footnote links Addendum H companion
  (`docs/research/12_multilingual_followup.md`) via `\href`; body DOI uses
  `\href{\zenodoconcepturl}{\texttt{\zenodoconceptdoi}}` (macro-based, concept
  DOI corrected from v0.7.0 version DOI back to perpetual concept DOI per
  SSOT contract).
- `ROADMAP.md` — rewritten to reflect actual v0.7.0 contents (scientific
  evidence push: Gate 1 LoCoMo FAIL, Gate 3 Hd1 PASS, G4/G5/G6/G9 closed,
  pre-reg corpus). Production-readiness items (Qdrant, Chroma, OTel, BYO
  appraisal) deferred to v0.9.0. v0.8.0 retargeted to open gates (Gate 2
  human eval, S3@N=200, G7 dominance, multilingual breadth).
- `docs/research/audit_2026-04.md` — Snapshot section updated with explicit
  per-gate status (Gate 1 CLOSED-NEG, Gate 2 OPEN, Gate 3 CLOSED-PASS) and
  pre-registration addenda count (5: B/D/E/F/H).
- `paper/main.tex` — §Limitations extended with two honest disclosure paragraphs:
  (1) "External-benchmark scope" reporting the LoCoMo FAIL (Gate 1 not met,
  F1 0.168 vs 0.271), previously only in the Conclusion; (2) "Component ablations"
  reporting Addendum E pre-registered results (He2 null: removing reconsolidation
  has no effect; He1 rejected: dual-path encoding is destructive at 0.35 vs 0.70).
  Overfull hbox at L177-183 (170 pt, link-types formula) fixed by rewriting
  inline math as `\emph{}` prose.
- `CITATION.cff` — removed empty `orcid: ""` field (no ORCID registered).
- `paper/SUBMISSION.md` — affiliation clarified to "Independent Researcher".
- `paper/main.tex` — §Limitations "Component ablations" rewritten to fix two
  BLOCKER errors introduced in the previous pass: (1) metric label corrected
  from "hit@k" to "top1\_accuracy" (0.70/0.35 are top1, not hit@k); (2) N
  corrected from 200 to 100 (ablation runs on v1.4, not v2). Disclosure
  expanded to cover all pre-registered S3 ablations: Ha (no\_mood), Hb
  (no\_resonance), Hc (no\_appraisal), Hd (no\_momentum) — all null
  (|Δ|≤0.01, p\_adj=1.000) — previously omitted. Addendum F (Hf1: deferring
  keyword appraisal partially recovers signal, Δ=+0.28, PASS) added as the
  nuance that makes He1 honest.
- `paper/main.tex` — §Limitations "External-benchmark scope" expanded to
  report LoCoMo H2 (judge-accuracy 0.279 vs 0.441, FAIL, Δ=−0.159), the
  co-primary hypothesis omitted from the previous pass; N=1540 scored pairs
  added for transparency.
- `paper/main.tex` — §Conclusion Hd1 numbers added as footnote (Addendum D:
  aft\_noAppraisal=0.78 vs naive\_cosine=0.55, Δ=+0.23 [+0.12,+0.34],
  d=0.52, N=100, v1.4, seed=1) — previously Gate 3 was cited as "CLOSED"
  without verifiable numbers.
- `paper/main.tex` — date updated from April 2026 to May 2026 (release
  2026-05-02, consistent with CHANGELOG and CITATION.cff).
- `paper/main.tex` — hash-embedder Δ≈+0.06 claim now cites provenance
  (v1.4 pilot; benchmarks/realistic/results.md).
- `README.md` — "External benchmark" comparison table cell updated from
  "❌ not yet evaluated" to "✅ LoCoMo (FAIL: F1 0.168 vs 0.271)", consistent
  with §Limitations and audit\_2026-04.md.

## [0.7.0] - 2026-05-02

### Added

- `benchmarks/datasets/realistic_recall_v2_it.json` — Italian multilingual slice
  (G6, 20 scenarios, 80 queries, 4 challenge types). SBERT bge-small-en-v1.5
  results: AFT top1=0.24 vs naive_cosine=0.15; hit@k=0.34 vs 0.19
  (Δ=+0.15, p=0.0005, **significant**). multilingual-e5-small results:
  AFT top1=0.29 vs naive_cosine=0.21; hit@k=0.42 vs 0.26
  (Δ=+0.16, p=0.001, **significant**). Embedder swap confirms EN-only embedder
  was the absolute-accuracy bottleneck (naive_cosine top1 +40%); AFT signal
  preserved under both embedders.
- `benchmarks/realistic/results.v2_it.me5.{json,md,protocol.json}` — G6
  Italian multilingual-e5-small benchmark results.
- `make bench-realistic-it-me5` Makefile target + `--embedder multilingual-e5-small`
  runner choice.
- `benchmarks/realistic/results.v2_it.sbert.{json,md,protocol.json}` — G6
  Italian SBERT benchmark results (committed with separate protocol file to avoid
  overwriting English v1 canonical protocol).
- `make_multilingual()` factory in appraisal layer + Italian keyword rules,
  `make bench-realistic-it-sbert` / `make bench-realistic-it-e5` Makefile targets.
  `--out-protocol` flag on IT targets writes to dedicated `results.protocol.v2_it.*.json`
  files (prevents canonical English v1 protocol from being overwritten).
- `benchmarks/appraisal_confound/results.{json,md,protocol.json}` — G3
  evidence committed (2026-04-26, SBERT, N=100, n\_bootstrap=10 000, seed=42):
  aft\_noAppraisal = 0.78 vs naive\_cosine = 0.55 (Δ ≈ +0.23, architecture
  attribution descriptive); Ha2 (aft\_keyword vs naive\_cosine) FAIL Δ = −0.39
  (keyword appraisal destructively overrides preset affect); Hb2 FAIL Δ = −0.62.
- `benchmarks/appraisal_confound/results.confirmatory.{json,md}` — Hd1
  confirmatory run (Addendum D, SBERT, seed=1): aft_noAppraisal=0.78 >
  naive_cosine=0.55, Δ=+0.23, d=0.52. **Hd1 PASS** — AFT architecture advantage
  replicates with new seed. Gate 3 CLOSED.
- `--n-bootstrap` and `--seed` CLI flags on
  `benchmarks/appraisal_confound/runner.py` for reproducibility and sensitivity
  runs; added Hd1 hypothesis test (aft_noAppraisal > naive_cosine, Δ > 0.10).
- `benchmarks/preregistration_addendum_f.md` — pre-registers Hf1 (Addendum F):
  `dual_path.top1 > aft_keyword_synchronous.top1`; tests whether deferring
  keyword appraisal to the slow path mitigates synchronous destructive override.
  Committed before implementation for CONFIRMATORY status.
- `AFTKeywordSynchronousReplayAdapter` in `benchmarks/ablation/runner.py` —
  new ablation adapter that injects `KeywordAppraisalEngine` synchronously at
  `begin_session` (no `elaborate()` call); contrasts with `AFTDualPathReplayAdapter`
  which uses the slow-path `elaborate()`. Closes He1 caveat.
- `benchmarks/ablation/results.sbert.{json,md,protocol.json}` updated to 8
  variants (Addendum F); Hf1 result: dual_path=0.35 > aft_keyword_synchronous=0.07,
  Δ=+0.28, **Hf1 PASS** — deferral partially mitigates synchronous keyword
  destruction. Holm family extended from 6 to 7 comparisons.
- `docs/research/audit_2026-04.md` — Addendum F closed section added with Hf1
  interpretation and updated theory–evidence coherence table.
- `benchmarks/preregistration_addendum_v3.md` — pre-registers Addendum D
  (Hd1: architecture attribution re-framing after Ha2 FAIL, Δ > 0.10 threshold)
  and Addendum E (He1/He2: dedicated ablations for dual-path encoding and
  APE-gated reconsolidation). Committed before execution for CONFIRMATORY status.
- `EmotionalMemoryConfig.enable_reconsolidation: bool = True` — new ablation
  flag; when False, skips the APE-gated reconsolidation window at retrieval time.
  Predictive-learning (`update_prediction`) still runs. Both sync and async
  engines gate the reconsolidation block on this flag.
- `benchmarks/ablation/results.sbert.{json,md,protocol.json}` — G9 confirmatory
  ablation results (SBERT, N=100, seed=0, 7 variants):
  - He2 (`no_reconsolidation`): Δ=0.00, p_adj=1.000 — **FAIL** (null result;
    benchmark doesn't exercise reconsolidation triggers).
  - He1 (`dual_path`): Δ=−0.35, p_adj≈0 — **FAIL** (expected: keyword
    appraisal degrades affect, same destructive-override as G3/Addendum A).
- `benchmarks/ablation/results.{json,md,protocol.json}` — re-generated hash
  sensitivity check with 7 variants (Holm denominator updated to 6).
- `make bench-ablation-sbert` — new Makefile target for paper-canonical SBERT
  ablation run; writes to `results.sbert.{json,md,protocol.json}`.
  `bench-ablation` now explicitly runs the hash embedder (sensitivity only).
- `AFTDualPathReplayAdapter` in `benchmarks/ablation/runner.py` — subclass of
  `AFTReplayAdapter` that injects `KeywordAppraisalEngine` and calls
  `engine.elaborate()` after encoding (slow-path dual-path encoding, He1).

### Fixed

- `benchmarks/appraisal_confound/runner.py`: `paired_bootstrap_diff` returns
  4 values `(diff, lo, hi, p_two_sided)`; runner unpacked 3 → ValueError.
- `benchmarks/appraisal_confound/runner.py`: `ci_payload` keys are
  `ci_lower`/`ci_upper`; markdown renderer used `lo`/`hi` → KeyError.
- `benchmarks/appraisal_confound/runner.py`: Ha2 pass criterion upgraded to
  pre-reg Addendum A spec (Δ > 0.05 practical threshold + one-tailed
  alpha=0.05 via `p_two_sided / 2`) from the previous `delta > 0.0` check.
- `benchmarks/appraisal_confound/runner.py`: `n_bootstrap` defaulted to 2000;
  now 10 000 per pre-reg Addendum A. Threaded through results dict and
  `_build_protocol` (eliminating drift from hard-coded constant).
- `benchmarks/appraisal_confound/runner.py`: `seed` hard-coded 42 in
  `_build_protocol`; now read from actual run value.
- `benchmarks/appraisal_confound/runner.py`: `_seed_everything` added to
  `run_study` (mirrors `benchmarks/realistic/runner.py`) for deterministic
  global RNG state.
- `benchmarks/appraisal_confound/runner.py`: `n_bootstrap` passed to
  `run_system_on_scenario` was dead compute (per-scenario CIs discarded);
  replaced with `n_bootstrap=1` per scenario call; single bootstrap pass
  runs on full flag lists.

### Changed (docs — G3/G9)

- `docs/research/audit_2026-04.md` G3 — replaced "unresolved" with actual
  results and honest interpretation: Ha2/Hb2 FAIL; architecture attribution
  holds descriptively via aft\_noAppraisal comparison and S2. After Hd1
  confirmatory, Gate 3 status updated to CLOSED.
- `docs/research/audit_2026-04.md` Q1 — updated reviewer-anticipation answer
  with closed status and Hd1 result.
- `docs/research/10_scientific_quality_bar.md` Gate 3 — status updated to
  "Partially closed": architecture-only advantage confirmed descriptively;
  pre-registered Ha2 failed; next step is a re-pre-registered confirmatory
  hypothesis.
- `docs/research/claim_validation_matrix.json` — added appraisal confound
  evidence note to `retrieval_affect_aware`; updated `not_yet_shown` and
  `next_study` for `replayable_multi_session_help`.

### Changed

- `benchmarks/ablation/runner.py`: added `no_reconsolidation` and `dual_path`
  variants; ablation runner now supports per-variant adapter overrides via
  `_ADAPTER_OVERRIDES`; CLI gains `--out-json`, `--out-md`, `--out-protocol`
  flags for explicit output paths (consistent with realistic runner).
- `benchmarks/realistic/runner.py`: `run_benchmark` and `_make_adapter` now
  accept `aft_adapter_cls: type[AFTReplayAdapter] | None = None` for per-call
  adapter overrides without modifying the realistic runner's default behavior.
- `docs/research/audit_2026-04.md`: G9 gap closed 2026-04-26 with results table
  and honest interpretation of He1 FAIL caveat and He2 null result.
- `docs/research/claim_validation_matrix.json`: `theory_faithful_operationalization`
  evidence note extended to cover dual-path and reconsolidation ablations.
- `tests/test_ablation_runner.py`: updated assertions for 7 variants / 6 pairwise rows.
- `benchmarks/datasets/realistic_recall_v2.json` — pre-registered S2 dataset
  (v2.0): 50 scenarios, 200 queries, 5 challenge types × 40
  (semantic_confound, affective_arc, recency_confound, same_topic_distractor,
  momentum_alignment). Committed before benchmark execution per construction rules.
- `benchmarks/realistic/results.v2.sbert.{json,md}` — SBERT bge-small-en v2
  benchmark results: AFT top1=0.53 vs naive_cosine=0.33, Δ=+0.205 [0.150,0.265],
  p_bootstrap<0.001, d=0.49. **G4 closed.**
- `benchmarks/realistic/results.v2.e5.{json,md}` — e5-small-v2 cross-embedder
  benchmark results: AFT top1=0.50 vs naive_cosine=0.34, Δ=+0.155 [0.090,0.225],
  p_bootstrap<0.001, d=0.31. **G5 closed** (advantage holds on both embedder classes).
- `docs/research/audit_2026-04.md` — critical self-review of the AFT research
  corpus: snapshot, corpus-at-a-glance, strengths, nine ranked gaps (G1–G9),
  theory–evidence coherence check, gate priority order, reviewer Q&A.
- `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT` env var for reasoning-budget control on
  o-series / gpt-5 models (empty ⇒ param omitted; consumed by
  `benchmarks/locomo/adapters/base.py::call_llm`).
- `benchmarks/locomo/README.md` — execution contract, env vars, operational notes
  for gpt-5-mini quirks, pre-reg cross-reference.
- `.claude/skills/` — 6 project-scoped Claude Code skills encoding the scientific
  workflow: `bench-locomo`, `bench-study`, `prereg-guard`, `evidence-update`,
  `paper-bundle`, `release-gate`.
- `docs/research/claim_validation_matrix.json` — canonical machine-readable
  matrix for public scientific claims, evidence levels, and allowed wording.
- `make bench-comparative-sbert` — paper-canonical SBERT comparative benchmark
  target; outputs to `benchmarks/comparative/results.sbert.{csv,md,protocol.json}`.
- `benchmarks/comparative/results.sbert.{csv,md,protocol.json}` — committed SBERT
  run: AFT = 0.80 = naive_cosine (ceiling effect, N = 20 items; recency = 0.25).
- `benchmarks/appraisal_confound/` — pre-registered appraisal confound study
  runner (Ha2: `aft_keyword > naive_cosine`; Hb2: equivalence test). No LLM key required.
- `benchmarks/preregistration_addendum_v2.md` — pre-registers appraisal confound,
  realistic_recall_v2 cross-embedder/multilingual, and human-eval publishability criteria.
- `docs/research/10_scientific_quality_bar.md` — formalises 3 mandatory claim gates
  and claim upgrade path.

- `benchmarks/locomo/results.{json,md}` — committed G2 LoCoMo benchmark results
  (pre-registered S1, 10 conversations, 1986 QA pairs). Gate 1: **FAIL**.
  AFT F1=0.168 vs naive_rag F1=0.271; judge_acc 0.279 vs 0.441. Both H1 and H2
  one-tailed p=1.0 after Holm correction. Affective weighting does not help on
  open-domain factual QA; claim ceiling unchanged.
- `benchmarks/locomo/runner.py` — `_compute_hypothesis_tests()` added: paired
  bootstrap H1 (token-F1), McNemar+bootstrap H2 (judge_accuracy), Holm–Bonferroni
  correction, Cohen's d, Gate 1 verdict in JSON and Markdown output.
- `docs/research/audit_2026-04.md` — G2 section updated with negative result;
  Q2 answer updated to reflect Gate 1 FAIL.
- `docs/research/10_scientific_quality_bar.md` — Gate 1 status updated:
  completed with negative result (2026-04-27).
- `docs/research/claim_validation_matrix.json` — new entry
  `locomo_external_qa_negative` documents the negative result and restricts
  allowed wording.

### Changed

- `docs/research/10_scientific_quality_bar.md` — Gate 3 status refreshed:
  appraisal-confound runner is implemented and awaiting execution (was
  "to be implemented").
- `docs/research/index.md`, `mkdocs.yml` — research nav now exposes
  `07_related_work.md`, `10_scientific_quality_bar.md`, and `audit_2026-04.md`.
- `benchmarks/locomo/adapters/base.py::call_llm` now retries generically on HTTP 400
  by stripping the `bad_param` reported by the API, instead of hardcoded model-name
  sniffing.
- `benchmarks/locomo/adapters/aft.py`: `SentenceTransformerEmbedder` instantiated
  once in `__init__` and reused across `reset()` calls, eliminating redundant model
  reloads between conversations.
- `benchmarks/locomo/{runner,scoring}.py` coerce gold/prediction to `str` before
  scoring (some LoCoMo gold answers are integers).
- `benchmarks/locomo/dataset.py`: corrected QA pair count in module docstring
  (~1986 total including cat-5 adversarial, not ~1540).
- `benchmarks/locomo/scoring.py`: module docstring now model-agnostic (judge model
  is resolved from `EMOTIONAL_MEMORY_LLM_MODEL` at runtime).
- `.gitignore`: refined `.claude/` exclusion to `!.claude/skills/` so project-scoped
  skills are version-controlled while local settings remain ignored.
- `Makefile`: `bench-locomo` and `bench-locomo-dry` prepend `PYTHONUNBUFFERED=1`
  so subprocess progress streams in real time.
- `docs/research/09_current_evidence.md` is now backed by a canonical claim
  validation matrix and documents allowed public wording for each major claim.
- `README.md` now points to the canonical claim-validation matrix so public
  validation wording is anchored to a versioned source of truth.
- `paper/main.tex` abstract, results section, and conclusion updated to v2 numbers:
  realistic benchmark now cited as v2 (N=200, SBERT Δ=+0.21 p<0.001 d=0.49;
  e5-small-v2 Δ=+0.16 p<0.001); LoCoMo Gate 1 FAIL documented in conclusion;
  G6 Italian multilingual caveat added to limitations.
- `scripts/reproduce_paper.py`: `_resolve_comparative_csv` prefers
  `results.sbert.csv` over `results.csv` for paper-canonical Table 3 generation.

### Added (from 0.6.3 prep, never tagged)

- `RedisAffectiveStateStore`, extending the new `AffectiveStateStore` boundary
  to a shared-state backend without changing the engine API.
- Comparative realistic replay benchmark infrastructure under
  `benchmarks/realistic/`, with AFT vs semantic-only and recency-only controls.
- Human-eval pilot pipeline under `benchmarks/human_eval/` to generate packet
  files, rating templates, and aggregated summaries.

### Changed

- Public docs now distinguish more clearly between theory-fidelity validation,
  early controlled comparative evidence, and still-open ecological / human
  validation gaps.
- Comparative benchmark docs and generated Markdown outputs now describe the
  current protocol as a controlled synthetic affect-aware retrieval probe,
  rather than implying general cross-system superiority.
- Persistence docs and limitations now distinguish local persisted state,
  optional shared state, and the still-missing distributed memory-store layer.
- The realistic replay benchmark now validates non-trivial candidate pools,
  promotes `top1_accuracy` to the headline metric, reports challenge-type
  aggregates, and exposes query-level recency triviality instead of relying on
  easy `hit@k` settings.
- The realistic replay dataset now spans 10 scenarios / 20 queries, with a
  larger `semantic_confound` subset and challenge-typed reporting that makes
  localized AFT gains visible instead of hiding weak subsets behind a single
  aggregate.
- The human-eval pipeline no longer treats blank rating templates as analyzable
  data: packet generation writes only the template, summary now fails fast when
  no completed ratings are present, and placeholder summary artifacts are no
  longer kept in the checked-in evidence surface.
- Human-eval v1 is now locked to a 10-scenario `aft` vs `naive_cosine` pilot
  with explicit rater instructions and an operational maintainer runbook.

## [0.6.2] - 2026-04-22

### Added

- `retrieve_with_explanations()` on sync and async engines, exposing the
  ranking-time score decomposition through `RetrievalExplanation`,
  `RetrievalBreakdown`, and `RetrievalSignals`.

### Changed

- Retrieval ranking is now built through a pure `build_retrieval_plan()`
  pipeline in `retrieval.py`; sync and async engines apply persistence-side
  effects afterward instead of duplicating ranking logic.
- Repository configuration is now centered on `pyproject.toml` + `Makefile`:
  Ruff moved into `pyproject.toml`, `pre-commit` now shells out through
  `uv run`, and local demo setup has a canonical `make install-demo` path.
- Demo and docs setup now distinguish between canonical local commands and
  deployment overlays: `demo/requirements.txt` is Space-only, while repo docs
  consistently point local contributors to `make install*` / `uv run`.

### Fixed

- `visualization.py` no longer breaks package-wide `mypy` due to matplotlib
  figure/kwargs typing mismatches in the standard release gate.

## [0.6.1] - 2026-04-21

### Added

- `observe()` / `reset_state()` on sync and async engines so integrations can update affective state
  without storing retrievable memories and can fully reset runtime state.
- Shared OpenAI-compatible HTTP LLM helper (`src/emotional_memory/llm_http.py`) plus
  `make llm-config` / `make llm-config-strict` preflight targets.
- Regression coverage for demo recall behavior, LangChain message policies, and shared LLM HTTP
  config handling.

### Changed

- Real-LLM validation now uses the shared config path everywhere and standardizes the default model
  on `gpt-5-mini`.
- Project quality gates now run consistently through `uv run`, matching the managed local env used
  for optional extras and release checks.

### Fixed

- Hugging Face / Gradio demo no longer stores recall commands or assistant replies as retrievable
  memories, preventing self-retrieval artifacts and affect drift.
- `EmotionalMemoryChatHistory` now keeps transcript order separate from episodic memory storage and
  exposes typed `add_user_message()` / `add_ai_message()` helpers.
- Real-LLM tests and benchmarks now fail fast on missing or incompatible provider config instead of
  silently degrading to fallback behavior.
- `visualization.py`, `scripts/reproduce_paper.py`, and related release paths no longer break
  `ruff` / `mypy` during the standard release gate.

## [0.6.0] - 2026-04-18

### Added

- `docs/tutorials/async.md` — async usage guide (`AsyncEmotionalMemory`, `as_async()`, `encode_batch()`)
- `docs/tutorials/persistence.md` — persistence guide (`SQLiteStore`, `save_state`, `export_memories`, `prune()`)
- `docs/tutorials/langchain.md` — LangChain integration guide (`EmotionalMemoryChatHistory`, `RunnableWithMessageHistory`)
- `mkdocs.yml` nav: new **Tutorials** section linking all three guides
- `Makefile` target `paper-arxiv` — builds `paper/arxiv-submission.tar.gz` (`.tex` + `.bbl` + figures + tables, no build artifacts)
- `paper/SUBMISSION.md` — arXiv submission checklist (category options, metadata template, pre-submission checks, post-acceptance steps)
- `demo/README.md`: `python_version: "3.11"` pinned in HF Space front-matter
- HuggingFace Space deployed to https://huggingface.co/spaces/homen3/emotional-memory-demo
- **Comparative baselines — Mem0 and LangMem adapters** (`benchmarks/comparative/adapters/`):
  - `mem0_adapter.py` — wraps `mem0ai>=2.0` with local qdrant backend; recall@5 = **0.95**, encode 1364 ms/item, p50 161 ms
  - `langmem_adapter.py` — wraps `langmem>=0.0.30` + `langgraph InMemoryStore`; recall@5 = **0.90**, encode 143 ms/item, p50 170 ms
  - `letta_adapter.py` — availability-guarded stub (cloud-only, requires `LETTA_API_KEY`); reports `not_evaluated` without key
  - `[mem0]` and `[langmem]` optional extras in `pyproject.toml`; `install-mem0` / `install-langmem` Makefile targets
- `benchmarks/comparative/runner.py`: `python-dotenv` integration + `EMOTIONAL_MEMORY_LLM_API_KEY → OPENAI_API_KEY` bridge for adapter compatibility

### Fixed

- `docs/mental_model.md`: broken relative link to `retrieval.py` replaced with absolute GitHub URL
- `Mem0Adapter.reset()`: removed `shutil.rmtree()` call on live qdrant dir (caused `SQLITE_READONLY` errors by orphaning SQLite/portalocker handles); reset now calls only `delete_all()`; temp dir lifecycle managed via `tempfile.TemporaryDirectory` + `close()`/`__del__`
- `LangMemAdapter.encode()`: now parses the stable langmem UUID from the `"created memory <UUID>"` return string instead of generating a random UUID (which broke recall mapping)
- `LangMemAdapter.retrieve()`: now `json.loads()` the JSON string returned by `search_memory_tool` instead of iterating over characters

## [0.5.2] - 2026-04-17

### Fixed

- **Paper (`paper/main.tex`) — figure 4 rendered empty**: generator used a duck-typed
  `_Link` class that was silently discarded by `isinstance(lnk, ResonanceLink)` in
  `visualization.py`. Generator now uses the real `ResonanceLink` Pydantic model.
- **Paper — figure 3 x-axis mislabeled**: timestamps were passed as step integers
  (0..19); x-axis showed 0..0.32 minutes. Generator now passes seconds (180 s/turn)
  so the axis correctly shows 0..57 minutes for a 20-turn conversation.
- **Paper — Table 2 (performance) missing**: `benchmark-results.json` was not
  generated with `--benchmark-json`. Added to bench-perf pipeline; table is now
  auto-included via `\input{tables/table2_perf.tex}`.
- **Paper — 10 dead bib entries** pruned from `refs.bib`; 2 missing references
  added (`ebbinghaus1885memory`, `kensinger2004emotional`).
- **Paper — symbol collision** `α`: appraisal vector in Layer 4 renamed to `\mathbf{a}`
  to avoid collision with arousal `α` in Layer 1.
- **Paper — PDF metadata empty**: `pdftitle`, `pdfauthor`, `pdfkeywords`, `pdfsubject`
  now populated via `\hypersetup`.
- **Paper — §Related Work**: 8 recent LLM-emotion papers now cited; MemEmo claim
  softened from "the first benchmark" to "a recent holistic benchmark".
- **Paper — §Conclusion**: future-work wording updated to reflect actual sbert baseline.
- **Paper — §Reproducibility**: DOI link, Python ≥3.11 requirement, and expected
  runtimes added.
- **`scripts/generate_paper_figures.py`**: wrong link type names (`"contrast"`,
  `"amplify"`) replaced with canonical Literal values (`"contrastive"`, `"emotional"`).
- **`benchmarks/conftest.py`**: `populate_store` now prints progress to stderr at
  100/500/1k/5k/10k milestones for long-running perf setups.

## [0.5.1] - 2026-04-17

### Fixed

- **SQLiteStore thread-safety** (`stores/sqlite.py`) — concurrent writes from multiple threads
  raised sqlite3 errors when a single `Connection` was shared without serialisation.
  Added a `threading.RLock` that serialises all connection access; `check_same_thread=False`
  was already set, but Python's sqlite3 leaves locking to the caller.
  `test_concurrent_write_from_other_thread` now passes reliably.

### Changed

- `CITATION.cff` added — enables the GitHub "Cite this repository" button and integrates
  with Zenodo for a citable DOI.
- README: fidelity benchmark heading clarified to "126 parametrized test cases, 20 phenomena"
  to accurately reflect pytest's counting of `@pytest.mark.parametrize` expansions.

## [0.5.0] - 2026-04-12

### Fixed

- **Plutchik categorization — sector 6 bug** (`categorize.py`) — sector 6 (270°, low-arousal
  neutral) incorrectly mapped to `"sadness"` (duplicate of sector 5); corrected to `"disgust"`,
  restoring all 8 Plutchik primary emotions to the circumplex.
- **Isotropic circumplex mapping** (`categorize.py`) — arousal coordinates were asymmetric
  (`[-0.5, 0.5]` span 1 vs valence `[-1, 1]` span 2); `atan2` on raw coordinates compressed
  high-arousal sectors and expanded high-valence ones. Fixed by scaling `a_centered × 2` before
  `atan2`, producing geometrically correct equal-angle sectors.
- **Neutral origin classified as "joy"** (`categorize.py`) — `atan2(0, 0) = 0°` → sector 0 →
  `"joy"` with `confidence=1.0`. Neutral points (`r < 0.05`) now return `confidence=0.0`.
- **`prune()` mutation during iteration** (`engine.py`, `async_engine.py`) — `delete()` during
  iteration over `list_all()` could skip entries or raise `RuntimeError` with lazy-iterator stores.
  IDs are now collected first, then deleted in a second pass.
- **`import_memories(overwrite=True)` used `save()` instead of `update()`** (`engine.py`,
  `async_engine.py`) — `save()` is `INSERT OR REPLACE` on `SQLiteStore` (worked by accident)
  but custom stores with pure-INSERT `save()` would silently create duplicates. Fixed to call
  `update()` for existing records.
- **LLM response regex greedy** (`appraisal_llm.py`) — `\{.*\}` captured from the first `{` to
  the last `}`, breaking multi-object LLM responses. Fixed to `\{[^{}]*\}` (non-greedy, flat
  schema only).
- **`SQLiteStore` thread safety** (`stores/sqlite.py`) — `sqlite3.connect()` defaulted to
  `check_same_thread=True`; `SyncToAsyncStore` dispatches via `asyncio.to_thread()` on arbitrary
  threads, raising `ProgrammingError`. Fixed to `check_same_thread=False` + WAL journal mode for
  concurrent reader/writer access.
- **`SQLiteStore._init_vec_from_db` empty-table bug** (`stores/sqlite.py`) — reopening a DB
  where `memory_vec` exists but is empty left `_dim=0` while `_vec_ready=True`. Fixed by parsing
  the embedding dimension from `sqlite_master` schema SQL when no rows are present.

### Changed

- **`MoodField` dominance signal range extended** (`mood.py`) — coefficient `0.25 → 0.5`,
  giving `dominance_signal = 0.5 + 0.5 × valence × arousal ∈ [0, 1]` (previously capped at
  `[0.25, 0.75]`; PAD model requires the full unit range).
- **`MoodDecayConfig` validates `base_half_life_seconds > 0`** (`mood.py`) — zero or negative
  values previously silenced regression silently; now raises `ValidationError`.
- **`AsyncEmotionalMemory._state` protected by `asyncio.Lock`** (`async_engine.py`) —
  concurrent `encode()` coroutines no longer race on affective state: the lock is held only
  during the synchronous state mutation, not during `await embed()`.
- **`async_engine.py` fully mirrors `engine.py`** — extracted `_add_bidirectional_links()` and
  `_elaborate_with_memory()` private helpers (deduplication + no double-fetch on
  `elaborate_pending()`); `close()` no longer performs a redundant inline `import asyncio`.
- **`AsyncEmotionalMemory.retrieve()` single `store.count()` call** — previously made two
  round-trips (one for logging, one for candidate limit); now reuses the first value.
- **`KeywordAppraisalEngine` per-dimension averaging** (`appraisal_llm.py`) — dimensions
  untouched by a rule were previously diluted when averaging over all firing rules. Each
  dimension is now averaged only over rules that contributed to it.
- **`as_async()` docstring clarified** (`async_adapters.py`) — `AffectiveState` reference
  sharing is safe because the object is always *replaced* (never mutated) on update.
- **`SQLiteStore` excluded from `__all__` when unavailable** (`__init__.py`) — previously
  declared in `__all__` even when `sqlite-vec` was absent, causing `AttributeError` on
  wildcard imports.

### Performance

- **Batch numpy cosine in `build_resonance_links()`** (`resonance.py`) — replaced Python
  per-item loop with `matrix @ q / (norms × q_norm)`; significant speedup for stores > 500.
- **`heapq.nlargest()` for top-k resonance links** (`resonance.py`) — O(n log k) vs O(n log n)
  full sort.
- **`export_memories()` single serialization** (`engine.py`, `async_engine.py`) — replaced
  `json.loads(m.model_dump_json())` double round-trip with `m.model_dump(mode="json")`.
- **`cosine_similarity` module-level import** (`retrieval.py`) — removed per-call import from
  the hot retrieval scoring path.
- **LLM fallback result cached** (`appraisal_llm.py`) — when `fallback_on_error=True` and the
  LLM call fails, the fallback `AppraisalVector` is now cached so repeated identical inputs
  don't re-invoke the LLM.
- **Retrieval weight constants** (`retrieval.py`) — `_MAX_MOOD_DIST = sqrt(6)`,
  `_MAX_AFFECT_DIST = sqrt(5)` replace the previous hardcoded approximations.
- **Zero-weight adaptive fallback** (`retrieval.py`) — when all weights clip to 0.0 under
  extreme mood states, retrieval now falls back to uniform `[1/6] × 6` instead of returning
  arbitrary zero-scored results.
- **Float threshold for momentum zero-check** (`retrieval.py`) — `mag_c == 0.0` exact
  comparison replaced with `mag_c < 1e-12` to avoid overflow on subnormal floats.
- **`MoodField.update()` uses `base.inertia`** (`mood.py`) — after `regress()`, the new field
  correctly inherits `base.inertia` rather than `self.inertia` (latent inconsistency with no
  current runtime effect, corrected for future `regress()` extensions).

### Added

- **Fidelity benchmark: emotional retrieval vs. cosine baseline**
  (`benchmarks/fidelity/test_emotional_vs_cosine.py`) — 3 tests demonstrating that the 6-signal
  retrieval outperforms pure cosine when embeddings are identical: mood-congruent recall (Bower
  1981), core-affect proximity (Russell 1980), and reconsolidation strengthening (Nader 2000).
- **`SQLiteStore` test coverage** (`tests/test_sqlite_store.py`) — 8 new tests: `__repr__`,
  brute-force cosine ranking path, `_ensure_vec()` edge cases (no-embedding save when vec ready,
  `update()` triggers vec creation, delete when vec absent), `_init_vec_from_db` empty-table
  regression, WAL mode verification, cross-thread write safety.
- **Concurrent encode test** (`tests/test_async_engine.py`) — 12 concurrent `encode()` calls
  on a shared `AsyncEmotionalMemory` verify that the `asyncio.Lock` prevents lost state updates.
- **Flaky test fix** (`tests/test_engine.py`) — `test_load_state_preserves_momentum_history`
  now passes explicit `now=fixed_now` to both `update()` calls, eliminating a timing race where
  sub-millisecond deltas produced inconsistent velocity values.

### Documentation

- **README** — updated fidelity benchmark count (106 → 126), added PAD dominance, Hebbian
  co-retrieval, ACT-R power-law decay, and emotional-vs-cosine to the phenomena table; updated
  phenomenon test counts to reflect v0.4.1 and v0.5.0 additions.

## [0.4.1] - 2026-04-12

### Fixed

- **CHANGELOG accuracy** — v0.4.0 incorrectly described `reconsolidate()` as using a
  "sigmoid-scaled adaptive learning rate"; the actual formula is linear: `alpha = min(ape * lr, 0.5)`.
  Pearce-Hall associability is handled exclusively by `update_prediction()`, not `reconsolidate()`.
- **Dead `adapt_rate` parameter removed** from `reconsolidate()` (`retrieval.py`) — the
  `adapt_rate=True` Pearce-Hall branch was unreachable dead code; no engine ever called it with
  `True`. Removing it eliminates a misleading public signature.
- **`stores/__init__.py` `__all__`** — added `SQLiteStore` so `from emotional_memory.stores import *`
  exports it correctly when `sqlite-vec` is installed.
- **Duplicate `AffectiveState` in docs** — removed the redundant `:::` directive from
  `docs/api/affect.md`; the class is documented exclusively in `docs/api/state.md`.
- **CHANGELOG v0.1.0/v0.2.0 terminology** — replaced stale "Stimmung" references with
  "MoodField"/"mood" throughout the historical changelog entries for consistency.
- **Module docstring artifacts** — removed internal "Step N:" prefixes from module-level
  docstrings in `engine.py`, `decay.py`, `retrieval.py`, `resonance.py`, `state.py`.

### Added

- **`elaborate()` / `elaborate_pending()` async tests** — 11 new tests in `test_async_engine.py`
  covering both methods (clear pending flag, blend affect, persist to store, window, edge cases).
- **`SyncToAsyncStore` direct tests** — `update()` and `search_by_embedding()` adapter methods
  now have dedicated unit tests.
- **NaN embedding warning tests** — sync and async engines both verified to emit `warnings.warn`
  when the embedder returns NaN values.
- **Reconsolidation window expiry test** — explicit test for the branch that clears
  `window_opened_at` when the lability window has elapsed.
- **Async `import_memories(overwrite=True)` test** — overwrite path previously untested.
- **Async `auto_categorize` during encode** — verified `emotion_label` is attached when the flag
  is set, and absent when it is not.
- **Concurrency tests** — threading test for independent sync engines; `asyncio.gather` test
  for independent async engines; concurrent read test for `SQLiteStore`.
- **`SQLiteStore` edge cases** — `update()` with a changed embedding vector replaces the vec
  table row correctly; dimension mismatch raises `sqlite3.OperationalError`.
- **Fidelity benchmark: Hebbian co-retrieval strengthening** (`test_hebbian_strengthening.py`) —
  4 tests validating Hebb (1949): co-retrieval increases link strength, monotonic growth over
  rounds, zero-increment leaves strength unchanged, strength capped at 1.0.
- **Fidelity benchmark: ACT-R power-law decay** (`test_decay_power_law.py`) — 5 tests
  verifying Anderson (1983) + McGaugh (2004): strictly decreasing strength, log-log linearity
  R² > 0.99, arousal slows decay, high-arousal floor respected, low-arousal can fall below floor.
- **Fidelity benchmark: PAD dominance** (`test_pad_dominance.py`) — 8 tests (+ parametrised)
  validating Mehrabian & Russell (1974): positive×high-arousal raises dominance, negative×high
  lowers it, low arousal stays near neutral, dominance clamped to [0, 1], formula verified
  numerically.

### Documentation

- **README** — `EmotionalMemory` API table now includes `elaborate()` and `elaborate_pending()`;
  `AsyncEmotionalMemory` coroutine list now includes `elaborate`, `elaborate_pending`, `count`.

## [0.4.0] - 2026-04-12

### Added

- **Discrete emotion categorization** (`categorize.py`) — `EmotionLabel`, `categorize_affect()`,
  `label_tag()`: maps continuous (valence, arousal) coordinates to Plutchik's 8 primary emotions
  with intensity tiers (low/moderate/high) via angular sector lookup in the Russell circumplex;
  optional dominance parameter disambiguates fear vs anger (Mehrabian & Russell 1974)
- **`auto_categorize` config flag** — when `True`, every `encode()` / `encode_batch()` call
  automatically attaches an `EmotionLabel` to the stored `EmotionalTag`
- **Dual-speed encoding** (LeDoux, 1996) — `dual_path_encoding` config flag enables fast
  thalamo-amygdala path (`pending_appraisal=True`, no appraisal call); `elaborate(memory_id)` runs
  the slow thalamo-cortical appraisal later and blends affect (70% appraised / 30% raw);
  `elaborate_pending()` processes all outstanding fast-path memories in one call
- **Adaptive prediction error** (Schultz 1997, Pearce-Hall 1980) — `compute_ape()` computes
  affective prediction error against `expected_affect` (EMA prediction) when available; called on
  every retrieval so the prediction model learns continuously; `update_prediction()` applies
  Pearce-Hall associability: large errors increase the learning rate, small errors decrease it
- **APE-gated reconsolidation window** — `window_opened_at` field on `EmotionalTag` separates
  window-opening (requires APE above threshold) from `last_retrieved` (any retrieval); fixes the
  prior behaviour where any retrieval could open the lability window
- 76 new tests across `tests/test_categorize.py`, `tests/test_prediction.py`,
  `tests/test_dual_path.py` and 4 new fidelity benchmarks in `benchmarks/fidelity/`

### Changed

- `reconsolidate()` now applies a linearly-scaled alpha (`min(ape * learning_rate, 0.5)`) so
  larger prediction errors produce proportionally larger core affect updates, capped at 50% per
  retrieval (Schultz 1997); Pearce-Hall associability is handled separately by `update_prediction()`
- `encode_batch()` now honours `dual_path_encoding` and `auto_categorize` flags, consistent with
  the single-item `encode()` path

## [0.3.0] - 2026-04-12

### Added

- **Spreading activation** (Collins & Loftus, 1975) — `spreading_activation()` in `resonance.py`
  performs BFS-based multi-hop propagation through the associative link graph; activation decays
  multiplicatively per hop and uses max-aggregation to prevent path-count inflation; configurable
  via `ResonanceConfig.propagation_hops` (1–5, default 2)
- **Bidirectional resonance links** — encoding a memory now creates backward links on all target
  memories so activation flows in both directions through the network; the weakest existing link is
  evicted if the target is already at `max_links`
- **Hebbian co-retrieval strengthening** (Hebb, 1949) — `hebbian_strengthen()` in `resonance.py`
  increments the strength of every link shared between memories returned in the same retrieval call
  ("neurons that fire together wire together"); increment configurable via
  `ResonanceConfig.hebbian_increment` (default 0.05, capped at 1.0)
- **Configurable link-classification thresholds** — causal, contrastive, and temporal thresholds
  that were previously hardcoded magic numbers in `_classify_link_type()` are now named fields on
  `ResonanceConfig`: `contrastive_temporal_threshold`, `contrastive_valence_threshold`,
  `causal_temporal_threshold`, `causal_semantic_threshold`
- **Vectorized `InMemoryStore.search_by_embedding`** — rewrites the per-memory Python loop with a
  NumPy batch matrix multiply + `np.argpartition` (O(n)) for top-k selection; significant speedup
  for stores > 500 memories
- `spreading_activation` and `hebbian_strengthen` exported from the top-level package (`__all__`)

### Breaking Changes

- **`StimmungField` → `MoodField`** — import from `emotional_memory.mood`; the old
  `emotional_memory.stimmung` module is removed entirely
- **`StimmungDecayConfig` → `MoodDecayConfig`** — same module move
- **`EmotionalMemoryConfig.stimmung_alpha` → `mood_alpha`**
- **`EmotionalMemoryConfig.stimmung_decay` → `mood_decay`**
- **`EmotionalTag.stimmung_snapshot` → `mood_snapshot`**
- **`AffectiveState.stimmung` → `mood`**
- **`get_current_stimmung()` → `get_current_mood()`** on both `EmotionalMemory` and
  `AsyncEmotionalMemory`
- **`make_emotional_tag()` parameter `stimmung` → `mood`**
- **`EmotionalTag` is now frozen** (`model_config = ConfigDict(frozen=True)`) — consistent
  with all other value objects; mutating tag fields now raises `ValidationError`

### Fixed

- **Decay formula boost** — `compute_effective_strength()` no longer returns a value above
  the initial `consolidation_strength` for very small elapsed times (power-law exponent can
  produce values > 1 when `elapsed < 1 s`)
- **Calm-event floor** — `consolidation_strength()` now has a minimum of `0.1`; memories
  encoded under low-arousal states are no longer immediately prunable
- **`RetrievalConfig.base_weights` length** — a Pydantic `field_validator` now raises
  `ValidationError` if the list does not contain exactly 6 elements
- **`ResonanceLink.strength` range** — field is now declared with `ge=0.0, le=1.0`
- **`as_async()` documentation** — docstring now correctly states that state is copied at
  wrap time; the two engines are independent afterwards

### Changed

- Docstrings reworked for theoretical honesty: "implements X" → "inspired by X" where the
  code is a simplification (Scherer CPM note added; Heidegger reference demoted to loose
  inspiration in `mood.py`)
- `appraisal.py` module docstring notes that the CPM evaluation is a simultaneous linear
  combination, not the original sequential model

## [0.2.0] - 2026-04-10

### Added

- **13 runnable examples** covering the full public API — `basic_usage`, `advanced_config`,
  `appraisal_engines`, `async_usage`, `emotional_journal`, `httpx_llm_integration`,
  `llm_appraisal`, `persistence`, `reconsolidation`, `resonance_network`, `retrieval_signals`,
  `sentence_transformers_embedder`, `visualization`; each is self-contained and always runnable
  without ML dependencies
- **Visualization module** (`visualization.py`) — 8 matplotlib plot functions: circumplex,
  decay curves, Yerkes-Dodson, retrieval radar, mood evolution, adaptive weights heatmap,
  resonance network, appraisal radar; install via `pip install emotional-memory[viz]`
- **`python-dotenv` optional extra** (`pip install emotional-memory[dotenv]`) and
  `make install-dotenv` Makefile target
- **`examples/httpx_llm_integration.py`** — SDK-agnostic LLM pipeline using raw httpx; covers
  `AffectiveMomentum`, `LLMCallable`, `ResonanceLink`, `SyncToAsyncAppraisalEngine`,
  `make_emotional_tag`, `consolidation_strength`, and `__version__` (previously uncovered)
- **`examples/emotional_journal.py`** — capstone multi-session journaling app combining
  `SQLiteStore`, `KeywordAppraisalEngine`, `MoodDecayConfig`, mood-congruent retrieval,
  reconsolidation, and `prune()`
- **MkDocs documentation site** with API reference (mkdocstrings) and research pages
- **`prune(threshold=0.05)`** on `EmotionalMemory` and `AsyncEmotionalMemory` — removes memories
  whose `compute_effective_strength()` has fallen below the given threshold; returns count removed
- **`export_memories()` / `import_memories(data, overwrite=False)`** on both engines — bulk
  serialise all memories to a list of JSON-safe dicts for backup or store migration;
  `import_memories` skips duplicate IDs by default, returns count written
- **`close()` and context-manager support** on both engines — `with EmotionalMemory(...) as em`
  and `async with AsyncEmotionalMemory(...) as em` propagate cleanup to the underlying store
  (calls `store.close()` when available, no-ops otherwise)
- **`SequentialEmbedder`** base class in `interfaces.py` — subclass and implement `embed()`;
  `embed_batch()` is provided automatically as a sequential fallback; exported from top-level `__init__`
- **`SQLiteStore` re-export** — now importable as `from emotional_memory import SQLiteStore`
  (when `sqlite-vec` is installed); also re-exported from `emotional_memory.stores`
- **Structured logging** — `engine.py`, `async_engine.py`, and `appraisal_llm.py` emit `DEBUG`
  log records at key pipeline points (encode start/stored/resonance, retrieve start/done,
  reconsolidate, cache hit/fallback) via `logging.getLogger(__name__)`
- **`__repr__`** on all non-Pydantic concrete classes — `EmotionalMemory`, `AsyncEmotionalMemory`,
  `InMemoryStore`, `SQLiteStore`, `LLMAppraisalEngine`, `KeywordAppraisalEngine`,
  `StaticAppraisalEngine`
- **`__slots__`** on all non-Pydantic classes — reduces per-instance memory footprint and
  prevents accidental attribute creation
- **Smoke test for `examples/basic_usage.py`** (`tests/test_examples.py`) — executed via
  `runpy.run_path` to catch silent breakage in the example script
- **LLM integration tests** (`tests/test_llm_integration.py`) — 5 end-to-end tests against a
  real OpenAI-compatible endpoint; gated behind `pytest.mark.llm` and API key env var
- **Appraisal quality benchmarks** (`benchmarks/appraisal_quality/`) — 15 natural-language
  phrases with directional assertions on Scherer's 5 dimensions; evaluates median over N repeats
- **numpy cosine similarity** — replaced pure-Python loop with `np.dot + np.linalg.norm`;
  added NaN guard returning 0.0 to prevent NaN propagation in scoring
- **Performance: hoisted `adaptive_weights()`** — computed once per `retrieve()` call instead
  of once per candidate per pass; `retrieval_score()` accepts `precomputed_weights` parameter
- **Performance: skip Pass 2** when no resonance links target the active memory set
- **Engine facade methods**: `get(memory_id)`, `list_all()`, `__len__()`/`count()` on both
  `EmotionalMemory` and `AsyncEmotionalMemory`
- **Input validation** on `encode_batch()` (metadata/contents length mismatch raises `ValueError`)
  and `retrieve()` (top_k < 1 raises `ValueError`)
- **CI jobs for optional extras** — dedicated sqlite-tests and viz-tests jobs install and
  exercise those extras explicitly so they are never silently skipped
- **`__init__.py` export smoke test** — verifies all `__all__` entries are importable
- **`AsyncEmotionalMemory`** — async-native facade mirroring `EmotionalMemory`; all I/O methods
  (`encode`, `retrieve`, `encode_batch`, `delete`) are coroutines; state accessors remain sync
- **Async protocols** in `interfaces_async.py`: `AsyncEmbedder`, `AsyncMemoryStore` (uses
  `count()` instead of `__len__`), `AsyncAppraisalEngine` — all `@runtime_checkable`
- **Sync-to-async bridge adapters** in `async_adapters.py`: `SyncToAsyncEmbedder`,
  `SyncToAsyncStore`, `SyncToAsyncAppraisalEngine`, and `as_async()` convenience wrapper
- **`SQLiteStore`** in `stores/sqlite.py` — persistent `MemoryStore` backed by SQLite +
  sqlite-vec for ANN vector search; install via `pip install emotional-memory[sqlite]`;
  context-manager support; lazy vector index creation
- **`LLMAppraisalEngine`** — provider-agnostic LLM-backed appraisal via user-supplied
  `LLMCallable` protocol; LRU cache (configurable size), fallback-on-error, markdown fence
  extraction, `LLMAppraisalConfig`
- **`KeywordAppraisalEngine`** — rule-based appraisal fallback using `KeywordRule` regex
  patterns with dimension score deltas; ships with defaults covering success, failure,
  novelty, danger, and social norms
- **`save_state()` / `load_state()`** on `EmotionalMemory` and `AsyncEmotionalMemory` —
  serialise and restore the full `AffectiveState` (core affect, momentum history, MoodField)
  as a JSON-safe dict, enabling session persistence
- **`get_current_mood(now)`** — read-only mood inspection with time-based regression
  applied on-the-fly without mutating engine state
- **`MoodDecayConfig`** — exponential mood regression toward PAD baselines, modulated
  by inertia; configurable half-life and inertia scale; applied via `MoodField.regress()`
- **`AdaptiveWeightsConfig`** — continuous sigmoid/Gaussian modulation of retrieval weights
  replacing hard thresholds; `_smooth_gate()` helper for tanh-based gate functions
- **`ResonanceConfig.candidate_multiplier`** — pre-filter resonance candidates in large stores
  to avoid loading all memories during encode
- **Context passthrough** — `appraise(content, context=metadata)` now forwarded in both
  `encode()` and `encode_batch()` paths, enabling LLM appraisal engines to use memory metadata

## [0.1.0] - 2026-04-09

### Added

- **Affective Field Theory (AFT)** — original 5-layer emotional model for LLM memory systems
- `CoreAffect` — continuous valence/arousal circumplex (Barrett/Russell 1980)
- `AffectiveMomentum` — time-normalised velocity and acceleration of affect transitions (Spinoza)
- `MoodField` — slow-moving global mood with inertia and PAD-based dominance update,
  evolved via EMA (Heidegger §29 / Mehrabian & Russell 1974)
- `AppraisalVector` — emotion derived from 5-dimension cognitive evaluation with `to_core_affect()`
  mapping (Scherer CPM 2009 / Lazarus / Stoics)
- `ResonanceLink` — associative memory graph with semantic, emotional, temporal, causal, and
  contrastive link types (Aristotle / Bower 1981 spreading activation)
- `EmotionalTag` — immutable snapshot of all 5 layers at encoding time + consolidation metadata
- `EmotionalMemory` — main facade:
  - `encode(content, appraisal, metadata)` — single-item encode with full AFT pipeline
  - `encode_batch(contents, metadata)` — batched encode via `embed_batch()`, per-item appraisal
  - `retrieve(query, top_k)` — two-pass spreading activation with mood-adaptive weights
  - `delete(memory_id)` — remove a memory from the store
  - `get_state()` / `set_affect()` — read and write the runtime affective state
- `InMemoryStore` — dict-backed `MemoryStore` with brute-force cosine search
- `Embedder` and `MemoryStore` — `typing.Protocol` interfaces for dependency injection (PEP 544)
- Power-law memory decay (ACT-R, Anderson 1983), arousal-modulated, with configurable `power`
  exponent and high-arousal floor (Merleau-Ponty body memory)
- Mood-congruent retrieval: 6-signal weighted scoring (semantic, mood-congruence,
  affect-proximity, momentum-alignment, recency, resonance-boost)
- Mood-adaptive retrieval weights (Heidegger: mood is the ground of disclosure)
- Two-pass spreading activation: first pass seeds active memory IDs for resonance boost
- Embedding pre-filter: `candidate_multiplier` limits scoring candidates in large stores
- Reconsolidation with lability window: tag updated on high APE only within
  `reconsolidation_window_seconds` of previous retrieval (Nader & Schiller 2000)
- `DecayConfig.power` — configurable power-law scaling exponent
- 296 tests: 219 unit/integration + 77 psychological fidelity benchmarks
- 14 performance benchmarks (encode throughput, retrieve latency, memory footprint, resonance build)
- PEP 561 typed (`py.typed` marker), mypy strict, 98% branch coverage
- CI: GitHub Actions matrix (Python 3.11-3.14), Codecov upload, benchmark regression tracking
- PyPI release workflow (OIDC trusted publishing)
- Pre-commit hooks: ruff check + format

[Unreleased]: https://github.com/gianlucamazza/emotional-memory/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/gianlucamazza/emotional-memory/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/gianlucamazza/emotional-memory/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/gianlucamazza/emotional-memory/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gianlucamazza/emotional-memory/releases/tag/v0.1.0
