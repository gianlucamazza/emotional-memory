# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `benchmarks/appraisal_confound/results.{json,md,protocol.json}` — G3
  evidence committed (2026-04-26, SBERT, N=100, n\_bootstrap=10 000, seed=42):
  aft\_noAppraisal = 0.78 vs naive\_cosine = 0.55 (Δ ≈ +0.23, architecture
  attribution descriptive); Ha2 (aft\_keyword vs naive\_cosine) FAIL Δ = −0.39
  (keyword appraisal destructively overrides preset affect); Hb2 FAIL Δ = −0.62.
- `--n-bootstrap` and `--seed` CLI flags on
  `benchmarks/appraisal_confound/runner.py` for reproducibility and sensitivity
  runs.

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

### Changed (docs — G3)

- `docs/research/audit_2026-04.md` G3 — replaced "unresolved" with actual
  results and honest interpretation: Ha2/Hb2 FAIL; architecture attribution
  holds descriptively via aft\_noAppraisal comparison and S2.
- `docs/research/audit_2026-04.md` Q1 — updated reviewer-anticipation answer
  to "partially resolved" with the descriptive architecture-attribution result.
- `docs/research/10_scientific_quality_bar.md` Gate 3 — status updated to
  "Partially closed": architecture-only advantage confirmed descriptively;
  pre-registered Ha2 failed; next step is a re-pre-registered confirmatory
  hypothesis.
- `docs/research/claim_validation_matrix.json` — added appraisal confound
  evidence note to `retrieval_affect_aware`; updated `not_yet_shown` and
  `next_study` for `replayable_multi_session_help`.

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
- `paper/main.tex` Table 3 and surrounding text realigned to current SBERT results:
  quadrant probe ties AFT and naive_cosine at 0.80 (ceiling); realistic benchmark
  (AFT 0.70 vs naive_cosine 0.50, SBERT, N = 100) is cited as ranking-shift evidence.
- `scripts/reproduce_paper.py`: `_resolve_comparative_csv` prefers
  `results.sbert.csv` over `results.csv` for paper-canonical Table 3 generation.

## [0.6.3] - 2026-04-22

### Added

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

[Unreleased]: https://github.com/gianlucamazza/emotional-memory/compare/v0.6.3...HEAD
[0.6.3]: https://github.com/gianlucamazza/emotional-memory/compare/v0.6.2...v0.6.3
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
