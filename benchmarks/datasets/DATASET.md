---
license: cc-by-4.0
language:
  - en
  - it
  - es
  - fr
pretty_name: Emotional Memory Benchmarks
task_categories:
  - text-retrieval
size_categories:
  - 1K<n<10K
---

# Emotional Memory Benchmarks

Draft Hugging Face dataset card for **`emotional-memory-benchmarks`**.
This file is the in-repo card. **Nothing in this PR is uploaded to Hugging Face.**
The planned Hub dataset will contain **only** the owner-authored sets listed in
[`PUBLISHABLE.txt`](PUBLISHABLE.txt). Third-party and third-party-derived
corpora stay in this git repository with their upstream licences and are
excluded by `tests/test_publishable_datasets.py`.

- **Hub copy:** [`gianlucamazza/emotional-memory-benchmarks`](https://huggingface.co/datasets/gianlucamazza/emotional-memory-benchmarks)
- **Owner:** Gianluca Mazza
- **Library:** [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory)
- **Target Hub publish date:** 26 October (this inventory is meant to merge first)
- **Concept DOI (software / project):** [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258)
  (`release.toml`, `codemeta.json`, `RELEASING.md`)
- **Attribution:** credit this dataset by using the [citation](#citation)
  below and the concept DOI
  [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258).

`license: cc-by-4.0` in the YAML is the **owner-decided** Hub licence for
`affect_reference_v1` ([`README.md`](README.md) §License) and every
owner-authored `realistic_recall_*` set listed in `PUBLISHABLE.txt`. It does
**not** apply to any third-party file in this directory. Project code remains
MIT.

## Intended use

These sets evaluate **affect-conditioned memory retrieval** for Affective Field
Theory (AFT) as implemented in `emotional-memory`: encode scripted or labeled
events, retrieve with a 6-signal scorer, and compare against a semantic-only
cosine baseline. They are **controlled research probes**, not production
leaderboards and not human-eval corpora.

Use them to:

- reproduce the in-repo realistic-replay, ablation, circularity, gating, and
  multilingual studies documented under `benchmarks/`
- inspect scenario / query construction (challenge types, oracle affect, gate
  labels)
- train or analyse retrieval methods **only** with the Addendum U caveat and
  the recorded negative results in mind

Do **not** treat a win on `realistic_recall_v2` as general superiority. The
project's own claim matrix scopes the advantage to an affect-discriminative,
mood-congruent regime on author-crafted data.

## Limitations

- **Circularity / construction (Addendum U).** Quoted from
  `benchmarks/preregistration_addendum_u_circularity_audit_closure.md`:
  **“AFT-favorable fraction: 62.5%”** (125/200 queries fall in the
  `affect_only_can_help` cell = not cosine-solvable AND affect-separating).
  The same closure states that **62.5% of queries are constructed so that
  affect can break a semantic tie**, and that the aggregate advantage is
  confined to that regime. `docs/research/claim_validation_matrix.json`
  (`replayable_multi_session_help` / `realistic_replay_vs_sota`,
  `not_yet_shown`) restates: “62.5% of v2 queries are AFT-favorable by
  construction”. Do not round this figure differently.
- **Oracle affect.** Headline `realistic_recall_v2` / Addendum R numbers use
  preset valence/arousal at encode and query-time state
  (`docs/research/08_limitations.md` §2.4). Automatic appraisal on
  affect-free variants is a **FAIL** (Addenda G, P).
- **Language / embedder.** Italian and Spanish **me5 FAIL at declared power**
  (N=120). French me5 is a PASS on a different 2-session runner. See
  [Negative results](#negative-results).
- **Not in this Hub dataset:** EmoBank, MADial-Bench, ES-MemEval/EvoEmo,
  DailyDialog-derived personas, and LoCoMo (LoCoMo is not vendored under
  `benchmarks/datasets/`). Those studies still bound how the owner sets
  should be read.

## Negative results

Wording below follows `README.md` and
`docs/research/claim_validation_matrix.json`. Closures are the scored
sources.

### LoCoMo — FAIL

From `docs/research/claim_validation_matrix.json`, claim
`locomo_external_qa_negative`, `allowed_public_wording`:

> On the LoCoMo conversational QA benchmark (1986 QA pairs, 10 conversations),
> AFT retrieval underperforms a naive RAG baseline (F1 0.168 vs 0.271;
> judge_acc 0.279 vs 0.441). Gate 1 was not met. AFT's affective weighting
> does not help on factual open-domain QA.

`README.md`: “on LoCoMo (1986 QA pairs) AFT underperforms a naive RAG
baseline (F1 0.168 vs 0.271; Gate 1 FAIL).”

- Closure: [`benchmarks/preregistration_s1_closure.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_s1_closure.md)
- Results: `benchmarks/locomo/results.md`
- Related FAILs: Addendum J (Hj1, all 10 weight configs), Addendum L (Hl1
  query-type routing) — `benchmarks/README.md` addenda index

### MADial-Bench — FAIL (inverted)

From `docs/research/claim_validation_matrix.json`, claim
`cross_domain_affect_replication`, `allowed_public_wording`:

> Addendum X FAIL (2026-07-02): first third-party retrieval-native corpus
> (MADial-Bench EN, NAACL 2025, N=160, oracle-free) — aft_query_appraised
> significantly below naive cosine (nDCG@5 0.221 vs 0.304, Δ=-0.083
> [-0.123, -0.043], p_one=0.9998) […] construct mismatch — the benchmark
> rewards counter-congruent supportive recall (emotion regulation), the
> opposite of AFT's mood-congruence prior.

`benchmarks/README.md` addenda index: **FAIL, inverted** (Δ=−0.083; construct
boundary: counter-congruent supportive recall).

- Closure: [`benchmarks/preregistration_addendum_x_madialbench_third_party_closure.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_x_madialbench_third_party_closure.md)
- Vendored data (not publishable): `madialbench/` — MIT; see
  [`madialbench/ATTRIBUTION.md`](madialbench/ATTRIBUTION.md)

### ES-MemEval / EvoEmo — FAIL (inverted)

From the same `allowed_public_wording`:

> Addendum X2 FAIL (2026-07-07): second third-party corpus
> (ES-MemEval/EvoEmo, WWW 2026, N=1,133 in-family queries, oracle-free) —
> aft_query_appraised significantly below naive cosine (upstream-verbatim
> nDCG@4 0.133 vs 0.284, Δ=-0.150 [-0.165, -0.136], p_one=1.0000, powered:
> MDE 0.018) […] failure mode distinct from X — the gold is
> affect-orthogonal (content-determined; 45.5% of queries affectively
> neutral […])

`benchmarks/README.md`: **FAIL, inverted** (Δ=−0.150 powered; distinct
failure mode: affect-orthogonal QA gold).

- Closure: [`benchmarks/preregistration_addendum_x2_esmemeval_third_party_closure.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_x2_esmemeval_third_party_closure.md)
- Vendored data (not publishable): `esmemeval/` — CC-BY-4.0; see
  [`esmemeval/ATTRIBUTION.md`](esmemeval/ATTRIBUTION.md)
- Artifact counts from `benchmarks/esmemeval/dataset.py`: 18 seekers, 401
  sessions, 1,427 QA in the released file, 1,133 in-family queries

### Italian / Spanish me5 — FAIL at declared power

`README.md`:

> Italian/Spanish me5 at declared power (N=120) FAIL; English-SBERT and
> SBERT-Spanish (N=80) hold.

`docs/research/09_current_evidence.md` (Hd2 table) and
`docs/research/claim_validation_matrix.json` (`retrieval_affect_aware`,
`not_yet_shown`: “Hd2 Branch C FAIL 2026-05-07”):

| Study | Dataset / embedder | N | Verdict | Δ (top1) | p |
| --- | --- | ---: | --- | --- | --- |
| Hd2_IT_v120 (power top-up) | `realistic_recall_v2_it` / me5 | 120 | **FAIL** | +0.058 | 0.276 |
| Hd2_ES_v120 (power top-up) | `realistic_recall_v2_es` / me5 | 120 | **FAIL** | 0.000 | 1.000 |

- Closure: [`benchmarks/preregistration_addendum_hd2_powertopup_closure.md`](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/preregistration_addendum_hd2_powertopup_closure.md)
- `docs/research/08_limitations.md` §1.2: “Italian and Spanish me5 FAIL at
  N=120 power.”

French (`realistic_recall_v2_fr`, Addendum M) is a **PASS** on the 2-session
realistic runner (Δ=+0.18, Hedges g=0.424) —
`benchmarks/preregistration_addendum_m_fr_closure.md`. That does not reverse
the IT/ES me5 power-top-up FAILs.

## Publishable sets (owner-authored)

Ids match [`PUBLISHABLE.txt`](PUBLISHABLE.txt). Counts below are from the
committed files (this inventory) unless a cited document gives the same
number. Origin is taken only from generator scripts, commit messages,
addenda, or README notes; **unknown** where the repo is silent.

### `affect_reference_v1`

| | |
| --- | --- |
| File | `affect_reference_v1.jsonl` |
| Rows | **258** examples (one JSON object per line) |
| Language | en |
| Licence | **CC-BY-4.0** (`README.md` §License) |
| Origin | **Script-generated** from hardcoded seed texts in `generate_dataset.py` (`source`: `synthetic-v1`). README: “original synthetic compositions.” Whether the seed strings were LLM-drafted is **unknown** (no generation log). |
| First commit | `69785c9` (2026-04-17) |

Quadrant probe for mood-congruent retrieval (`benchmarks/comparative/`).

### `realistic_recall_v1`

| | |
| --- | --- |
| File | `realistic_recall_v1.json` |
| Size | **50** scenarios, **100** queries, 100 sessions, 300 events (file + `README.md` v1.4) |
| Language | en |
| Licence | **CC-BY-4.0** (metadata field; owner decision) |
| Origin | **Hand-authored / scripted** multi-session scenarios. No generator script in-repo. `README.md`: “scripted multi-session replay scenarios.” Introduced `82dff45`; expanded 10→50 scenarios in `5345b2f`. LLM involvement in composing the texts is **unknown**. |
| Not third-party-derived | No repo evidence of DailyDialog/EmoBank seeding. |

### `realistic_recall_v2`

| | |
| --- | --- |
| File | `realistic_recall_v2.json` |
| Size | **50** scenarios, **200** queries (5 challenge types × 40); 100 sessions, 300 events |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **Hand-authored.** `docs/research/08_limitations.md` §2.4 / Addendum U: affect labels were “hand-crafted by the author with AFT theory in mind.” Added in `194d857`. No generator script. Construction rules: `benchmarks/preregistration_addendum_v2.md`. |
| Primary use | Headline realistic-replay benchmark (S2, Hd2, G4/G5, S3, Addenda R/T/U/Y/Z). |

### `realistic_recall_v2_it`

| | |
| --- | --- |
| File | `realistic_recall_v2_it.json` |
| Size | **30** scenarios, **120** queries, 60 sessions, 228 events (file). JSON `description` still says “20 scenarios” (stale header from the G6 slice). |
| Language | it |
| Licence | **CC-BY-4.0** |
| Origin | **Two layers.** Scenarios 1–20: **translated** to Italian (`README.md`; added `2d8b397`). Whether that translation was human or LLM is **unknown**. Scenarios 21–30: **LLM-assisted** (`gpt-4.1-mini`, seed=42) via `scripts/generate_realistic_v2_topup.py`; valence/arousal assigned programmatically; calls logged in `realistic_recall_v2_topup_provenance.jsonl` (`benchmarks/preregistration_addendum_hd2_powertopup.md`, commit `e308e94`). |
| Not third-party-derived | Translation / top-up of the owner v2 family, not DailyDialog/EmoBank. |

### `realistic_recall_v2_es`

| | |
| --- | --- |
| File | `realistic_recall_v2_es.json` |
| Size | **30** scenarios, **120** queries, 60 sessions, 228 events (file). JSON `description` still says “20 escenarios”. |
| Language | es |
| Licence | **CC-BY-4.0** |
| Origin | **Two layers.** Scenarios 1–20: **translated / localized** to Spanish — “only text and identifiers translated” (`898e132` commit message; `README.md`). Translation tool (human vs LLM) is **unknown**. Scenarios 21–30: **translated from the Italian top-up** by the same generator (`--lang es --translate-from` … `_it.json`; `e308e94`). |
| Not third-party-derived | Same as `_it`. |

### `realistic_recall_v2_fr`

| | |
| --- | --- |
| File | `realistic_recall_v2_fr.json` |
| Size | **30** scenarios, **120** queries, 60 sessions, **210** events (file + `README.md`; freeze commit `a864485` notes 210 vs a prereg ‘~228’ estimate) |
| Language | fr |
| Licence | **CC-BY-4.0** |
| Origin | **Hand-authored native French**, not translated from Italian (`benchmarks/preregistration_addendum_m_fr.md`, `…_m_fr_closure.md`). |

### `realistic_recall_v3`

| | |
| --- | --- |
| File | `realistic_recall_v3.json` |
| Size | **125** scenarios, **500** queries (v2’s 50/200 plus 75 new / 300 queries); 250 sessions, 750 events |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **Script-assembled** by `scripts/generate_realistic_v3.py` from scenario literals in that file (commit `320cd23`, author-blind). The first 50 scenarios are the v2 set. How the 75 new scenario texts were drafted (hand vs LLM) is **unknown** — no LLM call log. |

### `realistic_recall_v3_noAF`

| | |
| --- | --- |
| File | `realistic_recall_v3_noAF.json` |
| Size | **50** scenarios, **200** queries |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **Derived from owner `realistic_recall_v2`** by removing event valence/arousal and query `state` (`055a78a`; `benchmarks/appraisal_confound/runner_hg1.py` `dataset_note`; Addendum G). Not third-party text. |

### `realistic_recall_v4_noAF`

| | |
| --- | --- |
| File | `realistic_recall_v4_noAF.json` |
| Size | **40** scenarios, **160** queries (`pNN_` ids; 80 sessions, 240 events) |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **LLM-assisted**, author-blind: `generate_v4_noAF.py` drafts one structured-JSON scenario per call with **gpt-5-mini**; Python fixes ids / session split / challenge mapping (`benchmarks/preregistration_addendum_p_hg1_rerun.md`). Themes chosen disjoint from v3. Frozen `66dd3cc`. |

### `realistic_recall_v5_gate`

| | |
| --- | --- |
| File | `realistic_recall_v5_gate.json` |
| Size | **50** scenarios, **200** queries (100 affective / 100 affect-free `gate_label`; 100 sessions, 300 events) |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **LLM-assisted**, author-blind: `generate_v5_gate.py` with **gpt-5-mini** (`benchmarks/preregistration_addendum_q_affect_gating.md`). Mechanical lexical-leakage gate before inclusion. Frozen with Addendum Q (`e68d2cc`). |

### `shared_history_recall_v1`

| | |
| --- | --- |
| File | `shared_history_recall_v1.json` |
| Size | **12** scenarios, **36** queries, 14 sessions, 60 events. Encode-after-recall count is 24 across the whole scenario and 22 when counted within a single session. |
| Language | en |
| Licence | **CC-BY-4.0** |
| Origin | **Hand-authored** interleaved shared-history scenarios. Declared session `timeline` allows encodes after recalls. Query affect is a recorded oracle; 14/36 queries have `|valence| < 0.2` and 4 sit at `|valence| = 0.2`. No generator script. No committed scores. |
| Primary use | Opt-in `--shared-history` longitudinal replay only (protocol mode `shared_history_timeline`). Not a comparative-runner dataset. Distinct from the #170 shared-history replay on `realistic_recall_v1`. |

Supporting owner artifact **not** listed in `PUBLISHABLE.txt` (LLM audit log,
not a retrieval set): `realistic_recall_v2_topup_provenance.jsonl` (20 lines;
`it_generation` / ES translation calls).

## Excluded from the Hub dataset (kept in git)

These remain in `benchmarks/datasets/` for reproducibility. They **must not**
appear in `PUBLISHABLE.txt`.

| Set | File(s) | Counts (repo) | Origin | Licence | Attribution |
| --- | --- | --- | --- | --- | --- |
| `emobank_v1` | `emobank_v1.json` | 300 items, `sample_seed=42` | Sampled from EmoBank (JULIELab; MASC texts) | CC-BY-SA 4.0 | [`emobank_v1.ATTRIBUTION.md`](emobank_v1.ATTRIBUTION.md) |
| MADial-Bench EN | `madialbench/*.jsonl` | 160 dialogues + 160 memories | Vendored upstream, sha256-pinned | MIT | [`madialbench/ATTRIBUTION.md`](madialbench/ATTRIBUTION.md) |
| ES-MemEval / EvoEmo | `esmemeval/evo_emo.json` | 18 seekers, 401 sessions, 1,427 QA / 1,133 in-family | Vendored v1.0.0, sha256-pinned | CC-BY-4.0 | [`esmemeval/ATTRIBUTION.md`](esmemeval/ATTRIBUTION.md) |
| `dailydialog_personas_v1` | `dailydialog_personas_v1.json` | 120 personas, 396 queries, 536 sessions | Built from DailyDialog via `persona_builder.py` | CC BY-NC-SA 4.0 | [`dailydialog_personas_v1.ATTRIBUTION.md`](dailydialog_personas_v1.ATTRIBUTION.md) |

`dailydialog_personas_v1` **cannot** use the owner Hub licence: it is
DailyDialog-derived (`benchmarks/dailydialog/README.md`). No
`realistic_recall_*` set is recorded as seeded from DailyDialog or EmoBank;
if that were true it would also be excluded from the Hub licence and from
`PUBLISHABLE.txt`.

LoCoMo is used by `benchmarks/locomo/` but is **not** stored under
`benchmarks/datasets/` (absent from this directory listing).

## Dataset structure

Owner JSON sets share the realistic-replay schema (`ReplayDataset` in
`benchmarks/realistic/runner.py`): top-level `name`, `version`,
`description`, `default_top_k`, `scenarios[]` with `sessions[]` of
`events[]` and `queries[]`. Interleaved shared-history sessions also declare
`timeline[]`. Multilingual slices also have `language`.
Owner `realistic_recall_*` files and `shared_history_recall_v1.json` now carry
`"license": "CC-BY-4.0"` as the owner-decided licence line.

`affect_reference_v1.jsonl` fields: `id`, `text`, `valence`, `arousal`,
`dominance`, `expected_label`, `source` (`README.md`).

No Hub configs or Parquet shards are produced in this PR.

## Licence

| Set | Licence | Status |
| --- | --- | --- |
| `affect_reference_v1` | CC-BY-4.0 | Owner decision; declared in `README.md` |
| `realistic_recall_*` (all publishable ids) | CC-BY-4.0 | Owner decision (Gianluca); written into each JSON `license` field. |
| `shared_history_recall_v1` | CC-BY-4.0 | Owner decision (Gianluca); written into the JSON `license` field. |
| `emobank_v1` | CC-BY-SA 4.0 | Upstream; not publishable here |
| MADial-Bench vendored files | MIT | Upstream; not publishable here |
| ES-MemEval / EvoEmo | CC-BY-4.0 | Upstream; not publishable here |
| `dailydialog_personas_v1` | CC BY-NC-SA 4.0 | Upstream-derived; not publishable here |

Project **code** remains MIT (`LICENSE`, `pyproject.toml`).

## Citation

**How to credit:** use the citation below and the concept DOI
[10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258).

Use the project concept DOI for the library and these owner sets:

```bibtex
@software{mazza_emotional_memory,
  author    = {Mazza, Gianluca},
  title     = {emotional-memory: Affective Field Theory for LLM Memory},
  doi       = {10.5281/zenodo.19972258},
  url       = {https://doi.org/10.5281/zenodo.19972258},
  publisher = {Zenodo}
}
```

Version-specific DOIs live in `CITATION.cff` / `release.toml` and change per
release; the concept DOI above is the stable identifier
(`docs/contributing/ssot-policy.md`, `RELEASING.md`).

Cite third-party corpora from their ATTRIBUTION files, not as part of this
Hub dataset.
