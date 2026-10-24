# Benchmark Datasets

Owner-authored sets that may go to the planned Hugging Face dataset
`emotional-memory-benchmarks` are listed in [`PUBLISHABLE.txt`](PUBLISHABLE.txt)
and described in the draft Hub card [`DATASET.md`](DATASET.md). **This
repository does not upload anything to Hugging Face.** A unit test
(`tests/test_publishable_datasets.py`) fails if a third-party or
third-party-derived set appears in `PUBLISHABLE.txt`.

Owner-authored (publishable) files:

- `affect_reference_v1.jsonl`: synthetic quadrant-based retrieval probe (CC-BY-4.0)
- `realistic_recall_v1.json` … `realistic_recall_v5_gate.json`, including
  `_it` / `_es` / `_fr` and the `_noAF` / `_gate` variants, plus
  `shared_history_recall_v1.json` (licence
  **CC-BY-4.0** — owner decision; see `DATASET.md`)

Third-party or third-party-derived (kept in git, **not** publishable):

- `emobank_v1.json`: human-annotated VAD subset (CC-BY-SA 4.0) —
  [`emobank_v1.ATTRIBUTION.md`](emobank_v1.ATTRIBUTION.md)
- `madialbench/`: MADial-Bench EN (MIT) —
  [`madialbench/ATTRIBUTION.md`](madialbench/ATTRIBUTION.md)
- `esmemeval/`: ES-MemEval / EvoEmo (CC-BY-4.0) —
  [`esmemeval/ATTRIBUTION.md`](esmemeval/ATTRIBUTION.md)
- `dailydialog_personas_v1.json`: DailyDialog-derived personas
  (CC BY-NC-SA 4.0) —
  [`dailydialog_personas_v1.ATTRIBUTION.md`](dailydialog_personas_v1.ATTRIBUTION.md)

The first owner set is strongest for controlled retrieval behavior. The
`realistic_recall_*` family is a scripted multi-session bridge. Third-party
files carry their upstream licences and must not be mixed into the Hub
upload.

## affect_reference_v1

## Overview

`affect_reference_v1.jsonl` contains **258 affect-labeled text examples** for evaluating
mood-congruent retrieval and affective state tracking in LLM memory systems.

## Schema

Each line is a JSON object:

```json
{
    "id": "sv1_0000",
    "text": "I just got the promotion I've been working toward for years!",
    "valence": 0.823,
    "arousal": 0.871,
    "dominance": 0.614,
    "expected_label": "joy",
    "source": "synthetic-v1"
}
```

| Field            | Type   | Range      | Description                               |
| ---------------- | ------ | ---------- | ----------------------------------------- |
| `id`             | string | —          | Unique identifier                         |
| `text`           | string | —          | Natural-language expression               |
| `valence`        | float  | [-1, 1]    | Negative ↔ positive affect (Russell 1980) |
| `arousal`        | float  | [0, 1]     | Calm ↔ excited                            |
| `dominance`      | float  | [-1, 1]    | Submissive ↔ dominant (Mehrabian 1980)    |
| `expected_label` | string | Plutchik 8 | Expected Plutchik primary emotion         |
| `source`         | string | —          | Provenance tag                            |

## Distribution

Examples span all four Russell circumplex quadrants:

| Quadrant | Valence | Arousal | Primary emotions         | ~Count |
| -------- | ------- | ------- | ------------------------ | ------ |
| Q1       | +       | high    | joy, excitement          | 62     |
| Q2       | −       | high    | fear, anger, anxiety     | 45     |
| Q3       | −       | low     | sadness, depression      | 75     |
| Q4       | +       | low     | calm, contentment, trust | 76     |

Each quadrant has three intensity tiers (strong / moderate / mild).
Two independent PAD-jittered samples are generated per seed text.

## Generation

```bash
python benchmarks/datasets/generate_dataset.py
```

Requires only the Python standard library. Set `seed` for reproducibility.

## License

Texts are original synthetic compositions. Dataset is released under **CC-BY-4.0**.
Credit the citation and concept DOI [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258)
(see `DATASET.md`).

## Version history

| Version | Examples | Notes                                                   |
| ------- | -------- | ------------------------------------------------------- |
| v1      | 258      | Initial release — synthetic, 12 buckets, 2 samples/text |

## realistic_recall_v1

`realistic_recall_v1.json` contains replayable multi-session scenarios used by
`benchmarks/realistic/runner.py`.

Schema highlights:

- scenario identifier and description
- multiple sessions per scenario
- per-session affect-tagged events
- natural-language retrieval queries with expected target memories
- optional query-time affective state

**v1.4**: 50 scenarios, 100 queries — challenge-type distribution: 30
`semantic_confound`, 25 `affective_arc`, 25 `same_topic_distractor`, 20
`recency_confound`. Rebalanced to tighten per-challenge confidence intervals.
Its role is to support realistic replay studies, not to act as a broad
leaderboard.

## realistic_recall_v2

`realistic_recall_v2.json` is the current primary benchmark used in all v0.7+ evidence
claims. It is a superset of v1 with 5 challenge types and doubled query count.

**v2**: 50 scenarios, 200 queries — challenge-type distribution:
`affective_arc`, `momentum_alignment`, `same_topic_distractor`,
`semantic_confound`, `recency_confound`. Added `momentum_alignment` to directly
probe the affective-momentum layer.

Used in: Hd2 (Gate 3 confirmatory), G4/G5 cross-embedder, S3 ablation.

## realistic_recall_v2_it

`realistic_recall_v2_it.json` is the Italian-language slice used in G6 (multilingual).

**v2_it (file today):** 30 scenarios, 120 queries (G6 started at 20/80;
scenarios 21–30 are the Addendum Hd2 power top-up). Same schema as v2,
translated to Italian for the first 20; the extra 10 are LLM-assisted
(`scripts/generate_realistic_v2_topup.py`, provenance
`realistic_recall_v2_topup_provenance.jsonl`). Challenge types:
`affective_arc`, `momentum_alignment`, `same_topic_distractor`,
`semantic_confound`, `recency_confound`. Used with SBERT bge-small-en-v1.5
and multilingual-e5-small. IT me5 at N=120 **FAIL** at declared power —
see `DATASET.md`.

## realistic_recall_v2_es

`realistic_recall_v2_es.json` is the Spanish-language slice used in Hd2_ES.

**v2_es (file today):** 30 scenarios, 120 queries (Hd2_ES started at 20/80;
scenarios 21–30 translated from the Italian top-up). Same schema as v2,
localized to Spanish for the first 20. ES me5 at N=120 **FAIL** at declared
power — see `DATASET.md`.

## realistic_recall_v2_fr

`realistic_recall_v2_fr.json` is the French-language slice used in Addendum M (Hm1).

**v2_fr**: 30 scenarios, 60 sessions, 210 events, 120 queries — hand-authored in native
French, 2-session design (session_1 encode, session_2 query). Challenge types:
`affective_arc`, `momentum_alignment`, `recency_confound`, `same_topic_distractor`,
`semantic_confound` (24 queries per type). Used with multilingual-e5-small.

Result: Addendum M Branch A PASS — AFT top1=0.31 [0.23, 0.39] vs naive_cosine=0.12
[0.07, 0.18], Δ=+0.18 [0.11, 0.26], p_bootstrap<0.0001, Hedges g=0.424.
See `benchmarks/preregistration_addendum_m_fr_closure.md`.

## realistic_recall_v3

`realistic_recall_v3.json` extends v2 with 75 new scenarios (script
`scripts/generate_realistic_v3.py`, commit `320cd23`). **v3**: 125
scenarios, 500 queries. Used by Addendum I (Hi3).

## realistic_recall_v3_noAF

`realistic_recall_v3_noAF.json` is the affect-free variant of v2 (valence /
arousal and query `state` stripped) for Addendum G (Hg1). **v3_noAF**: 50
scenarios, 200 queries.

## realistic_recall_v4_noAF

`realistic_recall_v4_noAF.json` is a leakage-free affect-free set disjoint
from v3 (`pNN_` ids), LLM-drafted by `generate_v4_noAF.py` (gpt-5-mini) for
Addendum P. **v4_noAF**: 40 scenarios, 160 queries.

## realistic_recall_v5_gate

`realistic_recall_v5_gate.json` is the mixed-gate set for Addendum Q
(`qNN_` ids), LLM-drafted by `generate_v5_gate.py` (gpt-5-mini).
**v5_gate**: 50 scenarios, 200 queries (100/100 `gate_label`).

## shared_history_recall_v1

`shared_history_recall_v1.json` is a hand-authored shared-history dataset for
interleaved encode/retrieve order. Sessions declare a `timeline` so encodes
can follow recalls (for example `E E E R E R E R`). A documented share of
queries is near-neutral (`|valence| < 0.2`); a few sit on the gate boundary
(`|valence| = 0.2`). Use only with `--shared-history --embedder hash`.
Protocol mode is `shared_history_timeline`. No scores are published from this
file. The #170 shared-history replay on `realistic_recall_v1` is unchanged.

Licence for every `realistic_recall_*` file above: **CC-BY-4.0**
(JSON `license` field; owner decision). See `DATASET.md`.

## dailydialog_personas_v1

`dailydialog_personas_v1.json` is **DailyDialog-derived** (120 personas,
396 queries) and is **not** owner-original. Licence: CC BY-NC-SA 4.0.
See [`dailydialog_personas_v1.ATTRIBUTION.md`](dailydialog_personas_v1.ATTRIBUTION.md)
and `benchmarks/dailydialog/README.md`. Not listed in `PUBLISHABLE.txt`.

## emobank_v1

`emobank_v1.json` is a **human-annotated** affect subset used to validate the
appraisal signal against human labels (Addendum S, A5). Unlike the synthetic and
hand-authored sets above, this is **third-party data** and is **not** original to
this project.

**emobank_v1**: 300 rows sampled (`seed=42`, filtered to ≥4 words) from EmoBank.
Each row carries the original human Valence/Arousal/Dominance ratings (1–5 scale)
plus a normalized `human` block mapped to `CoreAffect` ranges
(`valence=(V−3)/2 ∈ [−1,1]`, `arousal=(A−1)/4 ∈ [0,1]`, `dominance=(D−1)/4 ∈ [0,1]`).

Result: Addendum S — `LLMAppraisalEngine` valence human-validated (Pearson
r=0.70 [0.66, 0.75]); arousal (r=0.28) and dominance (r=0.33) weak-to-moderate;
+0.15 valence bias persists; `KeywordAppraisalEngine` not human-validated (r=0.07).
See `benchmarks/preregistration_addendum_s_human_gold_appraisal_closure.md`.

### Source & License

- **Source:** EmoBank — JULIELab/EmoBank (<https://github.com/JULIELab/EmoBank>);
  texts from the Manually Annotated Sub-Corpus (MASC).
- **Citation:** Buechel, S., & Hahn, U. (2017). _EmoBank: Studying the Impact of
  Annotation Perspective and Representation Format on Dimensional Emotion
  Analysis._ EACL 2017, 578–585.
- **License:** **CC-BY-SA 4.0** (share-alike). This applies to `emobank_v1.json`
  only; it does **not** change the MIT license of the project's code. Redistribution
  of this file or derivatives must preserve attribution and the CC-BY-SA 4.0 terms.

---

## madialbench/ (Addendum X — third-party retrieval benchmark)

`madialbench/MADial-Bench-en-dialogue.jsonl` and
`madialbench/MADial-Bench-en-memory.jsonl` are the English split of
**MADial-Bench** (160 memory-anchored dialogues + 160-memory bank with ranked
gold recall sets and per-memory emotion labels), vendored byte-identical from
the upstream repo at commit `572e3a10d6d01852a65e4508e0b3ab2a00d0710c` and
pinned by sha256 in `benchmarks/preregistration_addendum_x_madialbench_third_party.md`
(the loader refuses to run on mismatching bytes). Like `emobank_v1.json`, this
is **third-party data**, not original to this project.

### Source & License

- **Source:** MADial-Bench — hejunqing/MADial-Bench
  (<https://github.com/hejunqing/MADial-Bench>).
- **Citation:** He, J., et al. (2025). _MADial-Bench: Towards Real-world
  Evaluation of Memory-Augmented Dialogue Generation._ NAACL 2025 (Long),
  aclanthology.org/2025.naacl-long.499; arXiv:2409.15240.
- **License:** **MIT** (per the upstream repository's LICENSE). Applies to the
  two vendored files; attribution preserved above.

---

## esmemeval/ (Addendum X2 — third-party retrieval benchmark)

`esmemeval/evo_emo.json` is the **EvoEmo** corpus from the **ES-MemEval v1.0.0**
release (18 seekers, 401 multi-session emotional-support conversations spanning
~15 months, 1,427 QA questions with turn-level evidence refs), vendored
byte-identical from the v1.0.0 release archive (GitHub tree `6926242`, Zenodo
deposit of the same snapshot) and pinned by sha256 in
`benchmarks/preregistration_addendum_x2_esmemeval_third_party.md` (the loader
refuses to run on mismatching bytes). Like `madialbench/`, this is
**third-party data**, not original to this project. Note (declared in the
pre-registration): the released artifact contains 1,427 QA questions while the
paper reports 1,209; the hash-pinned artifact is treated as ground truth.

### Source & License

- **Source:** ES-MemEval — slptongji/ES-MemEval
  (<https://github.com/slptongji/ES-MemEval>); Zenodo DOI
  [10.5281/zenodo.18338564](https://doi.org/10.5281/zenodo.18338564).
- **Citation:** Chen, T., Lu, J., Shen, Y., & Zhang, L. (2026). _ES-MemEval:
  Benchmarking Conversational Agents on Personalized Long-Term Emotional
  Support._ The Web Conference (WWW) 2026, doi:10.1145/3774904.3792143;
  arXiv:2602.01885.
- **License:** **CC-BY-4.0** (per the Zenodo record). Applies to the vendored
  file; attribution preserved above. Redistribution of this file or derivatives
  must preserve attribution per the CC-BY-4.0 terms.
