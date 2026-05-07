# DailyDialog Affect-Conditioned Retrieval Benchmark (Hk1)

Tests whether AFT outperforms naive cosine on affect-conditioned memory retrieval
using [DailyDialog](https://aclanthology.org/I17-1099/) (Li et al. 2017, IJCNLP).

Pre-registration: `benchmarks/preregistration_addendum_k_dailydialog.md`

## Quick start

```bash
# 1. Build personas once (requires: pip install datasets)
make build-dailydialog-personas

# 2. Run benchmark
make bench-dailydialog

# 3. Dry run (5 personas, fast)
make bench-dailydialog-dry
```

## How it works

1. **Persona builder** (`persona_builder.py`): downloads DailyDialog from HuggingFace Hub
   (`daily_dialog`), filters dialogs with ≥30% emotion-bearing turns, concatenates
   4–5 dialogs per synthetic persona, and serialises to
   `benchmarks/datasets/dailydialog_personas_v1.json`.

2. **Query generator** (`query_generator.py`): derives 4 affect-conditioned queries per
   persona from emotion labels only — no LLM in-loop, no data-snooping risk:
   - `emotion_state_recall` — which session felt a specific emotion?
   - `affect_conditioned_content` — when feeling X, what was discussed?
   - `affective_trajectory` — which session shifted valence direction?
   - `cross_session_control` — among same-topic sessions, most calm/animated?

3. **Runner** (`runner.py`): ingests sessions into AFT and naive_cosine adapters,
   runs retrieval per query, computes top1_accuracy / hit@k, paired bootstrap n=10,000,
   McNemar, Holm-Bonferroni m=4.

## Environment

No LLM key required — retrieval is embedding-only (no answer generation).

The persona builder requires `datasets`:

```bash
pip install datasets
# or: uv add datasets --group dev
```

## Output files

| File | Content |
|---|---|
| `results.json` | Full results + per-type stats + pairwise comparisons |
| `results.md` | Human-readable report |
| `results.protocol.json` | Benchmark protocol metadata |
| `results.checkpoint.jsonl` | (if run with checkpoint support) |

## Dataset license

DailyDialog is released under CC BY-NC-SA 4.0.
The raw corpus is NOT committed to this repository; it is downloaded on-demand
to the HuggingFace cache directory via `datasets.load_dataset("daily_dialog")`.
The pre-built personas JSON (`dailydialog_personas_v1.json`) is derived from the
corpus and is included for reproducibility under the same CC BY-NC-SA 4.0 terms.
