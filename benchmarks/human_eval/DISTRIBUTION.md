# Human Eval Distribution Guide

This document describes how to recruit raters, distribute evaluation packets,
and collect completed ratings for the AFT human-evaluation pilot.

## Who to recruit

Target **5–10 raters** for the pilot (Addendum C minimum: 5 raters, N=10
scenarios × 2 conditions = 20 rows per rater).

Suitable rater profiles (any combination):
- Colleagues or co-authors with LLM/NLP background
- Cognitive science or psychology researchers
- Software engineers who have worked with conversational AI
- Graduate students in HCI, AI, or psychology

**Exclusion criterion:** anyone who contributed to the AFT implementation or
dataset construction. Avoid conflicts of interest.

## Materials to send each rater

1. **`rater_instructions.md`** — send as-is (or paste into the email/form)
2. **`rater_faq.md`** — optional but recommended for first-time raters
3. The rating **CSV template** (`ratings_template.csv`) — generated from the
   Google Sheets template at `ratings_template_gsheets.csv`
4. The **scenario packet** from `packets.json` — either as JSON or as a
   rendered HTML/PDF (see below for rendering)

## Generating packets for distribution

```bash
make bench-realistic          # regenerate AFT and naive_cosine outputs
make human-eval-packets       # writes benchmarks/human_eval/packets.json
```

To export packets as human-readable HTML (one file per scenario), run:

```bash
uv run python -m benchmarks.human_eval.pipeline export --format html
```

Each HTML file contains the scenario narrative and both condition outputs,
formatted for easy side-by-side review.

## Distribution channels

### Option A: Google Sheets (recommended for remote raters)

1. Import `ratings_template_gsheets.csv` into a new Google Sheet
2. Create one sheet per rater (or one shared sheet with a `rater_id` column)
3. Share the sheet with raters alongside the packet HTML files
4. When ratings are collected, export as CSV and run:
   ```bash
   uv run python -m benchmarks.human_eval.pipeline import-csv --input ratings.csv
   ```

### Option B: JSONL file (recommended for local/co-located raters)

1. Copy `ratings_template.jsonl` to `ratings.jsonl`
2. Share the file with the rater along with `packets.json`
3. Rater fills in `rater_id` and numeric ratings for each row
4. Collect completed `ratings.jsonl` files from each rater and merge:
   ```bash
   cat rater1_ratings.jsonl rater2_ratings.jsonl > ratings.jsonl
   make human-eval-summary
   ```

## Collecting and merging ratings

Once you have completed ratings from all raters:

```bash
# If using JSONL:
cat rater*_ratings.jsonl > benchmarks/human_eval/ratings.jsonl
make human-eval-summary

# If using CSV:
uv run python -m benchmarks.human_eval.pipeline import-csv \
    --input completed_ratings.csv \
    --out benchmarks/human_eval/ratings.jsonl
make human-eval-summary
```

`make human-eval-summary` will:
- Validate that `rater_id` is non-empty on every row
- Compute per-dimension means and CIs
- Compute Krippendorff's alpha (ordinal) when ≥ 2 raters are present
- Write `benchmarks/human_eval/summary.{json,md}`

## Acceptance thresholds (Addendum C)

From `benchmarks/preregistration_addendum_v2.md`:

| Criterion | Threshold |
|---|---|
| Minimum raters | 5 |
| Inter-rater reliability (Krippendorff α) | ≥ 0.67 for "acceptable agreement" |
| Minimum scenarios | 10 |
| Holm-corrected p | < 0.05 on AFT vs naive_cosine for ≥ 1 dimension |

Results below these thresholds are reported as pilot data only; they do **not**
license "human validation" claims in the paper or README.

## Checklist before launching

- [ ] `benchmarks/human_eval/packets.json` is fresh (run `make human-eval-packets`)
- [ ] Packets reviewed manually for content quality
- [ ] `ratings_template.jsonl` is correct (run `make human-eval-packets` to regenerate)
- [ ] At least 5 raters confirmed before launch
- [ ] Raters have received `rater_instructions.md`
- [ ] Pre-reg file `benchmarks/preregistration_addendum_v2.md` committed (it is)

## After collection

Commit results as:
- `benchmarks/human_eval/ratings.jsonl` (raw ratings, one row per scenario×condition×rater)
- `benchmarks/human_eval/summary.json` (aggregate + Krippendorff alpha)
- `benchmarks/human_eval/summary.md` (human-readable)

These artifacts are **not** gitignored (unlike checkpoint files). Once committed
they become part of the reproducibility record.
