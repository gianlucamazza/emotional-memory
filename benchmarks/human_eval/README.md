# Human Evaluation Pilot

This directory contains a lightweight pilot protocol for the next study step:
human judgment of affective coherence and usefulness.

The goal is not scale yet. The goal is to make the evaluation runnable,
reviewable, and consistent with the repo's current evidence ladder.

## Included assets

- `protocol.md`: pilot procedure and rating dimensions
- `pilot_v1.json`: small scenario set for manual rating
- `rater_instructions.md`: exact instructions for human raters
- `pipeline.py`: packet generation and rating-summary CLI

## Intended use

- run after realistic replay scenarios exist
- compare response traces or memory-supported outputs side by side
- collect ratings on coherence, usefulness, and emotional consistency

This pilot is preparatory. It does not claim external validity by itself.

## Pilot v1 scope

- `10` scenario packets
- `2` conditions only: `aft` and `naive_cosine`
- `4` rating dimensions per condition

This yields `20` condition-level rows per rater in `ratings.jsonl`.

## Canonical runbook

```bash
make bench-realistic
make human-eval-packets
# copy and fill benchmarks/human_eval/ratings_template.jsonl as ratings.jsonl
make human-eval-summary
```

Operational sequence:

1. Regenerate realistic benchmark artifacts.
2. Generate packets and the empty ratings template.
3. Copy `ratings_template.jsonl` to `ratings.jsonl`.
4. Give raters the packet plus [rater_instructions.md](rater_instructions.md).
5. Collect completed rows in `ratings.jsonl`.
6. Run `make human-eval-summary`.

`make human-eval-summary` now rejects untouched templates and incomplete
placeholder files. It only summarizes records that include a `rater_id` and
completed numeric ratings for all dimensions.

No checked-in `summary.json` / `summary.md` artifacts are treated as evidence
until the pilot is run with real completed ratings.

## Agreement metric (Krippendorff's alpha)

When `ratings.jsonl` contains ratings from **≥ 2 raters**, `make human-eval-summary`
automatically computes Krippendorff's alpha (ordinal) per dimension and includes
it in `summary.md` under "## Inter-Rater Agreement".

Acceptance thresholds per Krippendorff (2004):
- alpha ≥ 0.67: acceptable for tentative conclusions
- alpha ≥ 0.80: strong agreement, suitable for publication

For three raters and the current 10-scenario packet, the 95% CI on alpha is wide.
Expand to ≥ 50 scenarios before reporting headline alpha in a paper.
