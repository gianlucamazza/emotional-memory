# Benchmark Datasets

This directory currently contains two dataset families:

- `affect_reference_v1.jsonl`: synthetic quadrant-based retrieval probe
- `realistic_recall_v1.json`: scripted multi-session replay scenarios

The first is strongest for controlled retrieval behavior. The second is a
small but more realistic bridge toward multi-session evaluation.

## affect_reference_v1

## Overview

`affect_reference_v1.jsonl` contains **258 affect-labeled text examples** for evaluating
mood-congruent retrieval and affective state tracking in LLM memory systems.

## Schema

Each line is a JSON object:

```json
{
  "id":             "sv1_0000",
  "text":           "I just got the promotion I've been working toward for years!",
  "valence":        0.823,
  "arousal":        0.871,
  "dominance":      0.614,
  "expected_label": "joy",
  "source":         "synthetic-v1"
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `id` | string | — | Unique identifier |
| `text` | string | — | Natural-language expression |
| `valence` | float | [-1, 1] | Negative ↔ positive affect (Russell 1980) |
| `arousal` | float | [0, 1] | Calm ↔ excited |
| `dominance` | float | [-1, 1] | Submissive ↔ dominant (Mehrabian 1980) |
| `expected_label` | string | Plutchik 8 | Expected Plutchik primary emotion |
| `source` | string | — | Provenance tag |

## Distribution

Examples span all four Russell circumplex quadrants:

| Quadrant | Valence | Arousal | Primary emotions | ~Count |
|---|---|---|---|---|
| Q1 | + | high | joy, excitement | 62 |
| Q2 | − | high | fear, anger, anxiety | 45 |
| Q3 | − | low | sadness, depression | 75 |
| Q4 | + | low | calm, contentment, trust | 76 |

Each quadrant has three intensity tiers (strong / moderate / mild).
Two independent PAD-jittered samples are generated per seed text.

## Generation

```bash
python benchmarks/datasets/generate_dataset.py
```

Requires only the Python standard library. Set `seed` for reproducibility.

## License

Texts are original synthetic compositions. Dataset is released under **CC0 1.0**
(public domain dedication) — use without restriction.

## Version history

| Version | Examples | Notes |
|---|---|---|
| v1 | 258 | Initial release — synthetic, 12 buckets, 2 samples/text |

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
