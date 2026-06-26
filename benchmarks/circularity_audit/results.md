# Circularity Audit of realistic_recall_v2 (Addendum U)

Dataset: `realistic_recall_v2` v2.0 · embedder: `sbert-bge` · N=200 · bootstrap n=2000 · seed=42.

Cells computed from data + the headline embedder, **not** from the author's `challenge_type`. `affect_only_can_help` = `not cosine-solvable AND affect-separating`.

**AFT-favorable fraction:** 62.5% of queries (125/200).

## AFT-cosine top-1 Δ by cell

| Cell | N | AFT | cosine | Δ [95% CI] | p |
|---|---:|---:|---:|---|---:|
| affect_only_can_help | 125 | 0.3040 | 0.0000 | +0.3040 [+0.2240, +0.3840] | 0.0000 |
| neutral | 75 | 0.8800 | 0.8667 | +0.0133 [+0.0000, +0.0400] | 0.6265 |
| overall | 200 | 0.5200 | 0.3250 | +0.1950 [+0.1450, +0.2500] | 0.0000 |

**Hu1** (advantage concentrated in favorable cell + neutral Δ CI includes 0): ✅ PASS — benchmark is AFT-favorable by construction.

## 2x2 partition

| cosine_solvable | affect_separating | N |
|---|---|---:|
| False | False | 10 |
| False | True | 125 |
| True | False | 6 |
| True | True | 59 |

## Hu2 — affect-separating vs author `challenge_type`

| challenge_type | affect-separating | not-separating |
|---|---:|---:|
| affective_arc | 39 | 1 |
| momentum_alignment | 39 | 1 |
| recency_confound | 39 | 1 |
| same_topic_distractor | 35 | 5 |
| semantic_confound | 32 | 8 |
