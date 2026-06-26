# Closure — Addendum U (B.1): Circularity audit of realistic_recall_v2

**Status:** EXECUTED 2026-06-27 · **Hu1 PASS**
**Pre-registration:** `preregistration_addendum_u_circularity_audit.md` (committed before execution)
**Runner:** `benchmarks/circularity_audit/runner.py` · **Artifact:** `benchmarks/circularity_audit/results.{json,md}`
**Config:** `realistic_recall_v2` (N=200), `sbert-bge`, paired bootstrap n=2000, seed=42. No LLM.

## Result

**AFT-favorable fraction: 62.5%** (125/200 queries fall in the
`affect_only_can_help` cell = not cosine-solvable AND affect-separating).

| Cell                   |   N | AFT top1 | cosine top1 | Δ [95% CI]                  |      p |
| ---------------------- | --: | -------: | ----------: | --------------------------- | -----: |
| `affect_only_can_help` | 125 |    0.304 |       0.000 | **+0.304 [+0.224, +0.384]** | <0.001 |
| `neutral`              |  75 |    0.880 |       0.867 | **+0.013 [0.000, +0.040]**  |   0.63 |
| overall                | 200 |    0.520 |       0.325 | +0.195 [+0.145, +0.250]     | <0.001 |

2×2 partition: not-cosine-solvable × affect-separating = 125; cosine-solvable (any) = 65;
not-cosine-solvable × not-separating = 10.

**Hu1 PASS.** The aggregate +0.195 advantage is **entirely concentrated** in the
affect-favorable cell (Δ=+0.304); on the **neutral subset AFT does not beat cosine**
(Δ=+0.013, CI includes 0, p=0.63).

### Hu2 — affect-separating vs author `challenge_type`

| challenge_type        | affect-separating | not-separating |
| --------------------- | ----------------: | -------------: |
| affective_arc         |                39 |              1 |
| momentum_alignment    |                39 |              1 |
| recency_confound      |                39 |              1 |
| same_topic_distractor |                35 |              5 |
| semantic_confound     |                32 |              8 |

The data-driven `affect-separating` metric flags **all five** challenge types as
overwhelmingly affect-separating — including `recency_confound`, which the paper
describes as the one "valence-neutral" type. The author's per-type labels therefore
_understate_ how affect-favorable the construction is: by the geometry of (valence,
arousal) vs the query `state`, the gold memory is the affect-closest candidate in
~88–98% of queries across every type.

## Interpretation (honest)

The §2.4 circularity concern is now quantified. **The advantage is real where affect
discriminates** (Δ=+0.304 is large and highly significant), but it is **confined to
that regime**, and the benchmark over-represents that regime: **62.5% of queries are
constructed so that affect can break a semantic tie**, and essentially the entire
aggregate +0.195 comes from them. On the 37.5% where affect cannot help (cosine
already solves it, or affect does not separate the gold), AFT is statistically
indistinguishable from cosine (Δ=+0.013, ns).

This does **not** make the win fake — when affect is discriminative and supplied
(oracle), AFT genuinely converts it. It bounds the _generality_ of the headline
number: a corpus with a lower affect-discriminative proportion would show a
proportionally smaller aggregate AFT advantage. The +0.205/+0.18 headline figures
should be read as "the advantage on a benchmark that is ~62% affect-discriminative by
construction", not as a regime-independent effect size.

Caveat: the `neutral` cell is dominated by cosine-solvable queries (cosine top1 0.867,
near ceiling), so the small neutral Δ partly reflects limited headroom, not only
affect-irrelevance. The load-bearing finding is the **62.5% favorable fraction** and
that AFT's gains live entirely in that subset.

## Claim-matrix impact

No new claim. Bounds the existing `replayable_multi_session_help` /
`realistic_replay_vs_sota` evidence: the v2 advantage is concentrated in the
affect-discriminative ~62% of queries and is null on the rest. `08_limitations` §2.4
updated to replace "has not been audited" with these numbers.
