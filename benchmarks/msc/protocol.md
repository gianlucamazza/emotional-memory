# MSC human-grounded retrieval: protocol draft

**Status: PREPARATORY / NOT EXECUTED.** This is not a completed preregistration.
No human ratings, pilot outcomes, powered sample size or confirmatory result exist.
The implementation plan was accepted on 2026-10-04; see
[ADR 0001](../../docs/research/decisions/0001-external-human-retrieval.md).
Freeze and commit the registration before any confirmatory system execution.

## Question and scope

Does text-appraised AFT improve human top-1 relevance over a semantic + recency +
valence reranker on external conversation? One primary contrast, AFT minus hybrid.
Cosine and Mem0 are secondary references. Human response judgments are secondary
and exploratory. A positive outcome does not establish per-layer necessity.

MSC consists of human **elicited** persona conversations, not spontaneous care or
therapy. Reference: [Xu, Szlam and Weston (2022)](https://aclanthology.org/2022.acl-long.356/).
It was mentioned but not previously evaluated in this repository. Development and
pilot are exploratory; confirmation uses untouched histories. Public-corpus model
pretraining contamination cannot be excluded and must be disclosed.

## Source and sampling

- Source: `https://parl.ai/downloads/msc/msc_v0.1.tar.gz`.
- Upstream v0.1 SHA256:
  `e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60`.
  Verify against the [ParlAI builder](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/msc/build.py).
- Native dialogue records from sessions 2–5 across original splits are pooled;
  this is a new retrieval study, **not** replication of the upstream task/splits.
- Union records sharing initial-data IDs or exact normalized antecedent/current
  dialogues. One unit per connected history; choose its longest available history,
  break ties by record hash. Shared personas alone are not independent participant
  identifiers; the statistical unit is a history, not a unique crowdworker.
- Select one original current-session turn with a following response, uniformly
  with seed `20261004` and history ID. Export its preceding current context, never
  its future response. No author/LLM rewrite or affect-based selection.
- Memories are consecutive two-turn exchanges from previous sessions; a terminal
  unmatched turn is retained. No cross-user distractors, persona summaries, source
  emotion labels or current-session turns enter the memory pool.
- Shuffle grouped histories with seed `20261004`: first 200 development, next 100
  pilot, remainder reserve. No movement between these partitions after inspection.
- Annotate every candidate before ranking. Exclude no-relevant-memory queries using
  majority human gold, report their count and retain their raw annotations. Select
  the first N eligible reserve cases in the already fixed order. No ranking-based
  exclusions. Pilot effective N can be below 100 after relevance exclusions.

The locally verified inventory is retained in `source_inventory.json`: 13,005
native records, 5,001 grouped histories, 200 development, 100 pilot and 4,701
reserve. The reserve contains 79,304 candidate exchanges (237,912 relevance
judgments for three raters), before no-relevant-memory exclusions. This is an
inventory, not a result or a feasibility commitment.

The archive has no license file. Confirm data-use and redistribution terms from
an authoritative source before distributing packets. Do not infer a data license
from ParlAI's software license. Record that source in the readiness record.

## Human relevance and blinding

Three independent raters label every candidate `relevant` (0/1): would recalling
this exchange concretely help answer the current turn, given the current context?
Merely sharing an emotion is insufficient; conflicting or outdated facts can be
relevant when needed to explain change. A second case-level `affect_required`
flag asks whether emotional context is necessary to decide relevance; it is
secondary and must be constant across a rater's candidates for that case.

Each rater sees the full chronological prior history and the position of the
candidate exchange, to judge relevance despite updates or contradictions.
Each rater gets their own randomized cards, with opaque item IDs and no system,
rank, gold or affect metadata. Keep `private-key.json` away from raters. The three
pseudonymous raters must complete all items independently. Majority labels define
gold; retain disagreement and notes rather than adjudicating towards a system.
Report nominal Krippendorff alpha using the existing human-evaluation helper;
constant ratings yield undefined agreement rather than a spurious perfect value.
If the pilot reveals unusable instructions or agreement below .67, stop and
revise the protocol before confirmation; do not recycle that pilot as new evidence.

## Systems

All arms receive identical source exchanges and current context + query. AFT,
hybrid and cosine share exact cached BAAI/bge-small-en-v1.5 vectors; pin the model
revision. Appraise only source text and current query/context using DIRECT_VAD,
with `fallback_on_error=False`. AFT and hybrid share these appraisal outputs.

- **AFT:** existing public encode/retrieve APIs, text-appraised memories and query.
  No gates or weight fitting. Default config except wall-clock decay/retrieval
  boosts disabled as in external-corpus adapters. Deterministic ordinal one-second
  clock for event order; these are not observed times. This does not test forgetting.
- **Hybrid:** normalized cosine `(cos+1)/2`, ordinal recency `(order+1)/pool_size`,
  valence proximity `1-|query_valence-memory_valence|/2`. Nonnegative simplex weights
  on a .1 grid, maximizing development majority-gold top-1. Ties favor semantic,
  then recency weight. Candidate score ties use source ID order.
- **Cosine:** the same semantic feature alone.
- **Mem0:** controlled original-exchange storage/search with `infer=False`, preserving
  the annotated candidate units. Report it as **Mem0 fixed-unit retrieval**, not
  the full extraction/consolidation pipeline. The adapter targets the locked
  `mem0ai==2.0.20` API (`filters`, `top_k`, threshold 0, no reranker). Provider, embedding and package
  configuration are pinned and disclosed; differences from the controlled arms
  remain a secondary-comparison limitation. Unknown IDs, merges or mismatched
  source metadata abort the run, rather than becoming false misses or fabricated IDs.

Top-k is five for returned context; top-1 is primary. Fresh per-case stores prevent
cross-case state. Memory contents used by downstream generation are the original
source exchanges for every arm. Appraisal models/endpoints, generation snapshot,
Mem0 configuration, dependency versions and tuning provenance must be frozen.
No hidden parameter sweep on pilot or test.

## Power and inference

- Effect for planning: absolute delta +.05. Two-sided alpha .05, power .80.
- Pilot discordance q gets a one-sided 95% Clopper–Pearson upper limit. Clamp q
  to at least .05 for feasibility of the alternative; conditional win probability
  is `(q+.05)/(2q)`.
- For each N, sum exact conditional McNemar rejection probabilities over
  `M ~ Binomial(N,q)`. Choose the smallest N reaching .80. No observed effect size
  drives N. Resource/corpus cap is declared before the calculation.
- If the available eligible histories or annotation resources cannot support N,
  report infeasibility. Do not relax power, switch corpus or add a post-result top-up.
- Report all four paired counts, absolute delta, two-sided exact McNemar p, and
  95% paired percentile bootstrap CI (10,000 draws, seed `20261004`). Resample
  histories, not individual raters. No multiplicity correction for the sole primary;
  secondary comparisons are explicitly exploratory, without confirmatory p claims.
- Superiority: positive delta and p < .05. Practical interpretation separately:
  CI upper < .05 excludes a five-point gain under this analysis; lower >= .05
  supports at least that gain; otherwise it remains compatible but unestablished.
  A non-significant test is not an equivalence result. Do not generalize from this
  corpus to arbitrary naturalistic tasks.

## End-to-end secondary endpoint

Presample 100 of the selected confirmatory IDs by a separate deterministic hash
order at freeze time. Generate one answer per arm using the same snapshot,
prompt and retrieved-context character budget (6,000 characters, up to five
exchanges). The current context is identical across arms. Store exact responses;
invalid/failed calls abort rather than invite selective regeneration.

Three raters, disjoint from the gold panel, independently score usefulness and
correctness on 1–5 scales. Show current context and full prior history for factual
checking, but hide the retrieval system and order responses independently.
1 = unusable/incorrect; 3 = mixed/partially supported; 5 = useful/fully supported.
Average ratings per case before paired bootstrap; retain distributions and ordinal
agreement. These results cannot replace a failed primary.

## Readiness, immutable artifacts and failures

Before freeze: license evidence, six recruited raters (three per panel), pilot
agreement, eligible reserve count, powered N, measured minutes per annotation,
total hours/cost, model-snapshot evidence, provider configuration and package pins.
Record raw submissions and input/result hashes. Compute the cost from observed
pilot annotation time, not invented throughput. Example readiness schema:

```json
{
  "license_evidence": "authoritative data-terms URL and review date",
  "recruitment_record": "path to confirmed pseudonymous roster",
  "resource_estimate": "path to pilot-based hours and cost estimate",
  "model_snapshot_evidence": "path to pinned model/provider evidence",
  "gold_raters": ["g1", "g2", "g3"],
  "response_raters": ["e1", "e2", "e3"]
}
```

Registration binds code, corpus manifest, cases, gold, config, pilot power and
readiness. Commit registration before confirmation. The CLI refuses overwrite,
wrong split, changed registration inputs, silent missing arms or incomplete human
ratings. Started/failed markers preserve attempted runs. A failure requires a
recorded deviation/recovery decision before another run; changing output filenames
does not create a new independent trial. Do not claim technical guards prevent a
malicious operator from fabricating human provenance.

## Commands

Run from the repository root. Outputs and corpus stay outside tracked data unless
explicitly selected for the registration/evidence record. No scored run is automatic.
Use `uv run python -m benchmarks.msc.cli --help` for argument names.

1. `prepare --archive /path/msc_v0.1.tar.gz --out /path/msc`
2. `packets --cases /path/msc/development.json --raters g1 g2 g3 --out /path/dev-packets`
3. Collect actual human files; `gold --cases ... --key ... --ratings r1.json r2.json r3.json --out ...`
4. Pin config (see `config.example.json`); `run --stage development --manifest ... --cases ... --config ... --out ...`
5. `tune --cases ... --gold ... --results ... --config ... --out ...`; use the resulting config for pilot and confirmation.
6. Prepare/collect pilot gold; `run --stage pilot ...`, then `analyze --cases ... --gold ... --results ... --out ...`.
7. `power --pilot_report ... --maximum AVAILABLE_CAP --out ...`; annotate reserve and verify agreement/resources.
8. `freeze --cases RESERVE --gold RESERVE_GOLD --config ... --manifest ... --power ... --readiness ... --out REGISTRATION_DIRECTORY`.
9. Commit `registration.json` before `run --stage confirmatory --registration ... --manifest ... --cases ... --config ... --out ...`.
10. `analyze ... --registration ...`; `generate ...`; `response-packets ...`; collect independent human files; `response-summary ...`.

The examples use explicit paths to prevent accidental reuse of historical datasets.
`config.example.json` is deliberately non-runnable until real model/dependency pins
are supplied. Optional runtime extras: `sentence-transformers`, `mem0`, `llm-test`.
No public library API changes. No historical preregistration is edited.
