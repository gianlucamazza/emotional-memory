# ADR 0001: External human-grounded retrieval study

Status: Accepted — user approved the implementation plan on 2026-10-04.

## Context

Positive retrieval evidence is confined to author-constructed affect-conditioned
benchmarks. Addendum T removes query-state labels but retains preset event affect.
Existing human-evaluation packets use internal scenarios. Neither establishes
naturalistic utility or the necessity of the five-layer architecture.

## Decision

Revise public claims now, independently of a new, single MSC study. Compare
text-appraised AFT with a semantic + recency + valence reranker on human top-1
relevance. Cosine and Mem0 are secondary references. Use original human turns,
prior-session exchanges only, grouped history splits, and blinded independent gold.
Freeze all decisions before confirmation; target a 5 percentage-point difference,
80% power, two-sided alpha .05. Reserve 200 development and 100 pilot histories.
The pilot determines discordance and sample size, never the confirmatory result.
Human end-to-end evaluation on 100 presampled cases is secondary.

No production API change. Benchmark-only configuration disables wall-clock decay
because MSC provides conversation order rather than reliable event timestamps.
Recency uses that order for all applicable arms. This tests retrieval utility, not
wall-clock forgetting. Use BAAI/bge-small-en-v1.5 throughout the controlled arms.
Human gold and original persona summaries never enter retrieval or appraisal.

The new protocol is a draft until source inventory, human pilot, resource estimate,
model snapshots and N are frozen. Implementation and synthetic tests do not confer
confirmatory status. No LLM ratings may substitute for recruited human raters.

## Consequences

A positive result is restricted to elicited MSC conversation and this protocol.
A negative or inconclusive result remains publishable. Annotation cost can make
the powered study infeasible; report this rather than shrink N after inspection.
No benefit claim about therapy, spontaneous customer care, or individual layers.
The unaltered historical protocols remain evidence; corrections are dated notes.

## Alternatives

- Another curated addendum: rejected because it cannot establish external utility.
- Reuse MADial/ES-MemEval as confirmation: rejected after prior outcome inspection.
- Cosine as primary: rejected because it does not test the value of complexity.
- Query-only appraisal: rejected because it leaves oracle affect at encoding.
- Primary LLM judgment: rejected because the endpoint must be human-grounded.
