# Addendum AA — Wrong-person emotional memory retrieval

**Status:** Protocol fixed before scored execution; exploratory, no PASS/FAIL claim.
**Corpus:** pinned ES-MemEval/EvoEmo v1.0.0 from Addendum X2.
**Question:** Does AFT rank a falsely attributed episode above cosine more often?

## Design

- Select 50 queries from each of information extraction, user modeling, temporal
  reasoning, and conflict detection with `Random(42).sample`, sorted by query ID
  within each stratum (N=200). Exclude the three in-family abstention questions.
- Reuse X2's pinned corpus, BGE small English embedder, 50-candidate pools, and
  time-invariant decay. The clean pool is unchanged. For each query, choose the
  highest-cosine session from another seeker that is not in its clean pool (ties:
  session key). Rewrite only the seeker's speaker prefix to the target seeker's
  name. Give this document a unique decoy key. Replace the lowest-cosine
  cross-seeker candidate in the clean pool (ties: key); never replace gold.
  Selection must not use affect values or retrieval outcomes.
- Score both the clean and altered pool with identical query, embeddings,
  appraisal, and candidate set across arms. Rank the entire 50-item pool before
  computing top-1 and top-4 metrics. The decoy is always non-gold.
- Arms: cosine, full AFT with direct-VAD LLM appraisal at encode and query time,
  and separate AFT ablations without s2 mood congruence, s3 affect proximity,
  or s6 resonance. Keep every other configuration identical. Cache appraisal
  for each distinct text; errors are fatal, with no keyword fallback. No answer
  generator or judge is used.

## Measures

- Primary descriptive contrast: paired difference in decoy@4 rate, full AFT
  minus cosine, on altered pools.
- Secondary: decoy@1; gold recall@4 before and after insertion; full AFT minus
  each ablation on decoy@4; breakdown by capability. Report paired bootstrap
  95% intervals (10,000 resamples, seed 0), without a significance gate.
- Save query IDs, source/target seeker and session keys, replaced key, ranks,
  gold keys, and appraisal values for an auditable sample. Manually review the
  first 20 selected cases by query ID for coherent speaker substitution and
  plausibility, recording failures without changing the scored sample.

## Interpretation boundary

This tests retrieval susceptibility to a wrong-person episode under a semantic
adversary. The injected document is intentionally more query-similar than a
random cross-seeker document; it is not a prevalence estimate for naturally
occurring false memories. The corpus has a narrow emotion distribution, and
the score does not establish that an answer generator would assert the decoy as
fact. Results cannot change AFT's evidence claims or public API by themselves.
