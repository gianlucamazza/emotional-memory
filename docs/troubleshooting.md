# Troubleshooting

Common retrieval and integration issues when using `emotional-memory`.

## Retrieval returns unexpected memories

1. **Inspect signals** — use `retrieve_with_explanations()` and read
   `breakdown.weights` / `breakdown.raw_signals` per result.
2. **Check runtime affect** — default retrieval uses `get_state().core_affect`
   for mood congruence and affect proximity. Call `set_affect()` or `observe()`
   if the session mood should differ.
3. **Query-type routing** — with `QueryClassifierConfig` enabled, weights depend on
   the classified query type. Verify routing with a heuristic classifier first.
4. **Regime** — AFT wins on affect-discriminative recall, not on factual QA or
   content-determined gold. See [limitations](research/08_limitations.md).

## Query-appraisal path underperforms cosine

- Use `retrieve_query_gated()` (Addendum Y) to route neutral queries to
  semantic-only retrieval.
- Pair query appraisal with `DIRECT_VAD_SCHEMA` for stronger human-gold agreement.
- Fall back to `KeywordAppraisalEngine` when no LLM key is available.

## LLM appraisal silent degradation

`LLMAppraisalEngine` defaults to `fallback_on_error=True` (neutral vector on failure).
Monitor logs at `DEBUG` for `appraisal_llm` warnings. Set `fallback_on_error=False`
in tests to surface errors.

## Async vs sync behaviour differs

Ensure both engines use the same `EmotionalMemoryConfig`. Async paths use an
`asyncio.Lock` on affective state; the sync engine is single-threaded.

## Persistence issues

| Backend | Symptom | Check |
|---------|---------|-------|
| Redis state store | State not restored | Default is fail-open; use `strict=True` in production |
| SQLite memories | Slow search | Install `sqlite-vec` (`[sqlite]` extra) for ANN |
| Chroma/Qdrant | Dimension mismatch | First embedding sets dimension; do not mix dims |
| InMemoryStore | Slow at large N | Full-scan cosine; move to SQLite/Qdrant/Chroma — see [Performance & Scaling](guides/performance_scaling.md) |

## Further reading

- [Performance & Scaling](guides/performance_scaling.md)
- [Configuration guide](tutorials/configuration_guide.md)
- [Benchmarks & addenda index](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/README.md)
- [Claim validation matrix](research/claim_validation_matrix.md)
