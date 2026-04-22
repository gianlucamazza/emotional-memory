# Retrieval

Use `retrieve()` for normal recall and `retrieve_with_explanations()` when you need
ranking-time introspection.

The stable explainability surface is:

- `EmotionalMemory.retrieve_with_explanations()`
- `AsyncEmotionalMemory.retrieve_with_explanations()`
- `RetrievalSignals`
- `RetrievalBreakdown`
- `RetrievalExplanation`

These models describe the ranking plan before retrieval-side mutation. The returned
`Memory` objects may already reflect post-retrieval updates such as reconsolidation.

Lower-level helpers in `emotional_memory.retrieval` are documented here because they
shape the scoring model, but they are not all promoted as stable top-level API. In
particular, `build_retrieval_plan()` remains an engine helper rather than a supported
root import.

::: emotional_memory.retrieval.RetrievalConfig

::: emotional_memory.retrieval.AdaptiveWeightsConfig

::: emotional_memory.retrieval.RetrievalSignals

::: emotional_memory.retrieval.RetrievalBreakdown

::: emotional_memory.retrieval.RetrievalExplanation

::: emotional_memory.retrieval.retrieval_score

::: emotional_memory.retrieval.retrieval_breakdown

::: emotional_memory.retrieval.adaptive_weights

::: emotional_memory.retrieval.reconsolidate

::: emotional_memory.retrieval.compute_ape

::: emotional_memory.retrieval.update_prediction
