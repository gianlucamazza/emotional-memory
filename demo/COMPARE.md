# Compare contract

Compare takes the selected retrieval mode's current memory store and affective
state, copies them into two scratch engines, and applies the mood-state and
query-gated strategies to those copies. Retrieval can update scratch memories
(retrieval counts, reconsolidation, predictions and links); those writes are discarded.
Neither live engine's memory or affective state is written back.

The selected mode matters after the live engines diverge. Compare does not compare
the two live histories, and changing the selected mode changes the source snapshot.
The result is a point-in-time diagnostic, not evidence of longitudinal superiority.

Embedder and appraisal services are shared: calls may populate caches or invoke an
external model. The memory isolation guarantee does not promise zero service side
effects, identical repeated outputs, or deterministic LLM responses.

Longitudinal evaluation must replay the same ordered external events into independent
arms, preserve each arm's own retrieval effects, and report state divergence. See
[the realistic benchmark protocol](../benchmarks/realistic/protocol.md).
