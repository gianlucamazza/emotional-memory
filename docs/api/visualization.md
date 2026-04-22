# Visualization

!!! note
    Requires the `viz` extra: `uv pip install "emotional-memory[viz]"`

Example: visualise the ranking-time breakdown returned by
`retrieve_with_explanations()`:

```python
explained = em.retrieve_with_explanations("project success", top_k=1)
top = explained[0]

plot_retrieval_radar(
    [
        top.breakdown.raw_signals.semantic_similarity,
        top.breakdown.raw_signals.mood_congruence,
        top.breakdown.raw_signals.affect_proximity,
        top.breakdown.raw_signals.momentum_alignment,
        top.breakdown.raw_signals.recency,
        top.breakdown.raw_signals.resonance,
    ]
)
```

Use `raw_signals` to inspect the ranking-time signal values and
`weighted_signals` to inspect each signal's contribution to the final score.

::: emotional_memory.visualization.plot_circumplex

::: emotional_memory.visualization.plot_decay_curves

::: emotional_memory.visualization.plot_yerkes_dodson

::: emotional_memory.visualization.plot_retrieval_radar

::: emotional_memory.visualization.plot_mood_evolution

::: emotional_memory.visualization.plot_adaptive_weights_heatmap

::: emotional_memory.visualization.plot_resonance_network

::: emotional_memory.visualization.plot_appraisal_radar
