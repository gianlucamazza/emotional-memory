# Benchmarks

Three benchmark suites cover the library: psychological **fidelity** (intra-theory
invariants), **performance** (throughput), and **appraisal quality** (LLM prompt
validation). For experimental results against baselines and external datasets see
[Current Evidence](research/09_current_evidence.md).

The full pre-registered confirmatory programme (addenda A–Z, closures, and verdicts)
is indexed in the repository [benchmarks/README.md](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/README.md).

## Psychological fidelity (127 parametrized test cases, 20 phenomena)

The library validates 20 phenomena from the affective science literature via 127 parametrized test cases (run `pytest --collect-only benchmarks/fidelity/` to enumerate them):

| Phenomenon | Reference | Cases | Test file |
|---|---|---|---|
| Mood-congruent recall | Bower 1981 | 3 | [test_mood_congruent.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_mood_congruent.py) |
| Emotional enhancement | Cahill & McGaugh 1995 | 3 | [test_emotional_enhancement.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_emotional_enhancement.py) |
| Yerkes-Dodson inverted-U | Yerkes & Dodson 1908 | 12 | [test_yerkes_dodson.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_yerkes_dodson.py) |
| Spacing effect | Ebbinghaus 1885 | 7 | [test_spacing_effect.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_spacing_effect.py) |
| Arousal floor | McGaugh 2004 | 7 | [test_arousal_floor.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_arousal_floor.py) |
| Reconsolidation (APE) | Nader & Schiller 2000 | 5 | [test_reconsolidation.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_reconsolidation.py) |
| State-dependent retrieval | Godden & Baddeley 1975 | 3 | [test_state_dependent.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_state_dependent.py) |
| Affective momentum | Spinoza, Ethics III | 9 | [test_momentum.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_momentum.py) |
| Mood-adaptive weights | Heidegger, Being & Time §29 | 14 | [test_mood_adaptive.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_mood_adaptive.py) |
| Appraisal-to-affect mapping | Scherer CPM 2009 | 11 | [test_appraisal_affect.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_appraisal_affect.py) |
| Spreading activation | Collins & Loftus 1975 | 5 | [test_spreading_activation.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_spreading_activation.py) |
| Hebbian co-retrieval strengthening | Hebb 1949 | 4 | [test_hebbian_strengthening.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_hebbian_strengthening.py) |
| ACT-R power-law decay | Anderson 1983 / McGaugh 2004 | 5 | [test_decay_power_law.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_decay_power_law.py) |
| PAD dominance | Mehrabian & Russell 1974 | 8 | [test_pad_dominance.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_pad_dominance.py) |
| Emotional retrieval vs. cosine | Bower 1981 / Russell 1980 / Nader 2000 | 3 | [test_emotional_vs_cosine.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_emotional_vs_cosine.py) |
| Design gap regression | (various) | 3 | [test_design_gaps.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_design_gaps.py) |
| Dual-path encoding | LeDoux 1996 | 6 | [test_dual_path_encoding.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_dual_path_encoding.py) |
| Emotion categorization | Plutchik 1980 | 10 | [test_emotion_categorization.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_emotion_categorization.py) |
| Affective prediction error | Schultz 1997 / Pearce-Hall 1980 | 5 | [test_prediction_error.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_prediction_error.py) |
| APE-gated reconsolidation window | Nader & Schiller 2000 | 3 | [test_reconsolidation_window.py](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/fidelity/test_reconsolidation_window.py) |

Run with: `make bench-fidelity`

For the comparative protocol and interpretation rules, see
[benchmarks/comparative/protocol.md](https://github.com/gianlucamazza/emotional-memory/blob/main/benchmarks/comparative/protocol.md).

## Performance (hash-based embedder, InMemoryStore)

Committed numbers from `make bench-perf` → `paper/tables/table2_perf.md`
(CPU-only, in-process). Regenerate with `make bench-perf && make reproduce-paper`.

| Operation | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|
| Encode (single) | 4.39 | 4.19 | 15.94 |
| Encode w/ resonance (pre-populated) | 3.13 | 3.09 | 10.32 |
| Encode scaling @ N=100 existing | 2.91 | 2.83 | 7.94 |
| Encode scaling @ N=1000 existing | 5.92 | 5.80 | 10.97 |
| Retrieve top-5 @ N=100 | 1.37 | 1.29 | 2.94 |
| Retrieve top-5 @ N=1 000 | 5.60 | 5.46 | 8.41 |
| Retrieve top-5 @ N=10 000 | 99.23 | 97.03 | 112.00 |
| Retrieve top-k @ N=1 000 (k=1…25) | 5.72–10.06 | — | 9.8–15.7 |
| Retrieve + reconsolidation @ N=200 | 1.87 | 1.78 | 4.24 |

`InMemoryStore.search_by_embedding` uses vectorized matrix multiplication
(numpy) over a **lazily cached** embedding matrix (rebuilt on save/update/delete),
so repeated queries avoid re-stacking all vectors. Retrieval scoring is O(n · d)
on the candidate pool; the engine **G2 pre-filter** caps the pool at
`top_k × candidate_multiplier` when the store is larger. Two-pass spreading
activation runs only when resonance links yield a non-empty activation map.

For store choice, appraisal LLM cost, and scaling knobs see
[Performance & Scaling](guides/performance_scaling.md).

Scorer-only overhead (6-signal plan vs cosine rank on a fixed candidate set)
is measured by `benchmarks/perf/bench_scoring.py` inside `make bench-perf`.

Production-shaped component shares (hash vs SBERT embed / plan / e2e) live in
[Performance & Scaling](guides/performance_scaling.md) and are regenerated with
`make bench-perf-profile` (not in default CI).

Run with: `make bench-perf`

## Appraisal quality (LLM prompt validation)

15 natural-language phrases with expected directional outcomes against Scherer's 5 dimensions:

| Phrase category | Key assertions |
|---|---|
| Personal loss ("I got fired") | `goal_relevance < -0.2`, `coping_potential < 0.6` |
| Achievement ("Got promoted") | `goal_relevance > 0.2`, `norm_congruence > 0.0` |
| Moral violation ("Coworker stole credit") | `norm_congruence < -0.2`, `goal_relevance < 0.0` |
| Grief, danger, betrayal, relief, … | dimension-specific directional bounds |

Assertions use wide bands (e.g. `> 0.3`, `< -0.2`) and evaluate the median over 3 LLM calls to tolerate non-determinism. Designed to catch systematic prompt regressions, not exact calibration.

Run with: `EMOTIONAL_MEMORY_LLM_API_KEY=... make bench-appraisal`

Works with any OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, …) via `EMOTIONAL_MEMORY_LLM_BASE_URL`.
See [LLM Environment Variables](contributing/llm-environment.md) for the full config surface.

## See also

- [Current Evidence](research/09_current_evidence.md) — experimental results vs baselines
- [Comparison](comparison.md) — positioning against other memory systems
