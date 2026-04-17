# Comparative Benchmark

Measures **recall@k** (mood-congruent retrieval accuracy) and **latency** for
emotional-memory (AFT) against alternative memory systems on the
`affect_reference_v1` dataset (258 affect-labeled examples).

## Quick start

```bash
# AFT vs naive cosine baseline (no external deps)
python -m benchmarks.comparative.runner

# Add mem0 if installed
python -m benchmarks.comparative.runner --systems aft,naive_cosine,mem0

# Custom top-k and output path
python -m benchmarks.comparative.runner --top-k 10 --out my_results.csv
```

## Metric — recall@k (mood-congruent)

For each of 4 Russell quadrant queries, the benchmark retrieves top-k items and
counts how many are **mood-congruent** (same valence sign × arousal level as the
query).  `recall@k = congruent / min(k, retrieved)`.  Final score is the mean
across all 4 queries.

A system that retrieves purely by semantic cosine similarity will score ~0.25
on a balanced dataset (random quadrant hit rate). AFT's multi-signal scorer
(mood congruence + affect proximity + momentum) should score significantly higher.

## Systems

| System | Status | Notes |
|---|---|---|
| `aft` | ✅ always available | emotional-memory with HashEmbedder |
| `naive_cosine` | ✅ always available | pure cosine, same embedder as AFT (intra-paper baseline) |
| `mem0` | ⚠️ optional | `pip install mem0ai` + OpenAI key required |

### Adding a new system

1. Create `adapters/my_system.py` implementing `MemoryAdapter` (see `adapters/base.py`)
2. Register the class in `runner.py::_make_adapters()`
3. Add a row to the table above

**Time-box rule**: if a baseline cannot be installed and running within 4 hours,
document `status="not_evaluated"` with `reason` in the CSV and continue.

## Output files

| File | Description |
|---|---|
| `results.csv` | Machine-readable results |
| `results.md` | Human-readable Markdown table |

Both are git-ignored (generated artefacts). Run `make bench-comparative` to regenerate.
