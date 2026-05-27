window.BENCHMARK_DATA = {
  "lastUpdate": 1779902989074,
  "repoUrl": "https://github.com/gianlucamazza/emotional-memory",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "info@gianlucamazza.it",
            "name": "Gianluca Mazza",
            "username": "gianlucamazza"
          },
          "committer": {
            "email": "info@gianlucamazza.it",
            "name": "Gianluca Mazza",
            "username": "gianlucamazza"
          },
          "distinct": true,
          "id": "2bc4d0750e274445f9c1a1e2261057c07c5d316d",
          "message": "feat(models): add frozen=True to all non-mutable Pydantic models (WS4)\n\nApply model_config = ConfigDict(frozen=True) to the 13 Pydantic value-objects\nthat were missing it, per CLAUDE.md convention ('All value objects are Pydantic\nfrozen=True').  Frozen models catch accidental mutation at the Python level\nand improve reasoning about immutability across the pipeline.\n\nModels updated:\n  decay.py         DecayConfig\n  resonance.py     ResonanceConfig\n  mood.py          MoodDecayConfig, MoodField (dict → ConfigDict)\n  engine.py        EmotionalMemoryConfig\n  appraisal_llm.py LLMAppraisalConfig (merged arbitrary_types_allowed)\n  retrieval.py     AdaptiveWeightsConfig, RetrievalConfig, RetrievalSignals,\n                   RetrievalBreakdown, RetrievalExplanation, RankedMemory,\n                   RetrievalPlan\n\nSkipped (intentionally mutable):\n  state.py:AffectiveState  — has PrivateAttr _history mutated in update()\n  models.py:Memory         — embedding/metadata fields mutable by design\n\nNo callsite changes needed: codebase already uses model_copy(update=…)\neverywhere; no direct field assignments on these models were found.\n\nVerified: 860 tests pass, 127 fidelity benchmarks pass, mypy strict clean,\nruff clean.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-27T19:22:22+02:00",
          "tree_id": "d18a7c91f32dfc0cbd546894fd4da959e99eae81",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/2bc4d0750e274445f9c1a1e2261057c07c5d316d"
        },
        "date": 1779902988621,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 525.2652337512459,
            "unit": "iter/sec",
            "range": "stddev: 0.0010357587651140547",
            "extra": "mean: 1.9038000913526634 msec\nrounds: 1434"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 374.89628249291235,
            "unit": "iter/sec",
            "range": "stddev: 0.0014934031485974064",
            "extra": "mean: 2.6674044174308547 msec\nrounds: 1962"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 319.051512882135,
            "unit": "iter/sec",
            "range": "stddev: 0.0018923910145052828",
            "extra": "mean: 3.134290105590012 msec\nrounds: 2576"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 164.35256199928796,
            "unit": "iter/sec",
            "range": "stddev: 0.004493884371014512",
            "extra": "mean: 6.084480751838432 msec\nrounds: 4352"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 403.5416849610993,
            "unit": "iter/sec",
            "range": "stddev: 0.0012824275791715161",
            "extra": "mean: 2.478058741555778 msec\nrounds: 1954"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 373.07492576933674,
            "unit": "iter/sec",
            "range": "stddev: 0.0002853029404408854",
            "extra": "mean: 2.6804267210878603 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135921.44192825095,
            "unit": "iter/sec",
            "range": "stddev: 8.379726967729781e-7",
            "extra": "mean: 7.357190931861004 usec\nrounds: 30304"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.272390882808395,
            "unit": "iter/sec",
            "range": "stddev: 0.0005983452645706754",
            "extra": "mean: 30.054948666664433 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8495891560723833,
            "unit": "iter/sec",
            "range": "stddev: 0.001884334746799525",
            "extra": "mean: 1.1770395053333307 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.025460586233976,
            "unit": "iter/sec",
            "range": "stddev: 0.8482190734500535",
            "extra": "mean: 39.276393356 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3556.1526524282986,
            "unit": "iter/sec",
            "range": "stddev: 0.000008928659255679234",
            "extra": "mean: 281.20277663478697 usec\nrounds: 2431"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 920.1001376423146,
            "unit": "iter/sec",
            "range": "stddev: 0.00001637045402690763",
            "extra": "mean: 1.0868382245463222 msec\nrounds: 717"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 368.3401042255041,
            "unit": "iter/sec",
            "range": "stddev: 0.00003569203851050539",
            "extra": "mean: 2.7148822203399905 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 498.2773924381918,
            "unit": "iter/sec",
            "range": "stddev: 0.0004812195905828285",
            "extra": "mean: 2.006914251330485 msec\nrounds: 752"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10858.997928236256,
            "unit": "iter/sec",
            "range": "stddev: 0.000004170575381744611",
            "extra": "mean: 92.08952857424686 usec\nrounds: 5127"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10917.195534188157,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037952473689168505",
            "extra": "mean: 91.59861585957786 usec\nrounds: 5990"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1800.9808602821365,
            "unit": "iter/sec",
            "range": "stddev: 0.00005285384629318148",
            "extra": "mean: 555.2529857776184 usec\nrounds: 1125"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1856.3870525136865,
            "unit": "iter/sec",
            "range": "stddev: 0.000011579745402378082",
            "extra": "mean: 538.6807663013624 usec\nrounds: 1549"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1198.700216912097,
            "unit": "iter/sec",
            "range": "stddev: 0.000018814632660349093",
            "extra": "mean: 834.2369392207526 usec\nrounds: 872"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 390.1060656844945,
            "unit": "iter/sec",
            "range": "stddev: 0.00011070399141537613",
            "extra": "mean: 2.5634054119239673 msec\nrounds: 369"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 18.893443813741378,
            "unit": "iter/sec",
            "range": "stddev: 0.0009175926668074006",
            "extra": "mean: 52.92841314999919 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.1069693844648,
            "unit": "iter/sec",
            "range": "stddev: 0.00013811556917880057",
            "extra": "mean: 2.1271754411753383 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.28098426293195,
            "unit": "iter/sec",
            "range": "stddev: 0.0004058728608952602",
            "extra": "mean: 2.629634510750569 msec\nrounds: 372"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 321.8182677261603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000895321477229211",
            "extra": "mean: 3.107343803276308 msec\nrounds: 305"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 211.86487520041362,
            "unit": "iter/sec",
            "range": "stddev: 0.00013210005206929512",
            "extra": "mean: 4.719989564358177 msec\nrounds: 202"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 973.1640264986727,
            "unit": "iter/sec",
            "range": "stddev: 0.000021058042184543267",
            "extra": "mean: 1.0275760023702067 msec\nrounds: 844"
          }
        ]
      }
    ]
  }
}