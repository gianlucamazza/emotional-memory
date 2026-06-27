window.BENCHMARK_DATA = {
  "lastUpdate": 1782546533796,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d729d49a70752079ba0c67429375f6b30d308ce",
          "message": "fix(appraisal): guard elaborate() type-safety + ship DIRECT_VAD_SCHEMA opt-in (#72)\n\nTwo coupled corrections from the Addendum V \"adopt direct-VAD\" decision.\n\nFix — elaborate() type-safety (sync + async). The dual-path slow path stored the\nappraisal on EmotionalTag.appraisal (typed AppraisalVector | None) without a type\ncheck; a custom AppraisalSchema returns a GenericAppraisalVector, so this was a\nsilent type violation. Now persists the appraisal only when isinstance AppraisalVector\n(else None), mirroring encode(); blended core_affect unchanged. Regression test added\n(dual_path + custom schema → no crash, tag.appraisal None, core_affect blended).\n\nAdd — DIRECT_VAD_SCHEMA (opt-in). The LLM rates valence/arousal/dominance directly\n(identity projection) instead of the 5 Scherer SECs. Better human-gold agreement than\nthe SEC→projection (Addendum V: valence r=0.79/bias≈0, arousal r=0.58, dominance r=0.43\nvs 0.70/0.23/0.31). Exported top-level + __all__; select via\nLLMAppraisalConfig(appraisal_schema=DIRECT_VAD_SCHEMA). SCHERER_CPM_SCHEMA stays the\ndefault (theory-faithful; required for dual-path SEC storage + SEC-reading features).\n\n- appraisal_schema.py: DIRECT_VAD_SCHEMA const + docstring.\n- __init__.py: export + __all__.\n- engine.py / async_engine.py: isinstance guard in _elaborate_with_memory.\n- tests: TestDirectVADSchema + TestElaborateWithCustomSchema.\n- docs: byo_appraisal_schema.md, CLAUDE.md, CHANGELOG (Added + Fixed).\n\nRefs #62\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T09:43:21+02:00",
          "tree_id": "7019d403d95e2c322ca26918ad5c536e4ff75618",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8d729d49a70752079ba0c67429375f6b30d308ce"
        },
        "date": 1782546532471,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 555.8197082346414,
            "unit": "iter/sec",
            "range": "stddev: 0.0008275489501878315",
            "extra": "mean: 1.7991445520637173 msec\nrounds: 1623"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.0000547704419,
            "unit": "iter/sec",
            "range": "stddev: 0.001066638865390058",
            "extra": "mean: 2.23214258425126 msec\nrounds: 1816"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 360.31082824407156,
            "unit": "iter/sec",
            "range": "stddev: 0.0015290252729838261",
            "extra": "mean: 2.7753814806881363 msec\nrounds: 2382"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 256.7336390979386,
            "unit": "iter/sec",
            "range": "stddev: 0.0023815244911546468",
            "extra": "mean: 3.8950875448718296 msec\nrounds: 3432"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 455.5446486649116,
            "unit": "iter/sec",
            "range": "stddev: 0.0009679764148215025",
            "extra": "mean: 2.1951745079889577 msec\nrounds: 1815"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.60045993758314,
            "unit": "iter/sec",
            "range": "stddev: 0.00024164469563914837",
            "extra": "mean: 2.6001008947370736 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137393.70659456222,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012078578523907381",
            "extra": "mean: 7.27835375277355 usec\nrounds: 50083"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.416851992694312,
            "unit": "iter/sec",
            "range": "stddev: 0.00023559399387553198",
            "extra": "mean: 35.19038633333101 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8262889410091807,
            "unit": "iter/sec",
            "range": "stddev: 0.00253087326239041",
            "extra": "mean: 1.2102304053333437 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.037399115753889496,
            "unit": "iter/sec",
            "range": "stddev: 1.5829869628598543",
            "extra": "mean: 26.738600093666662 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3177.500445752958,
            "unit": "iter/sec",
            "range": "stddev: 0.000013084511816587051",
            "extra": "mean: 314.71278039837836 usec\nrounds: 2459"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 867.4670063839096,
            "unit": "iter/sec",
            "range": "stddev: 0.000015811135589248872",
            "extra": "mean: 1.152781595888658 msec\nrounds: 730"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 359.7764561058368,
            "unit": "iter/sec",
            "range": "stddev: 0.00002291626838443538",
            "extra": "mean: 2.779503725240504 msec\nrounds: 313"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 503.84059151205327,
            "unit": "iter/sec",
            "range": "stddev: 0.0007643316380078635",
            "extra": "mean: 1.9847547356177577 msec\nrounds: 730"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10712.844952868458,
            "unit": "iter/sec",
            "range": "stddev: 0.000005411698777543962",
            "extra": "mean: 93.34588565404759 usec\nrounds: 7145"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10258.091745461285,
            "unit": "iter/sec",
            "range": "stddev: 0.000017444127897795264",
            "extra": "mean: 97.48401796488632 usec\nrounds: 8127"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1844.9467332621134,
            "unit": "iter/sec",
            "range": "stddev: 0.000010260636031379345",
            "extra": "mean: 542.0210686689397 usec\nrounds: 1165"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1828.6512387411267,
            "unit": "iter/sec",
            "range": "stddev: 0.000011076795181001983",
            "extra": "mean: 546.8511320334742 usec\nrounds: 1742"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1009.206398454326,
            "unit": "iter/sec",
            "range": "stddev: 0.000022050159499877238",
            "extra": "mean: 990.8775861226939 usec\nrounds: 807"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 377.2458760625631,
            "unit": "iter/sec",
            "range": "stddev: 0.00005529527872311371",
            "extra": "mean: 2.6507910714288583 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.488116066639282,
            "unit": "iter/sec",
            "range": "stddev: 0.009149996807797581",
            "extra": "mean: 48.80878245454183 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 456.7859133832736,
            "unit": "iter/sec",
            "range": "stddev: 0.00012094170594972018",
            "extra": "mean: 2.1892093663600654 msec\nrounds: 434"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 365.45546787640234,
            "unit": "iter/sec",
            "range": "stddev: 0.00008018937677085453",
            "extra": "mean: 2.7363115014007717 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 298.8009684438798,
            "unit": "iter/sec",
            "range": "stddev: 0.00006345179707340078",
            "extra": "mean: 3.3467093671345243 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 188.1550487873207,
            "unit": "iter/sec",
            "range": "stddev: 0.00012829524395414617",
            "extra": "mean: 5.314765702249853 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 842.215519706869,
            "unit": "iter/sec",
            "range": "stddev: 0.00002433290096821542",
            "extra": "mean: 1.187344541392502 msec\nrounds: 761"
          }
        ]
      }
    ]
  }
}