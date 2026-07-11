window.BENCHMARK_DATA = {
  "lastUpdate": 1783764303845,
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
          "id": "99583865dfee86d0d0edf058de351918c48718c2",
          "message": "docs(roadmap): archive P2 algorithmic levers (evaluated, mostly declined) (#118)\n\nRecord the code-grounded verdict for the three residual runtime levers so a\nfuture pass does not silently re-open them:\n\n- ACT-R per-trace spacing decay -> DECLINED: a true multi-trace base-level sum\n  is not log-log linear and would break the declared fidelity invariant\n  test_log_log_linearity (R^2>0.99); improves only a minor signal (s5=0.10),\n  no bearing on the X/X2 boundary.\n- Hebbian forgetting / LTD -> DEFERRED to a pre-registered opt-in (default OFF)\n  research addendum; no harness validates a decay rate (tuning = circularity),\n  and it breaks three pinned tests.\n- Incremental adjacency cache -> DECLINED: the G2 prefilter already caps the\n  candidate pool at top_k*candidate_multiplier (=15), so the O(N*links) rebuild\n  is not a production bottleneck; a cache speeds up only the microbench.\n\nDocs-only; no src/ change, no benchmark run.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-11T11:58:11+02:00",
          "tree_id": "0c3cad8cc8da89c1385b843fe3407329b3aa0cbe",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/99583865dfee86d0d0edf058de351918c48718c2"
        },
        "date": 1783764302656,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 579.155538425359,
            "unit": "iter/sec",
            "range": "stddev: 0.0007965935515958165",
            "extra": "mean: 1.7266518813216511 msec\nrounds: 1483"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 442.89040137086187,
            "unit": "iter/sec",
            "range": "stddev: 0.0011148009091378466",
            "extra": "mean: 2.2578949485126296 msec\nrounds: 1748"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 363.85076783478013,
            "unit": "iter/sec",
            "range": "stddev: 0.0015911894742042885",
            "extra": "mean: 2.7483795237010105 msec\nrounds: 2194"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 189.77679129242009,
            "unit": "iter/sec",
            "range": "stddev: 0.0036752824482887696",
            "extra": "mean: 5.2693482337317885 msec\nrounds: 3765"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 471.659761111016,
            "unit": "iter/sec",
            "range": "stddev: 0.0009571820326250687",
            "extra": "mean: 2.120172383678554 msec\nrounds: 1642"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 379.12515398518855,
            "unit": "iter/sec",
            "range": "stddev: 0.0002917195566563523",
            "extra": "mean: 2.6376514179775774 msec\nrounds: 445"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135376.84654582068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011458789651357814",
            "extra": "mean: 7.386787515851408 usec\nrounds: 32313"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.714066730922685,
            "unit": "iter/sec",
            "range": "stddev: 0.0005136672206914737",
            "extra": "mean: 36.08275933333971 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8146509330756819,
            "unit": "iter/sec",
            "range": "stddev: 0.004822955666416463",
            "extra": "mean: 1.2275196153333308 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.027074477068675173,
            "unit": "iter/sec",
            "range": "stddev: 1.4033018945150741",
            "extra": "mean: 36.935154738666675 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3171.668686350979,
            "unit": "iter/sec",
            "range": "stddev: 0.00001486266270623042",
            "extra": "mean: 315.29144399710464 usec\nrounds: 2232"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 858.4889654226993,
            "unit": "iter/sec",
            "range": "stddev: 0.00003698709335129542",
            "extra": "mean: 1.1648373366192588 msec\nrounds: 710"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.1582397785681,
            "unit": "iter/sec",
            "range": "stddev: 0.00004433188328904222",
            "extra": "mean: 2.815646345762593 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 489.7290688087786,
            "unit": "iter/sec",
            "range": "stddev: 0.0009490192270965567",
            "extra": "mean: 2.0419453605897013 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10236.424717534128,
            "unit": "iter/sec",
            "range": "stddev: 0.000016981986989932084",
            "extra": "mean: 97.69035845953957 usec\nrounds: 6726"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10131.941628134142,
            "unit": "iter/sec",
            "range": "stddev: 0.000028143009731768232",
            "extra": "mean: 98.69776561120557 usec\nrounds: 7927"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1814.921501540293,
            "unit": "iter/sec",
            "range": "stddev: 0.000013948260733565138",
            "extra": "mean: 550.9880174714538 usec\nrounds: 973"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1758.2950010965856,
            "unit": "iter/sec",
            "range": "stddev: 0.00009645534376925742",
            "extra": "mean: 568.7327777058662 usec\nrounds: 1552"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 988.0582406969856,
            "unit": "iter/sec",
            "range": "stddev: 0.00003563412750906391",
            "extra": "mean: 1.0120860884623466 msec\nrounds: 780"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.0767929268789,
            "unit": "iter/sec",
            "range": "stddev: 0.00006706523776889575",
            "extra": "mean: 2.709463231404308 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.95039999953585,
            "unit": "iter/sec",
            "range": "stddev: 0.0013917704287349927",
            "extra": "mean: 50.12430828571182 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 460.4470794422151,
            "unit": "iter/sec",
            "range": "stddev: 0.00004572512652956587",
            "extra": "mean: 2.17180224318373 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 349.96068609787955,
            "unit": "iter/sec",
            "range": "stddev: 0.00020646951567152556",
            "extra": "mean: 2.857463823008716 msec\nrounds: 339"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 291.085018070318,
            "unit": "iter/sec",
            "range": "stddev: 0.00016538570785628998",
            "extra": "mean: 3.435422429602433 msec\nrounds: 277"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 177.00967340165036,
            "unit": "iter/sec",
            "range": "stddev: 0.0013619539006692243",
            "extra": "mean: 5.649408762711589 msec\nrounds: 177"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 834.9520823609606,
            "unit": "iter/sec",
            "range": "stddev: 0.00004439598203755301",
            "extra": "mean: 1.197673520583768 msec\nrounds: 753"
          }
        ]
      }
    ]
  }
}