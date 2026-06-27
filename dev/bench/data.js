window.BENCHMARK_DATA = {
  "lastUpdate": 1782553998786,
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
          "id": "100ceb3059948c6e8baa3948c950837e62db60b1",
          "message": "feat(engine): add query_affect retrieval API (Addendum T production path) (#76)\n\nretrieve() and retrieve_with_explanations() (sync + async) now accept an\noptional keyword-only query_affect: CoreAffect that overrides the\naffect-proximity signal (s3) without mutating runtime state. New convenience\nmethod retrieve_with_query_appraisal() appraises the query text with the\nconfigured appraisal engine and retrieves with that affect — the\nproduction-reachable retrieve-time query-appraisal mechanism validated in\nAddendum T, with no oracle and no state mutation.\n\nThe Addendum T benchmark's save/set/restore workaround is intentionally left\nin place: set_affect() also shifts mood/momentum (s2/s4), so the shared\nrealistic adapter is not behaviourally equivalent to the s3-only API; changing\nit would alter the published A1/A3/U/T measurements. The new API is the cleaner\nproduction semantics and is used going forward (T2A).\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T11:47:34+02:00",
          "tree_id": "41235fdfede3897f70122c0e5d96e3ede4423248",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/100ceb3059948c6e8baa3948c950837e62db60b1"
        },
        "date": 1782553997134,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 524.0548688447101,
            "unit": "iter/sec",
            "range": "stddev: 0.0009231450625906809",
            "extra": "mean: 1.9081971363123116 msec\nrounds: 1768"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 408.9730169333733,
            "unit": "iter/sec",
            "range": "stddev: 0.0011999018817954682",
            "extra": "mean: 2.445149089537397 msec\nrounds: 2055"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 343.2506672123811,
            "unit": "iter/sec",
            "range": "stddev: 0.0016054056745226945",
            "extra": "mean: 2.913322814843257 msec\nrounds: 2587"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 197.5448772646444,
            "unit": "iter/sec",
            "range": "stddev: 0.0031180667005877667",
            "extra": "mean: 5.06214088589264 msec\nrounds: 4487"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 433.0752195780698,
            "unit": "iter/sec",
            "range": "stddev: 0.0010480120144541996",
            "extra": "mean: 2.30906769723344 msec\nrounds: 1952"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 389.97395909877457,
            "unit": "iter/sec",
            "range": "stddev: 0.00024099246819760985",
            "extra": "mean: 2.564273784616257 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 147811.90478937543,
            "unit": "iter/sec",
            "range": "stddev: 5.484486720522382e-7",
            "extra": "mean: 6.765354938257172 usec\nrounds: 50899"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.778724239156425,
            "unit": "iter/sec",
            "range": "stddev: 0.0006905127203553042",
            "extra": "mean: 30.507593666669663 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8421642633541067,
            "unit": "iter/sec",
            "range": "stddev: 0.008940172011145272",
            "extra": "mean: 1.1874168063333361 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.036261536999485006,
            "unit": "iter/sec",
            "range": "stddev: 0.8878449566348616",
            "extra": "mean: 27.577430046999996 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3616.5449750078487,
            "unit": "iter/sec",
            "range": "stddev: 0.000006582595627006638",
            "extra": "mean: 276.50699961164725 usec\nrounds: 2572"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 929.6761484757387,
            "unit": "iter/sec",
            "range": "stddev: 0.000020433227916687163",
            "extra": "mean: 1.0756433857527286 msec\nrounds: 744"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 377.705936463326,
            "unit": "iter/sec",
            "range": "stddev: 0.00006767410958354103",
            "extra": "mean: 2.6475623056485817 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 509.455822523114,
            "unit": "iter/sec",
            "range": "stddev: 0.00045901361900981344",
            "extra": "mean: 1.9628787341116902 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13109.560126938324,
            "unit": "iter/sec",
            "range": "stddev: 0.000009859709329695126",
            "extra": "mean: 76.28021003886613 usec\nrounds: 8746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13193.482728886765,
            "unit": "iter/sec",
            "range": "stddev: 0.000001987073080461167",
            "extra": "mean: 75.79499822367052 usec\nrounds: 9571"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2223.691956118475,
            "unit": "iter/sec",
            "range": "stddev: 0.00000959382507794735",
            "extra": "mean: 449.702575596636 usec\nrounds: 377"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2218.8767526261436,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878118475319508",
            "extra": "mean: 450.67847901712145 usec\nrounds: 1668"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1138.5659429906361,
            "unit": "iter/sec",
            "range": "stddev: 0.00033687812133934405",
            "extra": "mean: 878.2978325992526 usec\nrounds: 908"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 394.69493944902297,
            "unit": "iter/sec",
            "range": "stddev: 0.000028770077477152028",
            "extra": "mean: 2.533602283819389 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.1911700429407,
            "unit": "iter/sec",
            "range": "stddev: 0.0007425786416433197",
            "extra": "mean: 45.062968652169516 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 478.8993489844768,
            "unit": "iter/sec",
            "range": "stddev: 0.000020784742847366972",
            "extra": "mean: 2.0881214437241056 msec\nrounds: 462"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 394.6821875833589,
            "unit": "iter/sec",
            "range": "stddev: 0.000024871016086464513",
            "extra": "mean: 2.533684142481842 msec\nrounds: 379"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 324.0111273978937,
            "unit": "iter/sec",
            "range": "stddev: 0.000024062295264748884",
            "extra": "mean: 3.086313757280241 msec\nrounds: 309"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 212.05645416475258,
            "unit": "iter/sec",
            "range": "stddev: 0.00003433761796126254",
            "extra": "mean: 4.7157253663360414 msec\nrounds: 202"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 948.8427960155085,
            "unit": "iter/sec",
            "range": "stddev: 0.000011017119595219578",
            "extra": "mean: 1.0539153632185614 msec\nrounds: 870"
          }
        ]
      }
    ]
  }
}