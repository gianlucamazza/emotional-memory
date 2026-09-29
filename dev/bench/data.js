window.BENCHMARK_DATA = {
  "lastUpdate": 1790646134917,
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
          "id": "ac0f640e01220167f918556364fd2c75c156b113",
          "message": "ci: Dependabot auto-merge for patch/minor (+ Actions) (#140)",
          "timestamp": "2026-09-29T03:34:37+02:00",
          "tree_id": "d177e39587e34c77eff424213994b7eaa3e4466a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/ac0f640e01220167f918556364fd2c75c156b113"
        },
        "date": 1790646133230,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 481.16276481597254,
            "unit": "iter/sec",
            "range": "stddev: 0.0009309953137807931",
            "extra": "mean: 2.0782988068132497 msec\nrounds: 2143"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 502.9643659752836,
            "unit": "iter/sec",
            "range": "stddev: 0.0007768356108897847",
            "extra": "mean: 1.9882124214921846 msec\nrounds: 1796"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 483.0935503345422,
            "unit": "iter/sec",
            "range": "stddev: 0.0008673220759717451",
            "extra": "mean: 2.0699924461990857 msec\nrounds: 2026"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 377.4235367375861,
            "unit": "iter/sec",
            "range": "stddev: 0.0013142266448545321",
            "extra": "mean: 2.6495432919841373 msec\nrounds: 2757"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 537.5888079516055,
            "unit": "iter/sec",
            "range": "stddev: 0.0009781383031959572",
            "extra": "mean: 1.8601577733925243 msec\nrounds: 1571"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 415.67933925882403,
            "unit": "iter/sec",
            "range": "stddev: 0.00025950589413551334",
            "extra": "mean: 2.4057005137254297 msec\nrounds: 510"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 204731.24783173198,
            "unit": "iter/sec",
            "range": "stddev: 6.343812320012834e-7",
            "extra": "mean: 4.884452229890656 usec\nrounds: 50136"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 16.199116079029913,
            "unit": "iter/sec",
            "range": "stddev: 0.00033518094340306576",
            "extra": "mean: 61.731763333341405 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7007532058446568,
            "unit": "iter/sec",
            "range": "stddev: 0.02368878519120232",
            "extra": "mean: 1.427035926000002 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.04475134825931581,
            "unit": "iter/sec",
            "range": "stddev: 0.2237701735125074",
            "extra": "mean: 22.345695468333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4892.639690879377,
            "unit": "iter/sec",
            "range": "stddev: 0.000018789777366931316",
            "extra": "mean: 204.38864563522876 usec\nrounds: 3471"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1279.359047382061,
            "unit": "iter/sec",
            "range": "stddev: 0.00001588764614274883",
            "extra": "mean: 781.6414024243543 usec\nrounds: 907"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 490.9633134546252,
            "unit": "iter/sec",
            "range": "stddev: 0.00008522963499247633",
            "extra": "mean: 2.036812064354825 msec\nrounds: 404"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 509.9335838246541,
            "unit": "iter/sec",
            "range": "stddev: 0.0008085607057249721",
            "extra": "mean: 1.9610396956005554 msec\nrounds: 841"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 16016.51998744734,
            "unit": "iter/sec",
            "range": "stddev: 0.000002430625701079299",
            "extra": "mean: 62.43553535872536 usec\nrounds: 9545"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 16409.106704932186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024036066330453247",
            "extra": "mean: 60.94176959062762 usec\nrounds: 6419"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2863.1111950163645,
            "unit": "iter/sec",
            "range": "stddev: 0.000011376994230658337",
            "extra": "mean: 349.27040267965714 usec\nrounds: 1269"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2782.8858773242728,
            "unit": "iter/sec",
            "range": "stddev: 0.00001186404010293161",
            "extra": "mean: 359.3392054443475 usec\nrounds: 1947"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1690.5727979698338,
            "unit": "iter/sec",
            "range": "stddev: 0.00002738627168273028",
            "extra": "mean: 591.5154917912288 usec\nrounds: 1340"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 499.59091496771003,
            "unit": "iter/sec",
            "range": "stddev: 0.00009664006756983103",
            "extra": "mean: 2.0016376800299356 msec\nrounds: 1372"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 39.86775297110276,
            "unit": "iter/sec",
            "range": "stddev: 0.0003473771799722841",
            "extra": "mean: 25.082928569483897 msec\nrounds: 367"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 594.2955200425135,
            "unit": "iter/sec",
            "range": "stddev: 0.00005589167847985419",
            "extra": "mean: 1.682664543606965 msec\nrounds: 3050"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 499.2106794276784,
            "unit": "iter/sec",
            "range": "stddev: 0.000027328351332586992",
            "extra": "mean: 2.0031622743857427 msec\nrounds: 1425"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 413.4172806366387,
            "unit": "iter/sec",
            "range": "stddev: 0.00005122968399048961",
            "extra": "mean: 2.4188635715954057 msec\nrounds: 859"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 271.8135266379169,
            "unit": "iter/sec",
            "range": "stddev: 0.000055033587031503356",
            "extra": "mean: 3.6789927726154006 msec\nrounds: 409"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1326.6461770170315,
            "unit": "iter/sec",
            "range": "stddev: 0.000019544499150672267",
            "extra": "mean: 753.7804859533109 usec\nrounds: 1673"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 7752.813708176824,
            "unit": "iter/sec",
            "range": "stddev: 0.000005920592069850999",
            "extra": "mean: 128.98542872832206 usec\nrounds: 5458"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 115.05149445000067,
            "unit": "iter/sec",
            "range": "stddev: 0.000043147769198730676",
            "extra": "mean: 8.691760196427367 msec\nrounds: 112"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 3171.634903299847,
            "unit": "iter/sec",
            "range": "stddev: 0.000008470901693103298",
            "extra": "mean: 315.29480236189085 usec\nrounds: 2287"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 42.302639391649166,
            "unit": "iter/sec",
            "range": "stddev: 0.006103906794990492",
            "extra": "mean: 23.639186924998512 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 35070.859103782626,
            "unit": "iter/sec",
            "range": "stddev: 0.000002104591405602566",
            "extra": "mean: 28.51370127662893 usec\nrounds: 22566"
          }
        ]
      }
    ]
  }
}