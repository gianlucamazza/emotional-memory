window.BENCHMARK_DATA = {
  "lastUpdate": 1791622656360,
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
          "id": "879a8c9d988ba36c14c81cd1a0b08bb96856da7b",
          "message": "ci: push bench-data with x-access-token auth (#150)\n\nGitHub's git HTTP endpoint rejects Authorization: bearer, so the\nfirst-run seed of bench-data failed with \"could not read Username\".\nPush via https://x-access-token:${GH_TOKEN}@github.com/... instead,\nno-op when bench-data already has dev/bench/data.js, and never\nforce-push.\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>",
          "timestamp": "2026-10-10T10:50:18+02:00",
          "tree_id": "f249d5c8805a3e93c8ee3ac8e55cba21d31ed8f2",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/879a8c9d988ba36c14c81cd1a0b08bb96856da7b"
        },
        "date": 1791622655220,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 430.29689964555496,
            "unit": "iter/sec",
            "range": "stddev: 0.0011101726136667571",
            "extra": "mean: 2.3239767723721045 msec\nrounds: 1665"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 491.08332376820755,
            "unit": "iter/sec",
            "range": "stddev: 0.000893976595251966",
            "extra": "mean: 2.036314310831704 msec\nrounds: 1274"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 453.62800859952955,
            "unit": "iter/sec",
            "range": "stddev: 0.0009291370980469537",
            "extra": "mean: 2.2044494190014112 msec\nrounds: 1463"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 328.569196715839,
            "unit": "iter/sec",
            "range": "stddev: 0.0019302801484269675",
            "extra": "mean: 3.0434989341525025 msec\nrounds: 2035"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 454.11536203612735,
            "unit": "iter/sec",
            "range": "stddev: 0.0008218462758799558",
            "extra": "mean: 2.202083619273035 msec\nrounds: 1266"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 311.631697465705,
            "unit": "iter/sec",
            "range": "stddev: 0.00039106001997870086",
            "extra": "mean: 3.2089161921984846 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 150527.9599180015,
            "unit": "iter/sec",
            "range": "stddev: 7.11461600006412e-7",
            "extra": "mean: 6.643284081872494 usec\nrounds: 36141"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 11.644029534724625,
            "unit": "iter/sec",
            "range": "stddev: 0.000322277581593408",
            "extra": "mean: 85.88092266666081 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5391475437688349,
            "unit": "iter/sec",
            "range": "stddev: 0.006613567547565692",
            "extra": "mean: 1.8547798493333403 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02603102820369879,
            "unit": "iter/sec",
            "range": "stddev: 1.296179509517677",
            "extra": "mean: 38.41569346299999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3564.9399984904785,
            "unit": "iter/sec",
            "range": "stddev: 0.000012395966462564906",
            "extra": "mean: 280.509630014372 usec\nrounds: 2319"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 893.2627186292109,
            "unit": "iter/sec",
            "range": "stddev: 0.0000661437855047919",
            "extra": "mean: 1.1194914767455948 msec\nrounds: 602"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 353.50729964994053,
            "unit": "iter/sec",
            "range": "stddev: 0.00006961819828633838",
            "extra": "mean: 2.828795900368244 msec\nrounds: 271"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 429.710957490301,
            "unit": "iter/sec",
            "range": "stddev: 0.0004693030967123694",
            "extra": "mean: 2.327145683788087 msec\nrounds: 623"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13180.33695149953,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021900327333034286",
            "extra": "mean: 75.87059448326393 usec\nrounds: 7033"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13106.236602023855,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023206777218781886",
            "extra": "mean: 76.29955343897736 usec\nrounds: 7448"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2095.519722340705,
            "unit": "iter/sec",
            "range": "stddev: 0.00001658330357216305",
            "extra": "mean: 477.20858426614836 usec\nrounds: 979"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2079.1927850518514,
            "unit": "iter/sec",
            "range": "stddev: 0.000028103839251332314",
            "extra": "mean: 480.955882104536 usec\nrounds: 1408"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1151.5260431710024,
            "unit": "iter/sec",
            "range": "stddev: 0.00003361063352002006",
            "extra": "mean: 868.4128387111948 usec\nrounds: 1023"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 373.6404780946441,
            "unit": "iter/sec",
            "range": "stddev: 0.000046998599962519735",
            "extra": "mean: 2.676369554764078 msec\nrounds: 1123"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.933410081182192,
            "unit": "iter/sec",
            "range": "stddev: 0.001651837559098745",
            "extra": "mean: 47.77052549593611 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 457.64826798061534,
            "unit": "iter/sec",
            "range": "stddev: 0.00006106483966005433",
            "extra": "mean: 2.185084201044889 msec\nrounds: 2487"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.3488126217097,
            "unit": "iter/sec",
            "range": "stddev: 0.00010415614561180875",
            "extra": "mean: 2.685653790484775 msec\nrounds: 988"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.2983872722522,
            "unit": "iter/sec",
            "range": "stddev: 0.00018144773392590337",
            "extra": "mean: 3.2647902880113877 msec\nrounds: 684"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.45994217413423,
            "unit": "iter/sec",
            "range": "stddev: 0.00010083878402402582",
            "extra": "mean: 4.797084703998735 msec\nrounds: 250"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 937.312442227171,
            "unit": "iter/sec",
            "range": "stddev: 0.00001578197890403615",
            "extra": "mean: 1.0668801084341475 msec\nrounds: 1245"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5445.845989284399,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042046480879972364",
            "extra": "mean: 183.62619911904687 usec\nrounds: 4992"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 76.8251876724952,
            "unit": "iter/sec",
            "range": "stddev: 0.0009159315319678934",
            "extra": "mean: 13.016564362497718 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2186.893560792643,
            "unit": "iter/sec",
            "range": "stddev: 0.000011333101551419837",
            "extra": "mean: 457.26962570485074 usec\nrounds: 1774"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.99794503163435,
            "unit": "iter/sec",
            "range": "stddev: 0.005057766391441399",
            "extra": "mean: 34.485202275853794 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 21284.545831108775,
            "unit": "iter/sec",
            "range": "stddev: 0.000003314476960866888",
            "extra": "mean: 46.98244481864554 usec\nrounds: 12767"
          }
        ]
      }
    ]
  }
}