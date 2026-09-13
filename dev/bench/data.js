window.BENCHMARK_DATA = {
  "lastUpdate": 1789296458642,
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
          "id": "e58987008ca2b4bee0bb27d389de2ce8795519f5",
          "message": "feat: host-owned FlyAffectHost adapter for affective-fly (#136)\n\n* feat(integrations): add FlyAffectHost (memory owns time and mood_dt)\n\n* test: cover optional fly imports and document the host adapter\n\n* build: add fly-tests CI job and changelog patch workflow\n\n* docs: add fly host adapter to unreleased changelog\n\n* chore: add docs patch for FlyAffectHost README Makefile ROADMAP\n\n* docs: document FlyAffectHost in README, Makefile, and roadmap\n\n---------\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
          "timestamp": "2026-09-13T12:40:58+02:00",
          "tree_id": "43a414a687f2667937e86b6443761a887476af73",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e58987008ca2b4bee0bb27d389de2ce8795519f5"
        },
        "date": 1789296457813,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 474.70930623131574,
            "unit": "iter/sec",
            "range": "stddev: 0.0008069060457510542",
            "extra": "mean: 2.1065523402920636 msec\nrounds: 1437"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 481.47237211527994,
            "unit": "iter/sec",
            "range": "stddev: 0.0010889785945803626",
            "extra": "mean: 2.076962371914806 msec\nrounds: 1175"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 485.82766849007425,
            "unit": "iter/sec",
            "range": "stddev: 0.0007974819524500471",
            "extra": "mean: 2.058343039843624 msec\nrounds: 1280"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 385.28829410394496,
            "unit": "iter/sec",
            "range": "stddev: 0.0012349873364818514",
            "extra": "mean: 2.595459076496664 msec\nrounds: 1804"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 469.8005961089564,
            "unit": "iter/sec",
            "range": "stddev: 0.0010490400781224126",
            "extra": "mean: 2.1285626461148195 msec\nrounds: 1184"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 307.4076995025803,
            "unit": "iter/sec",
            "range": "stddev: 0.00047826481312777995",
            "extra": "mean: 3.2530089572190644 msec\nrounds: 374"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135746.50693854594,
            "unit": "iter/sec",
            "range": "stddev: 0.000001274775478346107",
            "extra": "mean: 7.366672060686703 usec\nrounds: 30713"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.642196916670235,
            "unit": "iter/sec",
            "range": "stddev: 0.0008348456033365638",
            "extra": "mean: 93.96556066666761 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5205252488072306,
            "unit": "iter/sec",
            "range": "stddev: 0.005404732519798876",
            "extra": "mean: 1.9211363949999978 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.026008522931688628,
            "unit": "iter/sec",
            "range": "stddev: 3.175827789841691",
            "extra": "mean: 38.44893470600001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3144.1645859860387,
            "unit": "iter/sec",
            "range": "stddev: 0.000012570936117221058",
            "extra": "mean: 318.04950811326273 usec\nrounds: 2157"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 843.6998530673925,
            "unit": "iter/sec",
            "range": "stddev: 0.000022954433081200907",
            "extra": "mean: 1.1852556289589904 msec\nrounds: 663"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 343.35878186213006,
            "unit": "iter/sec",
            "range": "stddev: 0.00005020608911505002",
            "extra": "mean: 2.912405486111997 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 437.16072916653775,
            "unit": "iter/sec",
            "range": "stddev: 0.00044384028206826575",
            "extra": "mean: 2.287488178333253 msec\nrounds: 600"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9819.090546380585,
            "unit": "iter/sec",
            "range": "stddev: 0.00001496709149158627",
            "extra": "mean: 101.84242575995086 usec\nrounds: 6250"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9938.875951928267,
            "unit": "iter/sec",
            "range": "stddev: 0.000005900687782175218",
            "extra": "mean: 100.61499960727322 usec\nrounds: 7639"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1773.3165008499502,
            "unit": "iter/sec",
            "range": "stddev: 0.000011819487321236416",
            "extra": "mean: 563.9151271195527 usec\nrounds: 472"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1756.3745395725034,
            "unit": "iter/sec",
            "range": "stddev: 0.000011984279618986732",
            "extra": "mean: 569.3546435963466 usec\nrounds: 1546"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 995.2484697502011,
            "unit": "iter/sec",
            "range": "stddev: 0.00005386479742673525",
            "extra": "mean: 1.004774215077157 msec\nrounds: 902"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 357.91239430212244,
            "unit": "iter/sec",
            "range": "stddev: 0.00008298886129451235",
            "extra": "mean: 2.793979800419753 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.154782515244662,
            "unit": "iter/sec",
            "range": "stddev: 0.0023644326279567375",
            "extra": "mean: 43.187622226277405 msec\nrounds: 274"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 448.8194817690132,
            "unit": "iter/sec",
            "range": "stddev: 0.00010487758744249449",
            "extra": "mean: 2.2280672756416893 msec\nrounds: 2496"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 359.392413183105,
            "unit": "iter/sec",
            "range": "stddev: 0.00016928773757884675",
            "extra": "mean: 2.7824738734551837 msec\nrounds: 972"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 285.51314301850664,
            "unit": "iter/sec",
            "range": "stddev: 0.00025226861431518454",
            "extra": "mean: 3.502465733898566 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 183.09319914844258,
            "unit": "iter/sec",
            "range": "stddev: 0.00033982899976590003",
            "extra": "mean: 5.461699312978038 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 839.674203803263,
            "unit": "iter/sec",
            "range": "stddev: 0.00003220705516818811",
            "extra": "mean: 1.1909380989323588 msec\nrounds: 1031"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5156.36813436509,
            "unit": "iter/sec",
            "range": "stddev: 0.000008419929752698596",
            "extra": "mean: 193.93495071374133 usec\nrounds: 4484"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 74.09419085364702,
            "unit": "iter/sec",
            "range": "stddev: 0.00045835662251398495",
            "extra": "mean: 13.496334712328917 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1825.6618914855503,
            "unit": "iter/sec",
            "range": "stddev: 0.000020594865360859728",
            "extra": "mean: 547.7465486154695 usec\nrounds: 1409"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 24.98801468096671,
            "unit": "iter/sec",
            "range": "stddev: 0.005595663660088528",
            "extra": "mean: 40.01918570832667 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17285.54784694191,
            "unit": "iter/sec",
            "range": "stddev: 0.000005981471435082702",
            "extra": "mean: 57.85179670639805 usec\nrounds: 11476"
          }
        ]
      }
    ]
  }
}