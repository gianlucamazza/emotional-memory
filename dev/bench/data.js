window.BENCHMARK_DATA = {
  "lastUpdate": 1782580086661,
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
          "id": "980a3f7d09b9b2c00489966db7ec4aae031f89db",
          "message": "docs(paper): correct arXiv checklist page count (13->18pp) (#87)\n\nThe bundle PDF grew to 18pp / ~533KB as addenda R/S/U/V/T/T2A/W were folded into\nthe paper; the checklist note was stale at \"13pp, ~497KB\".\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T19:02:17+02:00",
          "tree_id": "d564533819b668338cf87c5301938eece9361645",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/980a3f7d09b9b2c00489966db7ec4aae031f89db"
        },
        "date": 1782580085760,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 543.4061288348936,
            "unit": "iter/sec",
            "range": "stddev: 0.0008874764700356582",
            "extra": "mean: 1.8402442426331855 msec\nrounds: 1595"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 440.1183058528674,
            "unit": "iter/sec",
            "range": "stddev: 0.0011146554770481319",
            "extra": "mean: 2.2721163530387267 msec\nrounds: 1810"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 377.624586450364,
            "unit": "iter/sec",
            "range": "stddev: 0.0014752338000842895",
            "extra": "mean: 2.6481326584158804 msec\nrounds: 2222"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 226.7714396701424,
            "unit": "iter/sec",
            "range": "stddev: 0.002866907130548783",
            "extra": "mean: 4.409726381128865 msec\nrounds: 3508"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 470.54694355944696,
            "unit": "iter/sec",
            "range": "stddev: 0.0009279840491099093",
            "extra": "mean: 2.125186474351552 msec\nrounds: 1735"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.9218912348789,
            "unit": "iter/sec",
            "range": "stddev: 0.0002889064558324637",
            "extra": "mean: 2.6114986447369604 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134287.99881732374,
            "unit": "iter/sec",
            "range": "stddev: 0.000001182592441615409",
            "extra": "mean: 7.446681824191392 usec\nrounds: 45412"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.12155018883298,
            "unit": "iter/sec",
            "range": "stddev: 0.0014912129599849294",
            "extra": "mean: 35.559917333330304 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8230083077558574,
            "unit": "iter/sec",
            "range": "stddev: 0.00414493119108265",
            "extra": "mean: 1.2150545633333347 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.035403613230236414,
            "unit": "iter/sec",
            "range": "stddev: 2.132818568809699",
            "extra": "mean: 28.245704569666668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3153.631041238527,
            "unit": "iter/sec",
            "range": "stddev: 0.000013614038991013975",
            "extra": "mean: 317.0947986379756 usec\nrounds: 2349"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.0203629236877,
            "unit": "iter/sec",
            "range": "stddev: 0.00004121912877608561",
            "extra": "mean: 1.160065403337261 msec\nrounds: 719"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.4194731055935,
            "unit": "iter/sec",
            "range": "stddev: 0.00003210619466221019",
            "extra": "mean: 2.8215153959728 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.0977430518113,
            "unit": "iter/sec",
            "range": "stddev: 0.000829950198891653",
            "extra": "mean: 2.0321166152853367 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10187.500616127543,
            "unit": "iter/sec",
            "range": "stddev: 0.000004941946197006747",
            "extra": "mean: 98.15950326588727 usec\nrounds: 6430"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10270.625253434488,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056164740280966115",
            "extra": "mean: 97.3650557122217 usec\nrounds: 8149"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1815.860679614018,
            "unit": "iter/sec",
            "range": "stddev: 0.000010407284612022764",
            "extra": "mean: 550.7030419385267 usec\nrounds: 1073"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1821.2975863819945,
            "unit": "iter/sec",
            "range": "stddev: 0.00001149664112781292",
            "extra": "mean: 549.0590925267181 usec\nrounds: 1686"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1011.6693601526388,
            "unit": "iter/sec",
            "range": "stddev: 0.000017368839055061647",
            "extra": "mean: 988.4652430801324 usec\nrounds: 831"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 375.66359286348586,
            "unit": "iter/sec",
            "range": "stddev: 0.00006117455481515938",
            "extra": "mean: 2.6619561197760113 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.34715841304474,
            "unit": "iter/sec",
            "range": "stddev: 0.0006792806136572414",
            "extra": "mean: 44.74841863636088 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.95404197758575,
            "unit": "iter/sec",
            "range": "stddev: 0.000029427036715280526",
            "extra": "mean: 2.1278676438061033 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.1464727888186,
            "unit": "iter/sec",
            "range": "stddev: 0.000041985269204501077",
            "extra": "mean: 2.6585388202255826 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.49769316896106,
            "unit": "iter/sec",
            "range": "stddev: 0.00010291474534146766",
            "extra": "mean: 3.284097129251864 msec\nrounds: 294"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.27686550089686,
            "unit": "iter/sec",
            "range": "stddev: 0.00010018123270519735",
            "extra": "mean: 5.173924967214246 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 849.7785233175116,
            "unit": "iter/sec",
            "range": "stddev: 0.000022973149911587537",
            "extra": "mean: 1.1767772102500635 msec\nrounds: 761"
          }
        ]
      }
    ]
  }
}