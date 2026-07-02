window.BENCHMARK_DATA = {
  "lastUpdate": 1782985819366,
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
          "id": "6d6a509e03306be8aabf1547b8f0d9143bd65c38",
          "message": "fix(demo): drop Chatbot type kwarg removed in Gradio 6 (Space startup crash) (#100)\n\nThe 0.14.0 code-health pass added type=\"messages\" to silence the Gradio 5\ndeprecation, but the HF Space runs Gradio 6 where the parameter was removed\nentirely: the app crashed at startup with TypeError (RUNTIME_ERROR stage).\nMessages is the only format in 6.x, so dropping the kwarg is behavior-neutral.\nTest restored to forbid the kwarg, now with the rationale inline.\n\nDemo-only; the published 0.14.0 wheel is unaffected.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T11:44:36+02:00",
          "tree_id": "ec285ba7614a70af2f9664f2594fe78bb7ff0b23",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/6d6a509e03306be8aabf1547b8f0d9143bd65c38"
        },
        "date": 1782985817678,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 569.301715382973,
            "unit": "iter/sec",
            "range": "stddev: 0.0008062150978368115",
            "extra": "mean: 1.7565378304319588 msec\nrounds: 1551"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.9463875244486,
            "unit": "iter/sec",
            "range": "stddev: 0.0010610770013829362",
            "extra": "mean: 2.2274374575417255 msec\nrounds: 1790"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 384.80766117286987,
            "unit": "iter/sec",
            "range": "stddev: 0.001429360271332655",
            "extra": "mean: 2.5987008599362658 msec\nrounds: 2199"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 242.66857385264942,
            "unit": "iter/sec",
            "range": "stddev: 0.0025734782477129935",
            "extra": "mean: 4.120846733978868 msec\nrounds: 3511"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 453.1135056194569,
            "unit": "iter/sec",
            "range": "stddev: 0.0009708630683920411",
            "extra": "mean: 2.2069525352877926 msec\nrounds: 1842"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 386.0652647216272,
            "unit": "iter/sec",
            "range": "stddev: 0.00023915951156508583",
            "extra": "mean: 2.5902356191538005 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138258.33559478293,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012042937860210235",
            "extra": "mean: 7.232836962039447 usec\nrounds: 50798"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.342934045602984,
            "unit": "iter/sec",
            "range": "stddev: 0.00027251601604743",
            "extra": "mean: 35.282162333336 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8249387041463736,
            "unit": "iter/sec",
            "range": "stddev: 0.010256351934313354",
            "extra": "mean: 1.212211276999999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03653414518215986,
            "unit": "iter/sec",
            "range": "stddev: 1.1676122174963977",
            "extra": "mean: 27.371654517000007 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3159.443871078491,
            "unit": "iter/sec",
            "range": "stddev: 0.000013528216875992339",
            "extra": "mean: 316.5113990958938 usec\nrounds: 2433"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 865.265491647222,
            "unit": "iter/sec",
            "range": "stddev: 0.000022381361007578724",
            "extra": "mean: 1.155714644410794 msec\nrounds: 689"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.22064501846364,
            "unit": "iter/sec",
            "range": "stddev: 0.00010081190838569064",
            "extra": "mean: 2.8230991447375273 msec\nrounds: 304"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.8220980841817,
            "unit": "iter/sec",
            "range": "stddev: 0.0008022486425051051",
            "extra": "mean: 2.0291297892027247 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10440.129621061622,
            "unit": "iter/sec",
            "range": "stddev: 0.00000512305803469113",
            "extra": "mean: 95.78425137391285 usec\nrounds: 7097"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10527.297342939253,
            "unit": "iter/sec",
            "range": "stddev: 0.000005708208280045176",
            "extra": "mean: 94.99114230593177 usec\nrounds: 8292"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1863.8450418812754,
            "unit": "iter/sec",
            "range": "stddev: 0.000012621403903129634",
            "extra": "mean: 536.5252891359725 usec\nrounds: 1169"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1837.6339677946878,
            "unit": "iter/sec",
            "range": "stddev: 0.000012002009148390703",
            "extra": "mean: 544.1780123383779 usec\nrounds: 1702"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1002.19417811099,
            "unit": "iter/sec",
            "range": "stddev: 0.000015651320515945168",
            "extra": "mean: 997.810625766031 usec\nrounds: 815"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 376.7278556199514,
            "unit": "iter/sec",
            "range": "stddev: 0.00004352915629425069",
            "extra": "mean: 2.654436047354074 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.331192443204845,
            "unit": "iter/sec",
            "range": "stddev: 0.0009731676656739316",
            "extra": "mean: 44.78041208696358 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 473.116898743739,
            "unit": "iter/sec",
            "range": "stddev: 0.00006866136025346615",
            "extra": "mean: 2.113642532438149 msec\nrounds: 447"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.9802409383254,
            "unit": "iter/sec",
            "range": "stddev: 0.00003731411876757657",
            "extra": "mean: 2.624808041322763 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 307.2686694781391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000455368070184976",
            "extra": "mean: 3.2544808479770695 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.19515074776953,
            "unit": "iter/sec",
            "range": "stddev: 0.0001639722154700316",
            "extra": "mean: 5.0969659351347065 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 846.9271806447593,
            "unit": "iter/sec",
            "range": "stddev: 0.00003325048903154664",
            "extra": "mean: 1.1807390562653894 msec\nrounds: 782"
          }
        ]
      }
    ]
  }
}