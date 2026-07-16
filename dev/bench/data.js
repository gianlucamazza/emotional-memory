window.BENCHMARK_DATA = {
  "lastUpdate": 1784237027094,
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
          "id": "54db5ef925f97707340a4ebcf80610fcbabfed1c",
          "message": "docs: cross-link performance guide and fix CONTRIBUTING table\n\nWire Performance & Scaling from index, limitations, stores API,\nconfiguration guide, production-readiness, and README production section.\nMark gap-analysis items #8/#11 status. Repair the CONTRIBUTING test-suite\ntable that the CI alert note had split, and list bench-perf-profile.",
          "timestamp": "2026-07-16T23:17:09+02:00",
          "tree_id": "bc4c435bdb01a5e11e65ac782163f8c84863de49",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/54db5ef925f97707340a4ebcf80610fcbabfed1c"
        },
        "date": 1784237025988,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 852.5562223601916,
            "unit": "iter/sec",
            "range": "stddev: 0.0005554216056942077",
            "extra": "mean: 1.1729431722774006 msec\nrounds: 1515"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 404.4684307606745,
            "unit": "iter/sec",
            "range": "stddev: 0.0012742370530467236",
            "extra": "mean: 2.47238084346737 msec\nrounds: 3207"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 363.2149773087168,
            "unit": "iter/sec",
            "range": "stddev: 0.001509765805411155",
            "extra": "mean: 2.7531904312140845 msec\nrounds: 3838"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 195.64623818316832,
            "unit": "iter/sec",
            "range": "stddev: 0.002954372986242894",
            "extra": "mean: 5.111266177598457 msec\nrounds: 6937"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 592.0371402031099,
            "unit": "iter/sec",
            "range": "stddev: 0.0008900632732667776",
            "extra": "mean: 1.689083221462982 msec\nrounds: 2050"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 535.503412947822,
            "unit": "iter/sec",
            "range": "stddev: 0.0002664031254214004",
            "extra": "mean: 1.8674017304487982 msec\nrounds: 601"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 217378.14639124653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015798152197157834",
            "extra": "mean: 4.600278439214203 usec\nrounds: 46951"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 38.73765850479922,
            "unit": "iter/sec",
            "range": "stddev: 0.004952921074308957",
            "extra": "mean: 25.814673333343308 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.22612032024723,
            "unit": "iter/sec",
            "range": "stddev: 0.015027277398969145",
            "extra": "mean: 815.5806436666543 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.056137835347488244,
            "unit": "iter/sec",
            "range": "stddev: 0.23974777539665557",
            "extra": "mean: 17.81329817599999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 5558.78803423347,
            "unit": "iter/sec",
            "range": "stddev: 0.000005636148909163639",
            "extra": "mean: 179.89532859349174 usec\nrounds: 3868"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1465.7350860315562,
            "unit": "iter/sec",
            "range": "stddev: 0.00006173645595147233",
            "extra": "mean: 682.2515265752946 usec\nrounds: 1016"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 560.1546953078789,
            "unit": "iter/sec",
            "range": "stddev: 0.000109468779479959",
            "extra": "mean: 1.7852211333341912 msec\nrounds: 405"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 604.6780434073177,
            "unit": "iter/sec",
            "range": "stddev: 0.00044185415692636154",
            "extra": "mean: 1.6537726330611764 msec\nrounds: 1101"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 18630.186858498746,
            "unit": "iter/sec",
            "range": "stddev: 0.000003442737208081477",
            "extra": "mean: 53.67632689866546 usec\nrounds: 11863"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 17729.165804197044,
            "unit": "iter/sec",
            "range": "stddev: 0.000008760051975943027",
            "extra": "mean: 56.40423306116687 usec\nrounds: 8693"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 3424.9423556066913,
            "unit": "iter/sec",
            "range": "stddev: 0.000012065675279462568",
            "extra": "mean: 291.97571701111474 usec\nrounds: 1258"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 3371.1548948136356,
            "unit": "iter/sec",
            "range": "stddev: 0.000011755702492993428",
            "extra": "mean: 296.63424885591974 usec\nrounds: 2403"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1868.0375510693548,
            "unit": "iter/sec",
            "range": "stddev: 0.000016888690239300478",
            "extra": "mean: 535.3211446030899 usec\nrounds: 1473"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 547.4786871336311,
            "unit": "iter/sec",
            "range": "stddev: 0.00003710000375172605",
            "extra": "mean: 1.8265551216899067 msec\nrounds: 1586"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 54.22773342482375,
            "unit": "iter/sec",
            "range": "stddev: 0.0017632458244369927",
            "extra": "mean: 18.440748614107324 msec\nrounds: 964"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 675.395145870629,
            "unit": "iter/sec",
            "range": "stddev: 0.00006370529444033793",
            "extra": "mean: 1.4806147277101522 msec\nrounds: 3533"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 563.3353330384721,
            "unit": "iter/sec",
            "range": "stddev: 0.0000304431042821784",
            "extra": "mean: 1.7751416276452638 msec\nrounds: 1606"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 469.2468255949904,
            "unit": "iter/sec",
            "range": "stddev: 0.00004509730151321697",
            "extra": "mean: 2.1310746188469807 msec\nrounds: 955"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 299.48692164246734,
            "unit": "iter/sec",
            "range": "stddev: 0.00028971787423644814",
            "extra": "mean: 3.339043970653975 msec\nrounds: 443"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1422.4579076944456,
            "unit": "iter/sec",
            "range": "stddev: 0.000015768474072491792",
            "extra": "mean: 703.0084999990083 usec\nrounds: 1876"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 8921.83330764654,
            "unit": "iter/sec",
            "range": "stddev: 0.000006366029276234171",
            "extra": "mean: 112.08458682398164 usec\nrounds: 7043"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 128.58878931485296,
            "unit": "iter/sec",
            "range": "stddev: 0.0001967779376588723",
            "extra": "mean: 7.776727701755354 msec\nrounds: 114"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 3509.80224051862,
            "unit": "iter/sec",
            "range": "stddev: 0.000009272198250423614",
            "extra": "mean: 284.91633758038654 usec\nrounds: 2669"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 42.3907297841029,
            "unit": "iter/sec",
            "range": "stddev: 0.00618338121092095",
            "extra": "mean: 23.590063325001154 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 39721.705227591425,
            "unit": "iter/sec",
            "range": "stddev: 0.000002757882286431234",
            "extra": "mean: 25.17515283571919 usec\nrounds: 25498"
          }
        ]
      }
    ]
  }
}