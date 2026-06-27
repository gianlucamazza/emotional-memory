window.BENCHMARK_DATA = {
  "lastUpdate": 1782552098733,
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
          "id": "49a08c5434a8b6a8ffc4c32f7f422eb696f28b05",
          "message": "chore(release): v0.12.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.20959964\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-06-27T11:15:22+02:00",
          "tree_id": "24a938edc762816f520216a802f219a03a64593a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/49a08c5434a8b6a8ffc4c32f7f422eb696f28b05"
        },
        "date": 1782552097450,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 552.636907450936,
            "unit": "iter/sec",
            "range": "stddev: 0.0008267265936515778",
            "extra": "mean: 1.809506362165617 msec\nrounds: 1607"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.2198503806289,
            "unit": "iter/sec",
            "range": "stddev: 0.0010259832291036437",
            "extra": "mean: 2.231047998322249 msec\nrounds: 1788"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 354.936995655762,
            "unit": "iter/sec",
            "range": "stddev: 0.001576810463319768",
            "extra": "mean: 2.8174014324780523 msec\nrounds: 2377"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 226.8851276207195,
            "unit": "iter/sec",
            "range": "stddev: 0.0027077725922581403",
            "extra": "mean: 4.4075167486151186 msec\nrounds: 3791"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 472.464221848911,
            "unit": "iter/sec",
            "range": "stddev: 0.0009061897892597723",
            "extra": "mean: 2.1165623845265245 msec\nrounds: 1732"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 386.58858843001366,
            "unit": "iter/sec",
            "range": "stddev: 0.0002400347318382981",
            "extra": "mean: 2.5867292256637207 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135848.37958668516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011145773758862852",
            "extra": "mean: 7.361147796112634 usec\nrounds: 49981"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.54088411724517,
            "unit": "iter/sec",
            "range": "stddev: 0.0002574862738720724",
            "extra": "mean: 35.03745700000138 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8315874736435402,
            "unit": "iter/sec",
            "range": "stddev: 0.0032555751246184125",
            "extra": "mean: 1.2025193160000025 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.037748255373049176,
            "unit": "iter/sec",
            "range": "stddev: 1.2482723126449367",
            "extra": "mean: 26.491290527666667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3158.2434792245613,
            "unit": "iter/sec",
            "range": "stddev: 0.000014699047978352197",
            "extra": "mean: 316.63169941714835 usec\nrounds: 2402"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 851.4985177761522,
            "unit": "iter/sec",
            "range": "stddev: 0.00003329849420283567",
            "extra": "mean: 1.174400165265921 msec\nrounds: 714"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 352.45191405554544,
            "unit": "iter/sec",
            "range": "stddev: 0.00003320002100609575",
            "extra": "mean: 2.8372664755692107 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 488.6664960583465,
            "unit": "iter/sec",
            "range": "stddev: 0.000725983587031151",
            "extra": "mean: 2.0463854347824997 msec\nrounds: 782"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10581.004213727572,
            "unit": "iter/sec",
            "range": "stddev: 0.000005293640577188352",
            "extra": "mean: 94.50898797513197 usec\nrounds: 7651"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10416.53110424462,
            "unit": "iter/sec",
            "range": "stddev: 0.000013600994781885223",
            "extra": "mean: 96.00124935954075 usec\nrounds: 8197"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1833.0420380805908,
            "unit": "iter/sec",
            "range": "stddev: 0.000011869767334000394",
            "extra": "mean: 545.5412255831933 usec\nrounds: 1157"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1820.7070454624477,
            "unit": "iter/sec",
            "range": "stddev: 0.000012675320467804821",
            "extra": "mean: 549.2371782117241 usec\nrounds: 1689"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1004.6605968840158,
            "unit": "iter/sec",
            "range": "stddev: 0.000025245672012015218",
            "extra": "mean: 995.3610235153337 usec\nrounds: 808"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 377.61554786775156,
            "unit": "iter/sec",
            "range": "stddev: 0.00003479249596723565",
            "extra": "mean: 2.648196043956908 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.117296545836787,
            "unit": "iter/sec",
            "range": "stddev: 0.0005095018726241303",
            "extra": "mean: 41.46401725000241 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.8136236341084,
            "unit": "iter/sec",
            "range": "stddev: 0.00007234966784052868",
            "extra": "mean: 2.1239827180046675 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.1113194216138,
            "unit": "iter/sec",
            "range": "stddev: 0.00004002774839695888",
            "extra": "mean: 2.630808263015222 msec\nrounds: 365"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 305.60243045063294,
            "unit": "iter/sec",
            "range": "stddev: 0.00004433728041900945",
            "extra": "mean: 3.2722252847447173 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.1976667785679,
            "unit": "iter/sec",
            "range": "stddev: 0.00008588370676999102",
            "extra": "mean: 5.149392454546021 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 843.478135189608,
            "unit": "iter/sec",
            "range": "stddev: 0.000028679084646363223",
            "extra": "mean: 1.1855671869611735 msec\nrounds: 813"
          }
        ]
      }
    ]
  }
}