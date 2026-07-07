window.BENCHMARK_DATA = {
  "lastUpdate": 1783389126957,
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
          "id": "915b1fb1c2fa2c4cf9dd76b69e4d2e486aff1bc5",
          "message": "docs: propagate addendum x2 to residual surface (#105)\n\nPattern #94 residual propagation for the X2 closure (#104):\n\n- README: headline FAIL list (both third-party corpora), new 'When NOT to\n  use' bullet (content-determined QA, affect-orthogonal), research section\n  reworded to the two-corpora provenance bound\n- benchmarks/README: X2 row in the addenda verdict table + esmemeval/ in\n  the harness directory index\n- docs/research/index.md: X2 ladder entry (2026-07-07)\n- problem register: item 3 residual X2 -> EXECUTED with realized numbers\n- 10_scientific_quality_bar: second external data point note (distinct\n  failure mode; what the two tests jointly imply for Gate 2)\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T03:46:21+02:00",
          "tree_id": "c1451f8acb8c5aabacb9852b34068263fbb28d98",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/915b1fb1c2fa2c4cf9dd76b69e4d2e486aff1bc5"
        },
        "date": 1783389126309,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 598.4242897787149,
            "unit": "iter/sec",
            "range": "stddev: 0.0007929352762580908",
            "extra": "mean: 1.67105516450507 msec\nrounds: 1465"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 399.52898522533616,
            "unit": "iter/sec",
            "range": "stddev: 0.0013146452814774383",
            "extra": "mean: 2.502947312911466 msec\nrounds: 1975"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 317.2145139561822,
            "unit": "iter/sec",
            "range": "stddev: 0.001863458489958081",
            "extra": "mean: 3.1524408751931605 msec\nrounds: 2588"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 184.55474167160986,
            "unit": "iter/sec",
            "range": "stddev: 0.003447322134722349",
            "extra": "mean: 5.418446532137139 msec\nrounds: 4403"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 418.25084691058686,
            "unit": "iter/sec",
            "range": "stddev: 0.0011132087869111427",
            "extra": "mean: 2.3909096834746606 msec\nrounds: 2003"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.2113776574323,
            "unit": "iter/sec",
            "range": "stddev: 0.0003105124640698508",
            "extra": "mean: 2.6301159269909924 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135331.90844242525,
            "unit": "iter/sec",
            "range": "stddev: 8.862981972372072e-7",
            "extra": "mean: 7.3892403610448865 usec\nrounds: 43547"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.983527630495836,
            "unit": "iter/sec",
            "range": "stddev: 0.0004722523991727406",
            "extra": "mean: 30.318164000002906 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.839996453770174,
            "unit": "iter/sec",
            "range": "stddev: 0.013435881735065687",
            "extra": "mean: 1.1904812163333294 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.036449805436489874,
            "unit": "iter/sec",
            "range": "stddev: 0.5839869682702183",
            "extra": "mean: 27.434988692666675 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3440.607497471807,
            "unit": "iter/sec",
            "range": "stddev: 0.000009358685619708623",
            "extra": "mean: 290.6463468253237 usec\nrounds: 2520"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 884.8880083209873,
            "unit": "iter/sec",
            "range": "stddev: 0.00001627437550807887",
            "extra": "mean: 1.1300865087972316 msec\nrounds: 682"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 353.9181360509757,
            "unit": "iter/sec",
            "range": "stddev: 0.00003603767596639801",
            "extra": "mean: 2.825512168316708 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.51414327435333,
            "unit": "iter/sec",
            "range": "stddev: 0.0004552217893054289",
            "extra": "mean: 1.993961712567197 msec\nrounds: 748"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12344.35609233533,
            "unit": "iter/sec",
            "range": "stddev: 0.000003870539251400224",
            "extra": "mean: 81.00868060837169 usec\nrounds: 7364"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12118.973683354481,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036267062405359962",
            "extra": "mean: 82.5152381817207 usec\nrounds: 8271"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1996.9451070599434,
            "unit": "iter/sec",
            "range": "stddev: 0.000010030021340521297",
            "extra": "mean: 500.7648915659364 usec\nrounds: 1162"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2012.6348326604016,
            "unit": "iter/sec",
            "range": "stddev: 0.000011710145707180465",
            "extra": "mean: 496.86112143758834 usec\nrounds: 1614"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1165.8222056836837,
            "unit": "iter/sec",
            "range": "stddev: 0.00002246458802203506",
            "extra": "mean: 857.7637268570991 usec\nrounds: 875"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 385.24708364278047,
            "unit": "iter/sec",
            "range": "stddev: 0.00005830689879058842",
            "extra": "mean: 2.595736716665837 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.546633621391344,
            "unit": "iter/sec",
            "range": "stddev: 0.00030587523190719326",
            "extra": "mean: 44.35251917391517 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.4805660673946,
            "unit": "iter/sec",
            "range": "stddev: 0.000041929593187931844",
            "extra": "mean: 2.120978195010179 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 387.4861221795878,
            "unit": "iter/sec",
            "range": "stddev: 0.00004785259659680388",
            "extra": "mean: 2.5807375871297165 msec\nrounds: 373"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 318.6220998672774,
            "unit": "iter/sec",
            "range": "stddev: 0.00004832598088754161",
            "extra": "mean: 3.1385142474942938 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 209.41533911299248,
            "unit": "iter/sec",
            "range": "stddev: 0.00006470058019160926",
            "extra": "mean: 4.775199391962584 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 953.247119434663,
            "unit": "iter/sec",
            "range": "stddev: 0.000020007883892168906",
            "extra": "mean: 1.0490459185369103 msec\nrounds: 847"
          }
        ]
      }
    ]
  }
}