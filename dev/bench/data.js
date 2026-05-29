window.BENCHMARK_DATA = {
  "lastUpdate": 1780041710204,
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
          "id": "1fa7ab1fd260a993435bf0d58388948e5869316d",
          "message": "chore(release): v0.11.1\n\nPrereserved Zenodo DOI: 10.5281/zenodo.20440996\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-29T09:53:33+02:00",
          "tree_id": "68a617debf24de40d9a5a30c39e4fc5b9e2d456d",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1fa7ab1fd260a993435bf0d58388948e5869316d"
        },
        "date": 1780041709653,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 631.6722639263367,
            "unit": "iter/sec",
            "range": "stddev: 0.0007541172044730964",
            "extra": "mean: 1.5830994284666209 msec\nrounds: 1363"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 412.96891166960984,
            "unit": "iter/sec",
            "range": "stddev: 0.001418299667721161",
            "extra": "mean: 2.4214897822624395 msec\nrounds: 1883"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 279.61276191042987,
            "unit": "iter/sec",
            "range": "stddev: 0.002611091524467666",
            "extra": "mean: 3.576374673200132 msec\nrounds: 2500"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 162.45369844327112,
            "unit": "iter/sec",
            "range": "stddev: 0.004884170776571172",
            "extra": "mean: 6.155600085332623 msec\nrounds: 3668"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 411.0241371910559,
            "unit": "iter/sec",
            "range": "stddev: 0.0013336148870171666",
            "extra": "mean: 2.432947142311429 msec\nrounds: 1834"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 370.8223060237315,
            "unit": "iter/sec",
            "range": "stddev: 0.0005657238897533271",
            "extra": "mean: 2.6967094043582236 msec\nrounds: 413"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 132815.47007492348,
            "unit": "iter/sec",
            "range": "stddev: 9.732304273525273e-7",
            "extra": "mean: 7.529243388860371 usec\nrounds: 32294"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.5311904445329,
            "unit": "iter/sec",
            "range": "stddev: 0.0004706139738220259",
            "extra": "mean: 30.73972966667308 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8136301024469936,
            "unit": "iter/sec",
            "range": "stddev: 0.01359412906584862",
            "extra": "mean: 1.2290597373333394 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.021337176194498625,
            "unit": "iter/sec",
            "range": "stddev: 0.8174680153562318",
            "extra": "mean: 46.86655773399999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3529.753132101493,
            "unit": "iter/sec",
            "range": "stddev: 0.00000856344077619478",
            "extra": "mean: 283.3059317677082 usec\nrounds: 2213"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 913.510801249907,
            "unit": "iter/sec",
            "range": "stddev: 0.00002178399772650589",
            "extra": "mean: 1.094677806361736 msec\nrounds: 723"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.3442246175153,
            "unit": "iter/sec",
            "range": "stddev: 0.00021691272684047386",
            "extra": "mean: 2.7751242608687363 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 481.13935998489785,
            "unit": "iter/sec",
            "range": "stddev: 0.0012970558258221456",
            "extra": "mean: 2.07839990482464 msec\nrounds: 767"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12304.119358658716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034369930668680766",
            "extra": "mean: 81.27359389571225 usec\nrounds: 6651"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12321.06798743259,
            "unit": "iter/sec",
            "range": "stddev: 0.00000372835059086133",
            "extra": "mean: 81.16179547259974 usec\nrounds: 8615"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2058.2341992721995,
            "unit": "iter/sec",
            "range": "stddev: 0.000013911237488092412",
            "extra": "mean: 485.85335932791537 usec\nrounds: 1013"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1939.8018311143055,
            "unit": "iter/sec",
            "range": "stddev: 0.00010227197656641578",
            "extra": "mean: 515.5165769822771 usec\nrounds: 1286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1177.3710604704172,
            "unit": "iter/sec",
            "range": "stddev: 0.000025040592052284498",
            "extra": "mean: 849.3499063926806 usec\nrounds: 876"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 387.3814288994048,
            "unit": "iter/sec",
            "range": "stddev: 0.00007749254987471088",
            "extra": "mean: 2.5814350544400515 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.579422937585175,
            "unit": "iter/sec",
            "range": "stddev: 0.0005722508453694446",
            "extra": "mean: 51.07402823810368 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.12047630539496,
            "unit": "iter/sec",
            "range": "stddev: 0.00005400555014749967",
            "extra": "mean: 2.131648586042544 msec\nrounds: 430"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.63158640403265,
            "unit": "iter/sec",
            "range": "stddev: 0.00014873663017503258",
            "extra": "mean: 2.627212337912809 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 309.35254284923576,
            "unit": "iter/sec",
            "range": "stddev: 0.00010227898801583124",
            "extra": "mean: 3.2325578797241503 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 205.24213829117488,
            "unit": "iter/sec",
            "range": "stddev: 0.00014121114368382664",
            "extra": "mean: 4.872293810256988 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 966.957472663632,
            "unit": "iter/sec",
            "range": "stddev: 0.00002482760429787862",
            "extra": "mean: 1.0341716448452973 msec\nrounds: 825"
          }
        ]
      }
    ]
  }
}