window.BENCHMARK_DATA = {
  "lastUpdate": 1789203560974,
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
          "id": "037483f76a4438a878651a5e8f253caccb387be5",
          "message": "fix(ci): restore metadata and benchmark gates",
          "timestamp": "2026-09-12T10:51:54+02:00",
          "tree_id": "5e968ac525a73f9ccd5136e8a8dee7e1adc34dba",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/037483f76a4438a878651a5e8f253caccb387be5"
        },
        "date": 1789203559521,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 504.44869533328455,
            "unit": "iter/sec",
            "range": "stddev: 0.0008085794844944131",
            "extra": "mean: 1.982362149513162 msec\nrounds: 1438"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 471.731079990552,
            "unit": "iter/sec",
            "range": "stddev: 0.0010667223113711936",
            "extra": "mean: 2.1198518444449923 msec\nrounds: 1305"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 520.015740762121,
            "unit": "iter/sec",
            "range": "stddev: 0.0007293537712955875",
            "extra": "mean: 1.9230187119613475 msec\nrounds: 1229"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 333.60064762881706,
            "unit": "iter/sec",
            "range": "stddev: 0.0019353876165502872",
            "extra": "mean: 2.9975960991318473 msec\nrounds: 2189"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 471.0685296983299,
            "unit": "iter/sec",
            "range": "stddev: 0.0008086663684888799",
            "extra": "mean: 2.1228333818869105 msec\nrounds: 1325"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 335.22849972627563,
            "unit": "iter/sec",
            "range": "stddev: 0.00029079195830793733",
            "extra": "mean: 2.9830399289336405 msec\nrounds: 394"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137100.2990703193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010148971390785127",
            "extra": "mean: 7.293930113800087 usec\nrounds: 45803"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 12.326039218841894,
            "unit": "iter/sec",
            "range": "stddev: 0.0007476659585762136",
            "extra": "mean: 81.12906200001173 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5632537226510078,
            "unit": "iter/sec",
            "range": "stddev: 0.011898475566969744",
            "extra": "mean: 1.77539882966668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.028448429340609514,
            "unit": "iter/sec",
            "range": "stddev: 1.1949801559853521",
            "extra": "mean: 35.151325510000014 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3523.9443225253585,
            "unit": "iter/sec",
            "range": "stddev: 0.000009875163156481738",
            "extra": "mean: 283.7729284222549 usec\nrounds: 2375"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 887.0378673642634,
            "unit": "iter/sec",
            "range": "stddev: 0.00001710958365959548",
            "extra": "mean: 1.1273475877319548 msec\nrounds: 701"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.68808384231767,
            "unit": "iter/sec",
            "range": "stddev: 0.000040373837524611446",
            "extra": "mean: 2.811452071144774 msec\nrounds: 253"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 443.3471868718527,
            "unit": "iter/sec",
            "range": "stddev: 0.0004326444249749075",
            "extra": "mean: 2.2555686144210156 msec\nrounds: 638"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12282.652290748514,
            "unit": "iter/sec",
            "range": "stddev: 0.000004032483555104606",
            "extra": "mean: 81.41564023213583 usec\nrounds: 6557"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12135.19376837297,
            "unit": "iter/sec",
            "range": "stddev: 0.000003973639387441044",
            "extra": "mean: 82.40494705624097 usec\nrounds: 8084"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1923.532338303855,
            "unit": "iter/sec",
            "range": "stddev: 0.000013345380537764918",
            "extra": "mean: 519.8768848782581 usec\nrounds: 1025"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1944.1313843731796,
            "unit": "iter/sec",
            "range": "stddev: 0.000013000643403634638",
            "extra": "mean: 514.3685288134046 usec\nrounds: 1475"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1194.6108639823099,
            "unit": "iter/sec",
            "range": "stddev: 0.00003138901119802491",
            "extra": "mean: 837.0926718901899 usec\nrounds: 1021"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 375.54297782354735,
            "unit": "iter/sec",
            "range": "stddev: 0.00008001087644345949",
            "extra": "mean: 2.6628110737031543 msec\nrounds: 1099"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.22866017379779,
            "unit": "iter/sec",
            "range": "stddev: 0.0012515384543338233",
            "extra": "mean: 44.98696692384357 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 455.9466220509031,
            "unit": "iter/sec",
            "range": "stddev: 0.00006857417125599805",
            "extra": "mean: 2.193239189933854 msec\nrounds: 2543"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.8950889220625,
            "unit": "iter/sec",
            "range": "stddev: 0.00008001461039614816",
            "extra": "mean: 2.6532582392093422 msec\nrounds: 1112"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 311.41982302669686,
            "unit": "iter/sec",
            "range": "stddev: 0.00008621976238943454",
            "extra": "mean: 3.211099377942533 msec\nrounds: 680"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 201.80449728852852,
            "unit": "iter/sec",
            "range": "stddev: 0.00030056896501774017",
            "extra": "mean: 4.955290954543283 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 963.9891528162914,
            "unit": "iter/sec",
            "range": "stddev: 0.000024672143653564135",
            "extra": "mean: 1.0373560709459262 msec\nrounds: 1184"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5502.363683324465,
            "unit": "iter/sec",
            "range": "stddev: 0.000010001797920963523",
            "extra": "mean: 181.74007709279797 usec\nrounds: 4527"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 80.62111878509918,
            "unit": "iter/sec",
            "range": "stddev: 0.00008876602124005214",
            "extra": "mean: 12.403697878040674 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2325.5121761941105,
            "unit": "iter/sec",
            "range": "stddev: 0.000010478331239153919",
            "extra": "mean: 430.01279900266144 usec\nrounds: 1806"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 31.308204136553815,
            "unit": "iter/sec",
            "range": "stddev: 0.00443137702832439",
            "extra": "mean: 31.940509766654184 msec\nrounds: 30"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 21580.457779679215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033822739718292515",
            "extra": "mean: 46.338219986307664 usec\nrounds: 15651"
          }
        ]
      }
    ]
  }
}