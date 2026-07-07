window.BENCHMARK_DATA = {
  "lastUpdate": 1783443893435,
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
          "id": "e65c94980904a7ca353efe32f91158c6be33dd37",
          "message": "chore(release): v0.16.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.21246683\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T18:58:04+02:00",
          "tree_id": "3483646df012482f2c1184a9fe78b9232e9e00c7",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e65c94980904a7ca353efe32f91158c6be33dd37"
        },
        "date": 1783443891423,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 540.3337599597556,
            "unit": "iter/sec",
            "range": "stddev: 0.0008606037282676327",
            "extra": "mean: 1.8507079773702841 msec\nrounds: 1635"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 444.63665657343523,
            "unit": "iter/sec",
            "range": "stddev: 0.001075212106827254",
            "extra": "mean: 2.249027346747427 msec\nrounds: 1814"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 385.7922860707316,
            "unit": "iter/sec",
            "range": "stddev: 0.0014241575023553738",
            "extra": "mean: 2.592068416361904 msec\nrounds: 2188"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 213.53896351942916,
            "unit": "iter/sec",
            "range": "stddev: 0.0030428126934625137",
            "extra": "mean: 4.68298610950696 msec\nrounds: 3671"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 485.4868527862517,
            "unit": "iter/sec",
            "range": "stddev: 0.0008802506516647664",
            "extra": "mean: 2.059788013333239 msec\nrounds: 1650"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.9758635065902,
            "unit": "iter/sec",
            "range": "stddev: 0.0002490406584860652",
            "extra": "mean: 2.611130609756539 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134610.8656932248,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011967666953829583",
            "extra": "mean: 7.428820807667771 usec\nrounds: 33824"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.324055510997226,
            "unit": "iter/sec",
            "range": "stddev: 0.0003048791971478769",
            "extra": "mean: 36.59778833334334 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8189599181795065,
            "unit": "iter/sec",
            "range": "stddev: 0.012048289654050318",
            "extra": "mean: 1.221060979666665 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03149919584228773,
            "unit": "iter/sec",
            "range": "stddev: 2.315224093103918",
            "extra": "mean: 31.74684220533332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3152.7704653663104,
            "unit": "iter/sec",
            "range": "stddev: 0.000013560131731434511",
            "extra": "mean: 317.1813523963005 usec\nrounds: 2483"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 863.2587261833979,
            "unit": "iter/sec",
            "range": "stddev: 0.00001711444733032109",
            "extra": "mean: 1.158401264498254 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.19587428316555,
            "unit": "iter/sec",
            "range": "stddev: 0.00014387680318444943",
            "extra": "mean: 2.815348016128111 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 487.54285048716315,
            "unit": "iter/sec",
            "range": "stddev: 0.0008073613124111113",
            "extra": "mean: 2.0511017626466654 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10548.87210934147,
            "unit": "iter/sec",
            "range": "stddev: 0.000006192943476499505",
            "extra": "mean: 94.79686450217346 usec\nrounds: 7454"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10221.752668987878,
            "unit": "iter/sec",
            "range": "stddev: 0.000005890758043129933",
            "extra": "mean: 97.8305807607666 usec\nrounds: 8067"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1832.4544433517774,
            "unit": "iter/sec",
            "range": "stddev: 0.000011348902806144527",
            "extra": "mean: 545.7161587989499 usec\nrounds: 1165"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1807.1333639495952,
            "unit": "iter/sec",
            "range": "stddev: 0.00002549850126217043",
            "extra": "mean: 553.3625906913929 usec\nrounds: 1676"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 974.936363341895,
            "unit": "iter/sec",
            "range": "stddev: 0.000027570364255986493",
            "extra": "mean: 1.0257079719256668 msec\nrounds: 748"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 364.5327628198392,
            "unit": "iter/sec",
            "range": "stddev: 0.00006244675092802483",
            "extra": "mean: 2.7432376510262366 msec\nrounds: 341"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.35034273996672,
            "unit": "iter/sec",
            "range": "stddev: 0.001615725658854744",
            "extra": "mean: 44.74204318181695 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 466.5673785133145,
            "unit": "iter/sec",
            "range": "stddev: 0.00015829425215778658",
            "extra": "mean: 2.1433131548682907 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.36838796151136,
            "unit": "iter/sec",
            "range": "stddev: 0.00016981148321183916",
            "extra": "mean: 2.6855126061435746 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 301.64180851995786,
            "unit": "iter/sec",
            "range": "stddev: 0.00010379298218435811",
            "extra": "mean: 3.3151903076918323 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 190.89052785567534,
            "unit": "iter/sec",
            "range": "stddev: 0.00006051463705311476",
            "extra": "mean: 5.238604614033337 msec\nrounds: 171"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 827.352174398084,
            "unit": "iter/sec",
            "range": "stddev: 0.000022439660072963044",
            "extra": "mean: 1.2086751336908261 msec\nrounds: 748"
          }
        ]
      }
    ]
  }
}