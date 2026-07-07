window.BENCHMARK_DATA = {
  "lastUpdate": 1783415599839,
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
          "id": "5dda9f65514670ca4eecb8b8628ad65a9edb954a",
          "message": "docs(paper): refresh arXiv checklist footer for v0.15.0 / Addendum X2 (#108)\n\nBundle now covers A--X2 (both third-party corpora), abstract reworded to\n'both released third-party corpora' (~1890 chars), software snapshot v0.15.0\n(version DOI 10.5281/zenodo.21235738). Footer was still dated 2026-07-02\n('after Addendum X'). SUBMISSION.md + bundle already synced by the release.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T11:05:53+02:00",
          "tree_id": "47309a8b3fe7d07ca9f43a34c3b678a4a979fb90",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/5dda9f65514670ca4eecb8b8628ad65a9edb954a"
        },
        "date": 1783415598365,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 561.2923277925655,
            "unit": "iter/sec",
            "range": "stddev: 0.000836739941232809",
            "extra": "mean: 1.7816028306190672 msec\nrounds: 1535"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 443.08518182575307,
            "unit": "iter/sec",
            "range": "stddev: 0.0011419012438872232",
            "extra": "mean: 2.256902376828432 msec\nrounds: 1709"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 296.70877538742667,
            "unit": "iter/sec",
            "range": "stddev: 0.0024346923610163115",
            "extra": "mean: 3.3703081369745562 msec\nrounds: 2307"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 161.83311107124223,
            "unit": "iter/sec",
            "range": "stddev: 0.0046043987628490435",
            "extra": "mean: 6.1792051909561305 msec\nrounds: 3671"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 464.4417553748686,
            "unit": "iter/sec",
            "range": "stddev: 0.0009886381103384298",
            "extra": "mean: 2.153122514130673 msec\nrounds: 1663"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 372.0415486501013,
            "unit": "iter/sec",
            "range": "stddev: 0.00032823080023943835",
            "extra": "mean: 2.6878718348215536 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 130284.04172248533,
            "unit": "iter/sec",
            "range": "stddev: 0.000005183642138618631",
            "extra": "mean: 7.675537132399333 usec\nrounds: 33394"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.815790889118663,
            "unit": "iter/sec",
            "range": "stddev: 0.00024197118166994252",
            "extra": "mean: 35.95080233333192 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8137934566458657,
            "unit": "iter/sec",
            "range": "stddev: 0.008064657575430975",
            "extra": "mean: 1.2288130259999928 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02337323868797258,
            "unit": "iter/sec",
            "range": "stddev: 5.16809297267018",
            "extra": "mean: 42.783972446 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3141.697125047761,
            "unit": "iter/sec",
            "range": "stddev: 0.00001523893453732039",
            "extra": "mean: 318.2993013640033 usec\nrounds: 2200"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 855.629189612518,
            "unit": "iter/sec",
            "range": "stddev: 0.000022032104999949933",
            "extra": "mean: 1.1687305811210837 msec\nrounds: 678"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 348.1499809182759,
            "unit": "iter/sec",
            "range": "stddev: 0.00015954206822648751",
            "extra": "mean: 2.8723253046357002 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 481.9405035637827,
            "unit": "iter/sec",
            "range": "stddev: 0.0010395515572892118",
            "extra": "mean: 2.0749449208052595 msec\nrounds: 745"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10345.49112883378,
            "unit": "iter/sec",
            "range": "stddev: 0.000006236892906876357",
            "extra": "mean: 96.66046662713897 usec\nrounds: 6802"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10563.18079365624,
            "unit": "iter/sec",
            "range": "stddev: 0.000006151518362544084",
            "extra": "mean: 94.6684544678582 usec\nrounds: 7050"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1803.9365269262998,
            "unit": "iter/sec",
            "range": "stddev: 0.000011497982449257891",
            "extra": "mean: 554.3432294172151 usec\nrounds: 1081"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1772.6324710822194,
            "unit": "iter/sec",
            "range": "stddev: 0.000014372360026800375",
            "extra": "mean: 564.1327327088196 usec\nrounds: 1388"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 993.792596514967,
            "unit": "iter/sec",
            "range": "stddev: 0.00005886681581814594",
            "extra": "mean: 1.0062461760198267 msec\nrounds: 784"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 349.6731949179827,
            "unit": "iter/sec",
            "range": "stddev: 0.00013817331952621466",
            "extra": "mean: 2.8598131470573662 msec\nrounds: 340"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.442389164811765,
            "unit": "iter/sec",
            "range": "stddev: 0.001384931904931633",
            "extra": "mean: 48.91796120002141 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 452.53741349173157,
            "unit": "iter/sec",
            "range": "stddev: 0.0001645421022473548",
            "extra": "mean: 2.2097620443890023 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 333.9169187167426,
            "unit": "iter/sec",
            "range": "stddev: 0.00027228831617681377",
            "extra": "mean: 2.99475691091977 msec\nrounds: 348"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 256.41997645646956,
            "unit": "iter/sec",
            "range": "stddev: 0.0004242066300497231",
            "extra": "mean: 3.899852163701303 msec\nrounds: 281"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 168.23937172184654,
            "unit": "iter/sec",
            "range": "stddev: 0.0005181698444612719",
            "extra": "mean: 5.9439118784473335 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 842.1567436005842,
            "unit": "iter/sec",
            "range": "stddev: 0.000029446088943821118",
            "extra": "mean: 1.1874274089696981 msec\nrounds: 758"
          }
        ]
      }
    ]
  }
}