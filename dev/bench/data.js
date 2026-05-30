window.BENCHMARK_DATA = {
  "lastUpdate": 1780158793905,
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
          "id": "1164222deda93f3ec839e9805a5ddf4ac71d4669",
          "message": "chore(release): v0.11.2\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T18:23:50+02:00",
          "tree_id": "27aba5bc98ba501a8936c78ccc9adeaca0520a38",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1164222deda93f3ec839e9805a5ddf4ac71d4669"
        },
        "date": 1780158793082,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 498.921084159599,
            "unit": "iter/sec",
            "range": "stddev: 0.0010225381719293474",
            "extra": "mean: 2.004324995974938 msec\nrounds: 1739"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 391.4671674863977,
            "unit": "iter/sec",
            "range": "stddev: 0.001350578052181065",
            "extra": "mean: 2.5544926447369223 msec\nrounds: 1900"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 284.57673254071784,
            "unit": "iter/sec",
            "range": "stddev: 0.0023733138827746665",
            "extra": "mean: 3.513990729572095 msec\nrounds: 2570"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 167.75302622680314,
            "unit": "iter/sec",
            "range": "stddev: 0.00356164175425777",
            "extra": "mean: 5.961144323250502 msec\nrounds: 4430"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 431.77615051703896,
            "unit": "iter/sec",
            "range": "stddev: 0.0010529539824165723",
            "extra": "mean: 2.3160149044881937 msec\nrounds: 1916"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 377.39057185713784,
            "unit": "iter/sec",
            "range": "stddev: 0.0002810851890841849",
            "extra": "mean: 2.6497747282848194 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 149211.22098875142,
            "unit": "iter/sec",
            "range": "stddev: 7.501677490540472e-7",
            "extra": "mean: 6.70190883348771 usec\nrounds: 43437"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.2529244367654,
            "unit": "iter/sec",
            "range": "stddev: 0.0002679234368436747",
            "extra": "mean: 30.072542999988627 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8084744453051799,
            "unit": "iter/sec",
            "range": "stddev: 0.016637742941055474",
            "extra": "mean: 1.2368974749999968 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.028370079482793454,
            "unit": "iter/sec",
            "range": "stddev: 1.5142935783341112",
            "extra": "mean: 35.248403185 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3614.74627278265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075352857893664075",
            "extra": "mean: 276.64458983733726 usec\nrounds: 2460"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 903.0420704590911,
            "unit": "iter/sec",
            "range": "stddev: 0.00003772468398538039",
            "extra": "mean: 1.1073681201714305 msec\nrounds: 699"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 357.07155515593587,
            "unit": "iter/sec",
            "range": "stddev: 0.00006951855298226455",
            "extra": "mean: 2.800559119203131 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 488.7418614731984,
            "unit": "iter/sec",
            "range": "stddev: 0.00044229299573510586",
            "extra": "mean: 2.0460698762036325 msec\nrounds: 727"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13241.727159254,
            "unit": "iter/sec",
            "range": "stddev: 0.000002752345853234996",
            "extra": "mean: 75.51884946527905 usec\nrounds: 7759"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12955.805213248157,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025887030620652433",
            "extra": "mean: 77.18547659063559 usec\nrounds: 7497"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2089.259393958779,
            "unit": "iter/sec",
            "range": "stddev: 0.000015610414481203294",
            "extra": "mean: 478.6385084071231 usec\nrounds: 1011"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2045.1730696432078,
            "unit": "iter/sec",
            "range": "stddev: 0.00007722977525221597",
            "extra": "mean: 488.9561743419865 usec\nrounds: 1520"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1160.72682407953,
            "unit": "iter/sec",
            "range": "stddev: 0.0000477621980059734",
            "extra": "mean: 861.5291550559382 usec\nrounds: 890"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 375.4079472678307,
            "unit": "iter/sec",
            "range": "stddev: 0.000059749186535192044",
            "extra": "mean: 2.663768860723028 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.054125841440086,
            "unit": "iter/sec",
            "range": "stddev: 0.00038706995096365914",
            "extra": "mean: 47.496628809529376 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 447.6497462222442,
            "unit": "iter/sec",
            "range": "stddev: 0.00006478149727334432",
            "extra": "mean: 2.2338893486237588 msec\nrounds: 436"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 384.0069609602266,
            "unit": "iter/sec",
            "range": "stddev: 0.00003388670342187629",
            "extra": "mean: 2.60411946048961 msec\nrounds: 367"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 319.8048964192964,
            "unit": "iter/sec",
            "range": "stddev: 0.00003557214774038723",
            "extra": "mean: 3.126906470778043 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 211.3777664775046,
            "unit": "iter/sec",
            "range": "stddev: 0.00006657679186960665",
            "extra": "mean: 4.730866527092492 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 951.77096111635,
            "unit": "iter/sec",
            "range": "stddev: 0.00001544880064790363",
            "extra": "mean: 1.0506729463852114 msec\nrounds: 858"
          }
        ]
      }
    ]
  }
}