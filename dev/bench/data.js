window.BENCHMARK_DATA = {
  "lastUpdate": 1779986934035,
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
          "id": "10c7e3c05b82d28f10d07e0187a0a25f65a36575",
          "message": "chore: create issue template for README 'When NOT to use' section",
          "timestamp": "2026-05-28T18:43:02+02:00",
          "tree_id": "475c1b7467b35cd49d90f077fe6ecf2a9fde483a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/10c7e3c05b82d28f10d07e0187a0a25f65a36575"
        },
        "date": 1779986933324,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 697.4064885482017,
            "unit": "iter/sec",
            "range": "stddev: 0.000697930159071082",
            "extra": "mean: 1.4338839922205346 msec\nrounds: 1671"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 405.3258712715357,
            "unit": "iter/sec",
            "range": "stddev: 0.0014180309364169733",
            "extra": "mean: 2.4671506826419196 msec\nrounds: 2483"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 330.9765899425148,
            "unit": "iter/sec",
            "range": "stddev: 0.0018768211030642259",
            "extra": "mean: 3.0213617228145457 msec\nrounds: 3283"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 202.23652205270264,
            "unit": "iter/sec",
            "range": "stddev: 0.003461259518675312",
            "extra": "mean: 4.944705287897509 msec\nrounds: 4842"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 452.89668906625366,
            "unit": "iter/sec",
            "range": "stddev: 0.001081891332313128",
            "extra": "mean: 2.2080090761134077 msec\nrounds: 2470"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 461.9580842607827,
            "unit": "iter/sec",
            "range": "stddev: 0.00026676654777320124",
            "extra": "mean: 2.164698560476937 msec\nrounds: 587"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 178393.53951565898,
            "unit": "iter/sec",
            "range": "stddev: 7.286248267327009e-7",
            "extra": "mean: 5.605584163613853 usec\nrounds: 63436"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 42.73721612337798,
            "unit": "iter/sec",
            "range": "stddev: 0.0004500823298325527",
            "extra": "mean: 23.398810000003323 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.091795017057734,
            "unit": "iter/sec",
            "range": "stddev: 0.003948966504286683",
            "extra": "mean: 915.9228466666652 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.0416804201204577,
            "unit": "iter/sec",
            "range": "stddev: 1.6331492864535992",
            "extra": "mean: 23.99208062466667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4606.28493350123,
            "unit": "iter/sec",
            "range": "stddev: 0.000006988856853988571",
            "extra": "mean: 217.09469006727326 usec\nrounds: 3262"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1177.5721040860974,
            "unit": "iter/sec",
            "range": "stddev: 0.00002119352422613909",
            "extra": "mean: 849.2048992414699 usec\nrounds: 923"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 472.22675476144747,
            "unit": "iter/sec",
            "range": "stddev: 0.00003555648971190968",
            "extra": "mean: 2.1176267331679783 msec\nrounds: 401"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 580.0924021120501,
            "unit": "iter/sec",
            "range": "stddev: 0.0004697528201177595",
            "extra": "mean: 1.723863295501052 msec\nrounds: 978"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 15953.668832311527,
            "unit": "iter/sec",
            "range": "stddev: 0.00000296181824362365",
            "extra": "mean: 62.68150671240366 usec\nrounds: 7598"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 15583.049337641272,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025767642836918855",
            "extra": "mean: 64.17229249120538 usec\nrounds: 7125"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2637.439640051045,
            "unit": "iter/sec",
            "range": "stddev: 0.000010245363032215129",
            "extra": "mean: 379.1555965165694 usec\nrounds: 1378"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2657.1724182368544,
            "unit": "iter/sec",
            "range": "stddev: 0.000011025646101392225",
            "extra": "mean: 376.3398991863471 usec\nrounds: 1597"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1549.2264528802302,
            "unit": "iter/sec",
            "range": "stddev: 0.000011849855293696897",
            "extra": "mean: 645.4834269972987 usec\nrounds: 1089"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 497.24801880987246,
            "unit": "iter/sec",
            "range": "stddev: 0.00008480353463183335",
            "extra": "mean: 2.0110688472795295 msec\nrounds: 478"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.061894902652373,
            "unit": "iter/sec",
            "range": "stddev: 0.0003026754741046492",
            "extra": "mean: 47.47910881817512 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 570.821737045094,
            "unit": "iter/sec",
            "range": "stddev: 0.00010645566174167287",
            "extra": "mean: 1.7518604059764489 msec\nrounds: 569"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 455.192859296832,
            "unit": "iter/sec",
            "range": "stddev: 0.0002642074814336418",
            "extra": "mean: 2.1968710175831174 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 394.4940073739336,
            "unit": "iter/sec",
            "range": "stddev: 0.00019663693174214967",
            "extra": "mean: 2.534892752000966 msec\nrounds: 375"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 215.19458938671264,
            "unit": "iter/sec",
            "range": "stddev: 0.000598596962307244",
            "extra": "mean: 4.646956983676588 msec\nrounds: 245"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1267.5730928278483,
            "unit": "iter/sec",
            "range": "stddev: 0.000013128521220215753",
            "extra": "mean: 788.9091411439514 usec\nrounds: 1084"
          }
        ]
      }
    ]
  }
}