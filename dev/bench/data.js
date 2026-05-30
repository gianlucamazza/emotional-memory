window.BENCHMARK_DATA = {
  "lastUpdate": 1780174711158,
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
          "id": "4a6766cf4c1ffa85782ab0867e8c8b4e93e97339",
          "message": "fix(docs): use absolute README image URLs for PyPI rendering (#42)\n\nThe 13 doc/research images used repo-relative paths (docs/images/...). PyPI\nrenders the long_description in isolation, so relative image links resolve\nagainst pypi.org and 404 — the images were missing on the project page. Switch\nto absolute raw.githubusercontent.com/.../main/docs/images/... URLs, which\nrender on both PyPI and GitHub. README is the SSOT for docs/index.md (mkdocs\ninclude), so this also keeps the docs site consistent.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T22:52:09+02:00",
          "tree_id": "79b16572d689e1bd288e99fa42001608f2e49d36",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/4a6766cf4c1ffa85782ab0867e8c8b4e93e97339"
        },
        "date": 1780174709902,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 615.11289758028,
            "unit": "iter/sec",
            "range": "stddev: 0.0007623455717962587",
            "extra": "mean: 1.6257178217751276 msec\nrounds: 1442"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 417.5998161902459,
            "unit": "iter/sec",
            "range": "stddev: 0.001259298862351123",
            "extra": "mean: 2.394637069343034 msec\nrounds: 1918"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 313.58903100881037,
            "unit": "iter/sec",
            "range": "stddev: 0.0019141334480387879",
            "extra": "mean: 3.1888870499807265 msec\nrounds: 2601"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 181.30352929546754,
            "unit": "iter/sec",
            "range": "stddev: 0.003718298469664248",
            "extra": "mean: 5.515612431186132 msec\nrounds: 4316"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 440.8202478437978,
            "unit": "iter/sec",
            "range": "stddev: 0.001047048135976721",
            "extra": "mean: 2.2684983389291693 msec\nrounds: 1906"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 386.09100621653613,
            "unit": "iter/sec",
            "range": "stddev: 0.0002780351828506164",
            "extra": "mean: 2.5900629227274923 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135880.21962916505,
            "unit": "iter/sec",
            "range": "stddev: 8.417236259786977e-7",
            "extra": "mean: 7.359422900030124 usec\nrounds: 43930"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.12332446353101,
            "unit": "iter/sec",
            "range": "stddev: 0.0006595122202728445",
            "extra": "mean: 30.190206333334874 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8487789428781425,
            "unit": "iter/sec",
            "range": "stddev: 0.0053865241730103805",
            "extra": "mean: 1.1781630640000078 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03072744178467215,
            "unit": "iter/sec",
            "range": "stddev: 1.4482505137374662",
            "extra": "mean: 32.54419964433331 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3495.4544736404455,
            "unit": "iter/sec",
            "range": "stddev: 0.00001232063177633744",
            "extra": "mean: 286.0858316253566 usec\nrounds: 2441"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 858.1610399154679,
            "unit": "iter/sec",
            "range": "stddev: 0.00016389525175116492",
            "extra": "mean: 1.1652824510636182 msec\nrounds: 705"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.63911478004087,
            "unit": "iter/sec",
            "range": "stddev: 0.00016976441222869363",
            "extra": "mean: 2.7883182809363003 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 495.5674092871687,
            "unit": "iter/sec",
            "range": "stddev: 0.00045279190079649975",
            "extra": "mean: 2.017888951653246 msec\nrounds: 786"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12611.32002708962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036816264421301437",
            "extra": "mean: 79.29384060129787 usec\nrounds: 7522"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11839.263451245966,
            "unit": "iter/sec",
            "range": "stddev: 0.000014554044780107236",
            "extra": "mean: 84.46471388342657 usec\nrounds: 8521"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2041.7813217948794,
            "unit": "iter/sec",
            "range": "stddev: 0.000014823896662503399",
            "extra": "mean: 489.7684141418851 usec\nrounds: 1089"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2032.2520500591306,
            "unit": "iter/sec",
            "range": "stddev: 0.00006992197507722786",
            "extra": "mean: 492.0649483271054 usec\nrounds: 1645"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1195.7209306566488,
            "unit": "iter/sec",
            "range": "stddev: 0.000019071949846284496",
            "extra": "mean: 836.3155435029763 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 396.018200789261,
            "unit": "iter/sec",
            "range": "stddev: 0.00006503553740853119",
            "extra": "mean: 2.5251364659679987 msec\nrounds: 382"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.017767811394215,
            "unit": "iter/sec",
            "range": "stddev: 0.0017493439025750824",
            "extra": "mean: 47.578791857138945 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 472.9565481793899,
            "unit": "iter/sec",
            "range": "stddev: 0.00004919419869689382",
            "extra": "mean: 2.114359139014828 msec\nrounds: 446"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 391.5642763312649,
            "unit": "iter/sec",
            "range": "stddev: 0.000051992550376419786",
            "extra": "mean: 2.553859124661301 msec\nrounds: 369"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 325.24360579769177,
            "unit": "iter/sec",
            "range": "stddev: 0.00005020169822728838",
            "extra": "mean: 3.0746184772715277 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 212.43247704186123,
            "unit": "iter/sec",
            "range": "stddev: 0.0000908861418470597",
            "extra": "mean: 4.707378146341265 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 981.4823435531702,
            "unit": "iter/sec",
            "range": "stddev: 0.00001841941373706689",
            "extra": "mean: 1.0188670296194957 msec\nrounds: 844"
          }
        ]
      }
    ]
  }
}