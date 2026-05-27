window.BENCHMARK_DATA = {
  "lastUpdate": 1779919420890,
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
          "id": "f6ffd9f0e817bf0f2ef94f88d069367fca07f15a",
          "message": "refactor: align optional exports, CI coverage, and release tooling with best practices\n\n- PEP 562 __getattr__/__dir__ on __init__: dynamic __all__ (only available optionals),\n  clear ImportError with extra hint for unavailable names; RedisAffectiveStateStore\n  moved to optional (was hard import)\n- Remove stale mypy overrides for opentelemetry/* (only dynamic string imports)\n- Add F401 per-file-ignore for __init__.py (re-export module pattern)\n- CI: extra-tests now installs qdrant/chroma/redis/mem0/otel; new matrix job\n  optional-backends-tests (qdrant/chroma/redis) for isolated failure attribution;\n  remove coverage.omit for stores/qdrant.py and stores/chroma.py\n- Makefile: docs/docs-serve/check-all depend on install-docs (idempotent auto-install)\n- .gitignore: add .release_state*.bak pattern\n- release-gate skill: use scripts/resolve_version.py + check_release_metadata.py +\n  check_metadata_ssot.py as canonical version sources instead of grep __version__\n\nCo-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>",
          "timestamp": "2026-05-27T23:54:16+02:00",
          "tree_id": "8991b00d15d3157b70326c9c82d5f1e019e0efdd",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/f6ffd9f0e817bf0f2ef94f88d069367fca07f15a"
        },
        "date": 1779919419757,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 538.2250644320999,
            "unit": "iter/sec",
            "range": "stddev: 0.000869498887721142",
            "extra": "mean: 1.8579588095830044 msec\nrounds: 1607"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 456.23978213044836,
            "unit": "iter/sec",
            "range": "stddev: 0.0010152405914770307",
            "extra": "mean: 2.191829908673066 msec\nrounds: 1741"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 344.4121155146576,
            "unit": "iter/sec",
            "range": "stddev: 0.0016963731276842503",
            "extra": "mean: 2.903498323529335 msec\nrounds: 2346"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 179.3880248490417,
            "unit": "iter/sec",
            "range": "stddev: 0.0039040175409086127",
            "extra": "mean: 5.57450811357959 msec\nrounds: 3601"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 473.27066883182994,
            "unit": "iter/sec",
            "range": "stddev: 0.0010402977127587192",
            "extra": "mean: 2.1129557901153935 msec\nrounds: 1558"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 362.76626759307953,
            "unit": "iter/sec",
            "range": "stddev: 0.00037575434116183116",
            "extra": "mean: 2.756595883721238 msec\nrounds: 430"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135164.13572949576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011544101447708598",
            "extra": "mean: 7.398412268187043 usec\nrounds: 35751"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.79262985036004,
            "unit": "iter/sec",
            "range": "stddev: 0.0000390693195216383",
            "extra": "mean: 35.98076200000359 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7973047413006922,
            "unit": "iter/sec",
            "range": "stddev: 0.005099174644594253",
            "extra": "mean: 1.2542255779999987 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02567799113299455,
            "unit": "iter/sec",
            "range": "stddev: 4.661459079274076",
            "extra": "mean: 38.943856426333326 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3254.0921728476987,
            "unit": "iter/sec",
            "range": "stddev: 0.00001495220204553686",
            "extra": "mean: 307.3053702485898 usec\nrounds: 2131"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 864.2649559789382,
            "unit": "iter/sec",
            "range": "stddev: 0.00020912545974384442",
            "extra": "mean: 1.1570525833334488 msec\nrounds: 732"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 364.3712278696008,
            "unit": "iter/sec",
            "range": "stddev: 0.00007742925255991498",
            "extra": "mean: 2.7444537974273717 msec\nrounds: 311"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 488.42585218772433,
            "unit": "iter/sec",
            "range": "stddev: 0.0009206980271359732",
            "extra": "mean: 2.0473936740261944 msec\nrounds: 770"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10571.401174853381,
            "unit": "iter/sec",
            "range": "stddev: 0.000005576454933470876",
            "extra": "mean: 94.594839743547 usec\nrounds: 6552"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10406.844954423357,
            "unit": "iter/sec",
            "range": "stddev: 0.00001280799509636357",
            "extra": "mean: 96.09060232755336 usec\nrounds: 8077"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1826.407774027784,
            "unit": "iter/sec",
            "range": "stddev: 0.000015540504080214334",
            "extra": "mean: 547.5228556406636 usec\nrounds: 1046"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1839.1374846386452,
            "unit": "iter/sec",
            "range": "stddev: 0.000009353160372444422",
            "extra": "mean: 543.7331403184795 usec\nrounds: 1632"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1017.2889778234232,
            "unit": "iter/sec",
            "range": "stddev: 0.000045942495380652624",
            "extra": "mean: 983.0048509319206 usec\nrounds: 805"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.3606950996196,
            "unit": "iter/sec",
            "range": "stddev: 0.00010573261149499689",
            "extra": "mean: 2.707380653294179 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.54623565231744,
            "unit": "iter/sec",
            "range": "stddev: 0.0009953607633190423",
            "extra": "mean: 46.41181950000828 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 464.6548693553938,
            "unit": "iter/sec",
            "range": "stddev: 0.000031956370541109525",
            "extra": "mean: 2.152134984375133 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.61569758953283,
            "unit": "iter/sec",
            "range": "stddev: 0.00008966048008159545",
            "extra": "mean: 2.6837301983492474 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.40777821130996,
            "unit": "iter/sec",
            "range": "stddev: 0.00004791985909868251",
            "extra": "mean: 3.3067932508708213 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.97479463350012,
            "unit": "iter/sec",
            "range": "stddev: 0.000056609618123389116",
            "extra": "mean: 5.1288680769210675 msec\nrounds: 182"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 851.9532354305588,
            "unit": "iter/sec",
            "range": "stddev: 0.000046769402568911975",
            "extra": "mean: 1.1737733462501865 msec\nrounds: 800"
          }
        ]
      }
    ]
  }
}