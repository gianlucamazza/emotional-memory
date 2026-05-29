window.BENCHMARK_DATA = {
  "lastUpdate": 1780044284865,
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
          "id": "2c8e6c722caec527e298425e4dd4ab88c689b12b",
          "message": "fix(types): make top-level __all__ a static literal + TYPE_CHECKING re-exports (#37)\n\nbasedpyright flagged reportUnsupportedDunderAll on the computed __all__\nintroduced by f6ffd9f (`list(_CORE_ALL) + [comprehension]`): a non-literal\n__all__ leaves the export surface statically unresolvable.\n\nSwitch to a static literal __all__ (the form type checkers require) and add\nTYPE_CHECKING re-imports of every optional, extra-gated symbol so\n`from emotional_memory import SQLiteStore` now type-checks — previously those\nnames were only injected via globals() and invisible to mypy/pyright.\n\nRuntime contract is preserved exactly: a `if not TYPE_CHECKING:` branch filters\nout optional names whose extra did not resolve, so `import *` never raises and\nevery advertised name is importable; __getattr__ still raises ImportError with\nan install hint for missing extras.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-29T10:38:27+02:00",
          "tree_id": "394c25be7407f9afe32274ad8cfc1083072f629e",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/2c8e6c722caec527e298425e4dd4ab88c689b12b"
        },
        "date": 1780044283385,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 638.7807001250123,
            "unit": "iter/sec",
            "range": "stddev: 0.0006883798148850699",
            "extra": "mean: 1.565482488441331 msec\nrounds: 1341"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 421.64108314305065,
            "unit": "iter/sec",
            "range": "stddev: 0.0011693945576939947",
            "extra": "mean: 2.371685397793006 msec\nrounds: 1903"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 345.9586917507986,
            "unit": "iter/sec",
            "range": "stddev: 0.001549877926797029",
            "extra": "mean: 2.8905185036377734 msec\nrounds: 2474"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 166.7813257239102,
            "unit": "iter/sec",
            "range": "stddev: 0.003994984553851255",
            "extra": "mean: 5.995875111674073 msec\nrounds: 4540"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 422.1703818522944,
            "unit": "iter/sec",
            "range": "stddev: 0.0010987033620100132",
            "extra": "mean: 2.3687118826584856 msec\nrounds: 1926"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 379.31425045041755,
            "unit": "iter/sec",
            "range": "stddev: 0.00024547231730558724",
            "extra": "mean: 2.6363364909505713 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 150600.87228666493,
            "unit": "iter/sec",
            "range": "stddev: 6.265990878241436e-7",
            "extra": "mean: 6.640067781921777 usec\nrounds: 40291"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.150892552971996,
            "unit": "iter/sec",
            "range": "stddev: 0.0002686376682697419",
            "extra": "mean: 30.16510033333475 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8216130111764833,
            "unit": "iter/sec",
            "range": "stddev: 0.007368242479830437",
            "extra": "mean: 1.2171180183333281 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.031014392053607915,
            "unit": "iter/sec",
            "range": "stddev: 0.7990096861797668",
            "extra": "mean: 32.243095343333344 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3703.4030249232455,
            "unit": "iter/sec",
            "range": "stddev: 0.00000764381458046791",
            "extra": "mean: 270.0219212627352 usec\nrounds: 2629"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 919.737085231445,
            "unit": "iter/sec",
            "range": "stddev: 0.00002777244603901665",
            "extra": "mean: 1.0872672376240624 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 370.2722786175642,
            "unit": "iter/sec",
            "range": "stddev: 0.00012708488916956886",
            "extra": "mean: 2.7007152783177975 msec\nrounds: 309"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 500.1206435887476,
            "unit": "iter/sec",
            "range": "stddev: 0.00042129592564818156",
            "extra": "mean: 1.999517542055925 msec\nrounds: 749"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13475.741224328729,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021098853113657244",
            "extra": "mean: 74.20742082777814 usec\nrounds: 7951"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13346.399440042745,
            "unit": "iter/sec",
            "range": "stddev: 0.000002429088820225461",
            "extra": "mean: 74.92657510307492 usec\nrounds: 8748"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2228.602860311687,
            "unit": "iter/sec",
            "range": "stddev: 0.000010232430458447483",
            "extra": "mean: 448.71162009553484 usec\nrounds: 1045"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2223.004937744627,
            "unit": "iter/sec",
            "range": "stddev: 0.000009417812474461715",
            "extra": "mean: 449.84155591420347 usec\nrounds: 1547"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1164.6661845563094,
            "unit": "iter/sec",
            "range": "stddev: 0.000018893968387094926",
            "extra": "mean: 858.6151235952296 usec\nrounds: 890"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.7519072708394,
            "unit": "iter/sec",
            "range": "stddev: 0.00003875078014190874",
            "extra": "mean: 2.6126584374990176 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.1447150930812,
            "unit": "iter/sec",
            "range": "stddev: 0.0003251171753934487",
            "extra": "mean: 45.1575012727274 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 468.35500599671667,
            "unit": "iter/sec",
            "range": "stddev: 0.00003106160037522426",
            "extra": "mean: 2.135132511014541 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 388.1754805186166,
            "unit": "iter/sec",
            "range": "stddev: 0.00004762204183802397",
            "extra": "mean: 2.576154471848566 msec\nrounds: 373"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 322.3675914473404,
            "unit": "iter/sec",
            "range": "stddev: 0.000036655439732334896",
            "extra": "mean: 3.1020487993544252 msec\nrounds: 309"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 213.41841975237483,
            "unit": "iter/sec",
            "range": "stddev: 0.00005006151456931425",
            "extra": "mean: 4.685631170731562 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 956.1436556756702,
            "unit": "iter/sec",
            "range": "stddev: 0.00002071043208995824",
            "extra": "mean: 1.0458679447005674 msec\nrounds: 868"
          }
        ]
      }
    ]
  }
}