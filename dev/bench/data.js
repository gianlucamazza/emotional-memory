window.BENCHMARK_DATA = {
  "lastUpdate": 1782983037384,
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
          "id": "f438a39d53dd294bd9b8b001d31cac05eaf2bc60",
          "message": "fix(bench): restore scored dailydialog artifacts clobbered by dry-run (#98)\n\nPR #96's verification smoke (make bench-dailydialog-dry) overwrote the\ncommitted SCORED Hk1 artifacts (results.json/md/protocol.json, N=120/396)\nwith 5-persona dry output, and the madialbench results.dry.* files were\ncommitted before the gitignore entry could take effect. This restores and\nhardens:\n\n- benchmarks/dailydialog/results.* restored byte-identical to pre-#96\n  (0c39395)\n- madialbench results.dry.* untracked and removed\n- dailydialog/runner.py + t2a_runner.py: same dry-run output guard as\n  madialbench (smoke writes *.dry.*, never the committed default paths)\n- .gitignore generalized to benchmarks/**/*.dry.{json,md}\n\nVerified: git diff vs 0c39395 on the restored artifacts is empty;\nbench-dailydialog-dry now writes results.dry.json only.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T10:58:39+02:00",
          "tree_id": "be568f9b61de56f2b8cee23365e9a7ca165ab1fd",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/f438a39d53dd294bd9b8b001d31cac05eaf2bc60"
        },
        "date": 1782983036810,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 452.8682295409488,
            "unit": "iter/sec",
            "range": "stddev: 0.0011711146329858228",
            "extra": "mean: 2.208147833672618 msec\nrounds: 1966"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 444.95832610144487,
            "unit": "iter/sec",
            "range": "stddev: 0.001012013868208399",
            "extra": "mean: 2.2474014786095107 msec\nrounds: 1870"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 383.08692451226216,
            "unit": "iter/sec",
            "range": "stddev: 0.0015096088062853093",
            "extra": "mean: 2.6103736150044226 msec\nrounds: 2226"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 235.09845435685455,
            "unit": "iter/sec",
            "range": "stddev: 0.0025290296108385173",
            "extra": "mean: 4.2535371095298915 msec\nrounds: 3725"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 464.4830298092635,
            "unit": "iter/sec",
            "range": "stddev: 0.0009420042090039773",
            "extra": "mean: 2.1529311854744027 msec\nrounds: 1790"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 388.02805873527234,
            "unit": "iter/sec",
            "range": "stddev: 0.0002585156541719508",
            "extra": "mean: 2.5771332188176586 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138595.1134609822,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011636180920139912",
            "extra": "mean: 7.215261599259224 usec\nrounds: 49809"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.171372864682592,
            "unit": "iter/sec",
            "range": "stddev: 0.000301994903561718",
            "extra": "mean: 36.80344033332972 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8153748836219539,
            "unit": "iter/sec",
            "range": "stddev: 0.005782605011444224",
            "extra": "mean: 1.226429731999995 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03790722965932108,
            "unit": "iter/sec",
            "range": "stddev: 1.2246844015391007",
            "extra": "mean: 26.38019208966668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3154.228423325261,
            "unit": "iter/sec",
            "range": "stddev: 0.000011651534581747847",
            "extra": "mean: 317.0347437760315 usec\nrounds: 2611"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 865.6618724699698,
            "unit": "iter/sec",
            "range": "stddev: 0.00001673975031776616",
            "extra": "mean: 1.1551854503499466 msec\nrounds: 715"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.45551681075256,
            "unit": "iter/sec",
            "range": "stddev: 0.00005038894626754218",
            "extra": "mean: 2.7897464346404592 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.1652197783895,
            "unit": "iter/sec",
            "range": "stddev: 0.00042205504804497014",
            "extra": "mean: 1.9953499575293565 msec\nrounds: 777"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10595.254806977047,
            "unit": "iter/sec",
            "range": "stddev: 0.000005354946588586492",
            "extra": "mean: 94.38187360453976 usec\nrounds: 6899"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9963.86598232541,
            "unit": "iter/sec",
            "range": "stddev: 0.000010617031138289838",
            "extra": "mean: 100.36265057898899 usec\nrounds: 7169"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1802.5904267254552,
            "unit": "iter/sec",
            "range": "stddev: 0.00000916460787433697",
            "extra": "mean: 554.7571900825955 usec\nrounds: 1089"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1788.037611461207,
            "unit": "iter/sec",
            "range": "stddev: 0.000009681914941097885",
            "extra": "mean: 559.2723517615423 usec\nrounds: 1646"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1002.280219791531,
            "unit": "iter/sec",
            "range": "stddev: 0.00002121324159307893",
            "extra": "mean: 997.7249677819589 usec\nrounds: 807"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 373.3011312477617,
            "unit": "iter/sec",
            "range": "stddev: 0.000026503510844217466",
            "extra": "mean: 2.678802490250948 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.75749155653434,
            "unit": "iter/sec",
            "range": "stddev: 0.0009016240978689566",
            "extra": "mean: 40.391814240003896 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 475.0310509248668,
            "unit": "iter/sec",
            "range": "stddev: 0.00003083531822612049",
            "extra": "mean: 2.1051255450628736 msec\nrounds: 466"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 379.45744630420506,
            "unit": "iter/sec",
            "range": "stddev: 0.00008137140116152006",
            "extra": "mean: 2.6353416166679087 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.7666375175002,
            "unit": "iter/sec",
            "range": "stddev: 0.00006926211123641686",
            "extra": "mean: 3.259806894558254 msec\nrounds: 294"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.8730108442584,
            "unit": "iter/sec",
            "range": "stddev: 0.00005480638133609016",
            "extra": "mean: 5.079416400001504 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 855.258468183915,
            "unit": "iter/sec",
            "range": "stddev: 0.00001593742908202833",
            "extra": "mean: 1.1692371805723645 msec\nrounds: 803"
          }
        ]
      }
    ]
  }
}