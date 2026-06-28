window.BENCHMARK_DATA = {
  "lastUpdate": 1782645958263,
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
          "id": "374eb18f1be3fa332a43be12d2014182f1c1adec",
          "message": "chore(release): v0.14.0 (#88)\n\nPromote CHANGELOG [Unreleased] -> [0.14.0] and propagate the version across SSOT\nmetadata (pyproject, CITATION.cff, .zenodo.json, codemeta.json, README, demo,\npaper/SUBMISSION.md) via sync_release_metadata.\n\nRelease content is research + docs + benchmark tooling — no src/ change, so the\nPyPI wheel is code-identical to 0.13.0; the value is the citable Zenodo snapshot\nof Addendum W (arousal calibration adopted) and the V/T-led paper reframe.\n\nPre-flight: meta-check OK, sync-metadata idempotent, reproduce-paper-check clean,\nclaim-matrix refs all present. Preflight G5/G6 (clean tree / on main) clear on\nmerge. NOT published — `make release` is the gated publish step.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-28T13:19:18+02:00",
          "tree_id": "b1eed49f483aaf65b5ce6fa2cea14b6b7b557e6c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/374eb18f1be3fa332a43be12d2014182f1c1adec"
        },
        "date": 1782645957682,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 610.3054198550909,
            "unit": "iter/sec",
            "range": "stddev: 0.0007427704397805732",
            "extra": "mean: 1.6385238725840532 msec\nrounds: 1397"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 468.43661001652544,
            "unit": "iter/sec",
            "range": "stddev: 0.0009959556081375724",
            "extra": "mean: 2.1347605601635666 msec\nrounds: 1712"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 311.3900727657831,
            "unit": "iter/sec",
            "range": "stddev: 0.0019037650596450038",
            "extra": "mean: 3.2114061669273757 msec\nrounds: 2558"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 170.39249461208024,
            "unit": "iter/sec",
            "range": "stddev: 0.004102225578014218",
            "extra": "mean: 5.86880309650155 msec\nrounds: 4259"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 429.59124269152323,
            "unit": "iter/sec",
            "range": "stddev: 0.001076227953096611",
            "extra": "mean: 2.327794192764936 msec\nrounds: 1935"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.79120620574287,
            "unit": "iter/sec",
            "range": "stddev: 0.00028876514161486406",
            "extra": "mean: 2.619232668918811 msec\nrounds: 444"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135055.4309521975,
            "unit": "iter/sec",
            "range": "stddev: 9.609859028202136e-7",
            "extra": "mean: 7.404367176866418 usec\nrounds: 47168"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.66215045660394,
            "unit": "iter/sec",
            "range": "stddev: 0.0009193464082506527",
            "extra": "mean: 30.616477666669084 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8316434977362792,
            "unit": "iter/sec",
            "range": "stddev: 0.01968905106550496",
            "extra": "mean: 1.202438307666668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.029515612914081498,
            "unit": "iter/sec",
            "range": "stddev: 1.1138577262980898",
            "extra": "mean: 33.88037385199999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3319.278336373304,
            "unit": "iter/sec",
            "range": "stddev: 0.000011335758862413601",
            "extra": "mean: 301.2703059703682 usec\nrounds: 2278"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 856.641290114377,
            "unit": "iter/sec",
            "range": "stddev: 0.000019175351754929856",
            "extra": "mean: 1.1673497548390204 msec\nrounds: 620"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 342.51907418764176,
            "unit": "iter/sec",
            "range": "stddev: 0.00003460436607860789",
            "extra": "mean: 2.9195454366204765 msec\nrounds: 284"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 503.78292384705975,
            "unit": "iter/sec",
            "range": "stddev: 0.0004441878521977613",
            "extra": "mean: 1.984981929049234 msec\nrounds: 747"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 11811.178064435648,
            "unit": "iter/sec",
            "range": "stddev: 0.000010784526399216565",
            "extra": "mean: 84.66555956946206 usec\nrounds: 6228"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11945.393535916159,
            "unit": "iter/sec",
            "range": "stddev: 0.000003509574321081286",
            "extra": "mean: 83.71427839470543 usec\nrounds: 8197"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1985.6623233690275,
            "unit": "iter/sec",
            "range": "stddev: 0.000011594090888369995",
            "extra": "mean: 503.61030082059625 usec\nrounds: 1097"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1970.860923595767,
            "unit": "iter/sec",
            "range": "stddev: 0.000013022063913434099",
            "extra": "mean: 507.39247403390334 usec\nrounds: 1656"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1170.9252387846389,
            "unit": "iter/sec",
            "range": "stddev: 0.000019942440415713413",
            "extra": "mean: 854.0254893112983 usec\nrounds: 842"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 383.38560645631713,
            "unit": "iter/sec",
            "range": "stddev: 0.00004742263634917808",
            "extra": "mean: 2.6083399667586105 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.659644527360406,
            "unit": "iter/sec",
            "range": "stddev: 0.00199362209957258",
            "extra": "mean: 50.86561960000324 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.13081976173333,
            "unit": "iter/sec",
            "range": "stddev: 0.00006760134120669048",
            "extra": "mean: 2.127067526665895 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 384.0867933727663,
            "unit": "iter/sec",
            "range": "stddev: 0.00005617536385793759",
            "extra": "mean: 2.6035781944459457 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 317.01750599600405,
            "unit": "iter/sec",
            "range": "stddev: 0.00006200027437635362",
            "extra": "mean: 3.1543999340295255 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.03866212691827,
            "unit": "iter/sec",
            "range": "stddev: 0.0001377071649251406",
            "extra": "mean: 4.8067988410247 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 957.4118186536144,
            "unit": "iter/sec",
            "range": "stddev: 0.00001696282989685607",
            "extra": "mean: 1.0444826150216906 msec\nrounds: 852"
          }
        ]
      }
    ]
  }
}