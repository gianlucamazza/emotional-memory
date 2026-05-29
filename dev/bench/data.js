window.BENCHMARK_DATA = {
  "lastUpdate": 1780040782141,
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
          "id": "08cd183486e25a06a2ac33fe3f7f2f6d46262637",
          "message": "docs: surface embedders subpackage; complete CLAUDE.md module table (#34)\n\n* docs: surface embedders subpackage; complete CLAUDE.md module table\n\n- Add docs/api/embedders.md with SentenceTransformerEmbedder API reference\n- Wire it into mkdocs.yml nav between Stores and State Stores\n- Extend CLAUDE.md \"Additional Modules\" table with engine, models,\n  retrieval, decay, and embedders/ (previously omitted)\n\nCloses the documentation drift identified in the gap audit: the\nembedders/ subpackage was importable from the top level via PEP 562\nbut had no API page and was absent from both the nav and CLAUDE.md.\n\nVerified: mkdocs build --strict passes; new page renders.\n\n* fix(coverage): restore omit for optional-dep store adapters\n\nCommit f6ffd9f removed the coverage omit for stores/qdrant.py and\nstores/chroma.py, intending to surface them in coverage measurement.\nHowever, the main CI `test` matrix only installs dev+viz extras, so\nthose files end up in the denominator without their tests running\n(import-skip), dragging coverage from ~86% to 79.46% and failing the\n80% gate.\n\nThe architecturally cleaner fix would be to move the coverage gate to\nthe `extra-tests` job (which installs every extra), but that requires\nediting .github/workflows/ci.yml. Restoring the omit is the minimal\nfix: those adapters are still exercised by `extra-tests` and\n`optional-backends-tests` jobs, just not double-counted in the\nmatrix.\n\nVerified locally with dev+viz extras: coverage 85.67% (gate passes).\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-05-29T09:40:17+02:00",
          "tree_id": "e760ce92ace32bafdd22ba88792d867ecc7570ae",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/08cd183486e25a06a2ac33fe3f7f2f6d46262637"
        },
        "date": 1780040781383,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 567.5036915320342,
            "unit": "iter/sec",
            "range": "stddev: 0.0008701135270499348",
            "extra": "mean: 1.762103075136653 msec\nrounds: 1464"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 394.4368037204146,
            "unit": "iter/sec",
            "range": "stddev: 0.001448564747796002",
            "extra": "mean: 2.535260377753243 msec\nrounds: 1816"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 375.9063972490677,
            "unit": "iter/sec",
            "range": "stddev: 0.0013665548318997577",
            "extra": "mean: 2.6602367166883325 msec\nrounds: 2319"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 173.24916229216979,
            "unit": "iter/sec",
            "range": "stddev: 0.0038331990935530718",
            "extra": "mean: 5.772033681257207 msec\nrounds: 4295"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 435.06399091061115,
            "unit": "iter/sec",
            "range": "stddev: 0.0010658392814206783",
            "extra": "mean: 2.2985124507936154 msec\nrounds: 1890"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 389.97545180479653,
            "unit": "iter/sec",
            "range": "stddev: 0.00025586472354757923",
            "extra": "mean: 2.5642639693653155 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 150214.9771261345,
            "unit": "iter/sec",
            "range": "stddev: 5.959144704308806e-7",
            "extra": "mean: 6.6571258015125 usec\nrounds: 49594"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 34.076047287557834,
            "unit": "iter/sec",
            "range": "stddev: 0.0004050323115086509",
            "extra": "mean: 29.346126666666805 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.842004862616255,
            "unit": "iter/sec",
            "range": "stddev: 0.018371055965301808",
            "extra": "mean: 1.1876415973333299 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03413702716994506,
            "unit": "iter/sec",
            "range": "stddev: 2.2743816017295315",
            "extra": "mean: 29.293704897666675 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3646.937845363469,
            "unit": "iter/sec",
            "range": "stddev: 0.000008908247357229707",
            "extra": "mean: 274.2026440816229 usec\nrounds: 2450"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 929.1536294579064,
            "unit": "iter/sec",
            "range": "stddev: 0.00002931651943171523",
            "extra": "mean: 1.0762482847787262 msec\nrounds: 611"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 376.83371424344455,
            "unit": "iter/sec",
            "range": "stddev: 0.00004404734946590647",
            "extra": "mean: 2.653690373770468 msec\nrounds: 305"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 481.8608038485954,
            "unit": "iter/sec",
            "range": "stddev: 0.0005558700616179614",
            "extra": "mean: 2.0752881164291757 msec\nrounds: 773"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13052.596286625849,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023193349377780997",
            "extra": "mean: 76.61311037595144 usec\nrounds: 8009"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13095.67483947281,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022816086147051494",
            "extra": "mean: 76.36108961607792 usec\nrounds: 8648"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2105.494748927852,
            "unit": "iter/sec",
            "range": "stddev: 0.000013906075621603882",
            "extra": "mean: 474.9477530206211 usec\nrounds: 745"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2107.52398705122,
            "unit": "iter/sec",
            "range": "stddev: 0.000012146736398688286",
            "extra": "mean: 474.4904476267281 usec\nrounds: 1222"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1169.6224810624599,
            "unit": "iter/sec",
            "range": "stddev: 0.000014382701744057437",
            "extra": "mean: 854.9767264148528 usec\nrounds: 848"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 297.3792871926651,
            "unit": "iter/sec",
            "range": "stddev: 0.0006554621909481432",
            "extra": "mean: 3.36270898165185 msec\nrounds: 327"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.957789487294818,
            "unit": "iter/sec",
            "range": "stddev: 0.0003764459710028467",
            "extra": "mean: 45.54192490909062 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 415.47928838687915,
            "unit": "iter/sec",
            "range": "stddev: 0.00026409088066869804",
            "extra": "mean: 2.4068588445950074 msec\nrounds: 444"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 386.32642177425043,
            "unit": "iter/sec",
            "range": "stddev: 0.000031271876081062356",
            "extra": "mean: 2.5884846172503035 msec\nrounds: 371"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 293.9692709503646,
            "unit": "iter/sec",
            "range": "stddev: 0.0003229485698961286",
            "extra": "mean: 3.401716093546545 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 184.38310924075282,
            "unit": "iter/sec",
            "range": "stddev: 0.0005443828177011659",
            "extra": "mean: 5.4234902758597014 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 953.1825335023045,
            "unit": "iter/sec",
            "range": "stddev: 0.000018258241216688063",
            "extra": "mean: 1.0491169999996461 msec\nrounds: 849"
          }
        ]
      }
    ]
  }
}