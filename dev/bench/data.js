window.BENCHMARK_DATA = {
  "lastUpdate": 1782559765176,
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
          "id": "1062997e7bf1e19a50561ec9417a7b667fde1e78",
          "message": "chore(polish): post-v0.13.0 hygiene (security lock + docs consistency) (#80)\n\n* chore(deps): refresh lock to clear transitive security advisories\n\nuv.lock bumps patched transitive versions of optional/dev dependencies:\ncryptography 49.0.0, langsmith 0.9.3, starlette 1.3.1, pydantic-settings 2.14.2,\ngradio 6.19.0 (PYSEC-2026-211). Full-lock pip-audit (uv export --all-extras) now\nreports only chromadb <=1.5.9 (CRITICAL, no upstream fix, optional [chroma] extra,\nnot in the runtime wheel). Runtime wheel deps unchanged (numpy + pydantic only).\n\nSECURITY.md: supported-versions table -> 0.13.x; advisory note updated.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: fix stale claims (T2A regime-bound caveat, m_fr deferral)\n\nCLAUDE.md: the retrieve-time query-appraisal path is production-reachable but\nbounded to the affect-discriminative regime (Addendum T2A: FAIL on naturalistic\ndialogue). benchmarks/preregistration_addendum_m_fr.md: \"deferred to v0.12\"\n(stale; FR keyword appraisal still not implemented) -> \"deferred (not yet\nimplemented as of v0.13.0)\".\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: complete v0.13.0 docs (index T/T2A/U/V, query-appraisal tutorial, help)\n\n- docs/research/index.md: add Addenda U, V, T, T2A entries (were missing).\n- New tutorial docs/tutorials/query_appraisal_retrieval.md covering the three\n  query_affect usage patterns + the T2A regime-bound caveat; added to mkdocs nav.\n- Makefile help: document bench-t2a-dailydialog[-dry] targets.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* test(bench): add keyless smoke tests for the T2A runner\n\nCovers the runner's pure logic without an LLM key or model download: _pearson,\n_verdict (PASS/FAIL), _make_adapter rejection, and a write_results round-trip\n(JSON/MD/protocol) on a synthetic results dict. The full 3-arm benchmark (which\nneeds an API key for the aft_query_appraised arm) stays covered by CI.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(changelog): record post-v0.13.0 polish (security lock refresh + docs)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T13:24:05+02:00",
          "tree_id": "4f8a573344c31ed0abd18bfde08b569a3b3dca70",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1062997e7bf1e19a50561ec9417a7b667fde1e78"
        },
        "date": 1782559764644,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 572.7617339648734,
            "unit": "iter/sec",
            "range": "stddev: 0.0007933240547705615",
            "extra": "mean: 1.745926692898322 msec\nrounds: 1563"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.5771966531247,
            "unit": "iter/sec",
            "range": "stddev: 0.0010590379192975264",
            "extra": "mean: 2.2292706973539693 msec\nrounds: 1814"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 358.78587470542993,
            "unit": "iter/sec",
            "range": "stddev: 0.0015521852261384135",
            "extra": "mean: 2.7871777305085357 msec\nrounds: 2360"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 237.12511190116211,
            "unit": "iter/sec",
            "range": "stddev: 0.002546727715551782",
            "extra": "mean: 4.217183038871132 msec\nrounds: 3756"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 447.9819889094801,
            "unit": "iter/sec",
            "range": "stddev: 0.0009803577026081213",
            "extra": "mean: 2.232232600320147 msec\nrounds: 1874"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.55386082062046,
            "unit": "iter/sec",
            "range": "stddev: 0.0003129979038158573",
            "extra": "mean: 2.6004159673915264 msec\nrounds: 460"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138540.67248519266,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013393112239071015",
            "extra": "mean: 7.218096910182681 usec\nrounds: 51718"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.19632374639044,
            "unit": "iter/sec",
            "range": "stddev: 0.000669574490820501",
            "extra": "mean: 35.46561633333548 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8229514169360885,
            "unit": "iter/sec",
            "range": "stddev: 0.0033682375630206575",
            "extra": "mean: 1.2151385603333391 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.038823913458373396,
            "unit": "iter/sec",
            "range": "stddev: 0.8247654734576748",
            "extra": "mean: 25.757320963333328 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 2878.6919152163073,
            "unit": "iter/sec",
            "range": "stddev: 0.000012784304447499425",
            "extra": "mean: 347.38000086572623 usec\nrounds: 2311"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 758.7585980927298,
            "unit": "iter/sec",
            "range": "stddev: 0.00001625090312516743",
            "extra": "mean: 1.317942231578887 msec\nrounds: 665"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 310.54449895830913,
            "unit": "iter/sec",
            "range": "stddev: 0.000029636261597980595",
            "extra": "mean: 3.2201504240274783 msec\nrounds: 283"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 505.774289200243,
            "unit": "iter/sec",
            "range": "stddev: 0.00041439632714443614",
            "extra": "mean: 1.9771665372339364 msec\nrounds: 752"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10427.631322504909,
            "unit": "iter/sec",
            "range": "stddev: 0.000004693530739731444",
            "extra": "mean: 95.89905598616633 usec\nrounds: 7484"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10370.744921046748,
            "unit": "iter/sec",
            "range": "stddev: 0.000005075692148651511",
            "extra": "mean: 96.42508880635617 usec\nrounds: 8344"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1818.5147882797053,
            "unit": "iter/sec",
            "range": "stddev: 0.000009011449601690476",
            "extra": "mean: 549.8992949878559 usec\nrounds: 1217"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1804.5195516478807,
            "unit": "iter/sec",
            "range": "stddev: 0.000010501209969157942",
            "extra": "mean: 554.1641258953407 usec\nrounds: 1676"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1001.2796013058748,
            "unit": "iter/sec",
            "range": "stddev: 0.000036518724151552245",
            "extra": "mean: 998.7220339811118 usec\nrounds: 824"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 378.25466437870944,
            "unit": "iter/sec",
            "range": "stddev: 0.000028220745206151423",
            "extra": "mean: 2.6437215298918235 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.699820086835132,
            "unit": "iter/sec",
            "range": "stddev: 0.01037659957700597",
            "extra": "mean: 44.05321258823345 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 476.6822120677621,
            "unit": "iter/sec",
            "range": "stddev: 0.000025352615859997714",
            "extra": "mean: 2.097833681819548 msec\nrounds: 462"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 381.0277387253983,
            "unit": "iter/sec",
            "range": "stddev: 0.000036009519220816204",
            "extra": "mean: 2.624480840542391 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 307.97791914096746,
            "unit": "iter/sec",
            "range": "stddev: 0.000023891841675280516",
            "extra": "mean: 3.246986026755641 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.43703425418775,
            "unit": "iter/sec",
            "range": "stddev: 0.00010579734555923211",
            "extra": "mean: 5.116737489473914 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 849.9447871786799,
            "unit": "iter/sec",
            "range": "stddev: 0.000015833407938639285",
            "extra": "mean: 1.1765470123293724 msec\nrounds: 811"
          }
        ]
      }
    ]
  }
}