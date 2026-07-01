window.BENCHMARK_DATA = {
  "lastUpdate": 1782946611964,
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
          "id": "b9c471a14c2e2abeda5d730137f62add38f9d8c3",
          "message": "feat(bench): Addendum X harness — MADial-Bench third-party retrieval (#92)\n\n- benchmarks/madialbench/: pinned loader (sha256-gated, fails on tamper),\n  metrics replicated verbatim from the benchmark's embedding_score_new.py,\n  adapters (naive_cosine, aft_query_appraised with DecayConfig zeroed per\n  Amendment A1), runner with Hx1 paired bootstrap + MDE + D1/D2 diagnostics\n- benchmarks/datasets/madialbench/: vendored EN split (MIT, byte-identical\n  to upstream commit 572e3a1, hashes match the pre-registration pins)\n- tests/test_madialbench.py: hand-computed metric cases, loader integrity,\n  protocol-exact query construction, tamper rejection (8 tests)\n- Makefile: bench-x-madial (LLM) + bench-x-madial-dry (no-LLM smoke)\n- datasets README: MADial-Bench source/citation/license section\n\nSmoke run green (10 queries, keyword stub, no scored verdict).\nNo scored run executed — pre-registration integrity preserved.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T00:50:32+02:00",
          "tree_id": "90252e48c223fb61d535de31991c1a7a5673e88e",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/b9c471a14c2e2abeda5d730137f62add38f9d8c3"
        },
        "date": 1782946610867,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 596.3970680731702,
            "unit": "iter/sec",
            "range": "stddev: 0.0007946384001721215",
            "extra": "mean: 1.6767352717389832 msec\nrounds: 1472"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 413.25797603383006,
            "unit": "iter/sec",
            "range": "stddev: 0.001367534091480117",
            "extra": "mean: 2.4197960063525503 msec\nrounds: 1889"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 325.48960177322294,
            "unit": "iter/sec",
            "range": "stddev: 0.0018667643214295335",
            "extra": "mean: 3.072294766260232 msec\nrounds: 2460"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 164.6998325440246,
            "unit": "iter/sec",
            "range": "stddev: 0.004180703600999278",
            "extra": "mean: 6.071651589158101 msec\nrounds: 4335"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 452.5548593178122,
            "unit": "iter/sec",
            "range": "stddev: 0.0010164883595735735",
            "extra": "mean: 2.2096768588617404 msec\nrounds: 1828"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.8235560427128,
            "unit": "iter/sec",
            "range": "stddev: 0.00024244330927228658",
            "extra": "mean: 2.6190107555546795 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134729.87453147428,
            "unit": "iter/sec",
            "range": "stddev: 8.212637294628074e-7",
            "extra": "mean: 7.422258823275232 usec\nrounds: 48168"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.04014863749206,
            "unit": "iter/sec",
            "range": "stddev: 0.0004492103137597002",
            "extra": "mean: 30.266207666670653 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8457020218687095,
            "unit": "iter/sec",
            "range": "stddev: 0.004199154844902848",
            "extra": "mean: 1.1824495793333274 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03588447342878114,
            "unit": "iter/sec",
            "range": "stddev: 0.7949580089310355",
            "extra": "mean: 27.867205631000008 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3440.7375132179372,
            "unit": "iter/sec",
            "range": "stddev: 0.000009255645739991982",
            "extra": "mean: 290.6353641213257 usec\nrounds: 2436"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 886.6554691155327,
            "unit": "iter/sec",
            "range": "stddev: 0.00001654825556378579",
            "extra": "mean: 1.1278337920788242 msec\nrounds: 606"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 348.4320533528023,
            "unit": "iter/sec",
            "range": "stddev: 0.00028218868758935384",
            "extra": "mean: 2.8700000197383027 msec\nrounds: 304"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 498.72140220079865,
            "unit": "iter/sec",
            "range": "stddev: 0.0004504557985729141",
            "extra": "mean: 2.0051275032254843 msec\nrounds: 775"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12090.910192750694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034684359295903387",
            "extra": "mean: 82.70675938024638 usec\nrounds: 7356"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11870.33557066305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036651965101303888",
            "extra": "mean: 84.24361670713428 usec\nrounds: 8667"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2037.7946477346247,
            "unit": "iter/sec",
            "range": "stddev: 0.00001725418659786762",
            "extra": "mean: 490.7265808709822 usec\nrounds: 1286"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2088.7946962069323,
            "unit": "iter/sec",
            "range": "stddev: 0.000027295466536352362",
            "extra": "mean: 478.74499194004665 usec\nrounds: 1737"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1169.9532250332054,
            "unit": "iter/sec",
            "range": "stddev: 0.000020021848513384796",
            "extra": "mean: 854.7350258140605 usec\nrounds: 891"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 387.85717513436646,
            "unit": "iter/sec",
            "range": "stddev: 0.00006279557503296489",
            "extra": "mean: 2.5782686620495476 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.37940318381745,
            "unit": "iter/sec",
            "range": "stddev: 0.001206586408605518",
            "extra": "mean: 51.601176285709286 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 465.5847832108623,
            "unit": "iter/sec",
            "range": "stddev: 0.000057503806743165694",
            "extra": "mean: 2.147836518847529 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 373.1479477622992,
            "unit": "iter/sec",
            "range": "stddev: 0.00018933955839104706",
            "extra": "mean: 2.679902183562362 msec\nrounds: 365"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 313.21492910490485,
            "unit": "iter/sec",
            "range": "stddev: 0.00011987263919180549",
            "extra": "mean: 3.1926958362354134 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.89876392105342,
            "unit": "iter/sec",
            "range": "stddev: 0.00006661267999211458",
            "extra": "mean: 4.787007741117693 msec\nrounds: 197"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 952.5906199358493,
            "unit": "iter/sec",
            "range": "stddev: 0.00002031208863563489",
            "extra": "mean: 1.04976889239928 msec\nrounds: 855"
          }
        ]
      }
    ]
  }
}