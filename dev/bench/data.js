window.BENCHMARK_DATA = {
  "lastUpdate": 1782516068094,
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
          "id": "78435df7b14ca3aa7c1bf9f852ca8c0fd93a4151",
          "message": "feat(bench): Addendum U — circularity audit of realistic_recall_v2 (Hu1 PASS) (#70)\n\nPre-registered, deterministic (no LLM) audit quantifying §2.4's never-measured\n\"AFT-favorable vs neutral\" split. SBERT, N=200, bootstrap n=2000 seed=42:\n\n- 62.5% of queries are AFT-favorable by construction (gold not cosine-top-1 AND\n  affect-closest to the query state).\n- Aggregate top-1 advantage is concentrated there (Δ=+0.304 [+0.224, +0.384],\n  p<0.001) and is NULL on the neutral 37.5% (Δ=+0.013 [0.000, +0.040], p=0.63) — Hu1 PASS.\n- Hu2: the data-driven affect-separating metric flags all five challenge types as\n  ~88-98% separating — the author's per-type labels understate how favorable the design is.\n\nThe advantage is real where affect discriminates but confined to that regime; the\n+0.205/+0.18 headline figures are scoped to a ~62%-affect-discriminative benchmark,\nnot regime-independent. Honest re-scoping, no new claim.\n\n- benchmarks/circularity_audit/runner.py + pre-reg + closure + results.{json,md}.\n- Makefile: bench-circularity-audit (no LLM).\n- 08_limitations §2.4: \"not audited\" -> the numbers.\n- claim_validation_matrix.json: bound replayable_multi_session_help / realistic_replay_vs_sota.\n\nRefs #61; addresses 08_limitations §2.4\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T01:14:55+02:00",
          "tree_id": "164eb7b64cdf4ebe939b2ef5b7e304036867bd4a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/78435df7b14ca3aa7c1bf9f852ca8c0fd93a4151"
        },
        "date": 1782516066336,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 557.8437119984952,
            "unit": "iter/sec",
            "range": "stddev: 0.0008488384273037606",
            "extra": "mean: 1.7926167822479595 msec\nrounds: 1566"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 555.4210853808107,
            "unit": "iter/sec",
            "range": "stddev: 0.0007548649275562072",
            "extra": "mean: 1.8004357888472575 msec\nrounds: 1345"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 326.7842908982316,
            "unit": "iter/sec",
            "range": "stddev: 0.0018995811531549237",
            "extra": "mean: 3.0601226186586303 msec\nrounds: 2326"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 173.30251151375487,
            "unit": "iter/sec",
            "range": "stddev: 0.004244204826937332",
            "extra": "mean: 5.77025682585466 msec\nrounds: 3744"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 458.4074347839409,
            "unit": "iter/sec",
            "range": "stddev: 0.0009895075518836153",
            "extra": "mean: 2.181465491438474 msec\nrounds: 1752"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 374.1494769430743,
            "unit": "iter/sec",
            "range": "stddev: 0.0003519466537673106",
            "extra": "mean: 2.6727285794178646 msec\nrounds: 447"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 139289.8340936907,
            "unit": "iter/sec",
            "range": "stddev: 0.000001171574665220599",
            "extra": "mean: 7.179274830116954 usec\nrounds: 35320"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.395386101860552,
            "unit": "iter/sec",
            "range": "stddev: 0.00021852277985654866",
            "extra": "mean: 35.216989000000844 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8119455198164497,
            "unit": "iter/sec",
            "range": "stddev: 0.0034170033284963252",
            "extra": "mean: 1.2316097269999868 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03210590795823778,
            "unit": "iter/sec",
            "range": "stddev: 3.0194309793538263",
            "extra": "mean: 31.146915430666667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3211.810807346555,
            "unit": "iter/sec",
            "range": "stddev: 0.000013789300952082167",
            "extra": "mean: 311.350842245329 usec\nrounds: 2263"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 863.9228890324124,
            "unit": "iter/sec",
            "range": "stddev: 0.000014595408485840749",
            "extra": "mean: 1.1575107138554843 msec\nrounds: 664"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.9914836775803,
            "unit": "iter/sec",
            "range": "stddev: 0.000031302114682338976",
            "extra": "mean: 2.7855814008616604 msec\nrounds: 232"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 498.5122627696828,
            "unit": "iter/sec",
            "range": "stddev: 0.0010306220085308738",
            "extra": "mean: 2.0059687086614537 msec\nrounds: 762"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10653.693235119516,
            "unit": "iter/sec",
            "range": "stddev: 0.000005304040467841405",
            "extra": "mean: 93.864163152693 usec\nrounds: 5976"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10812.833742921151,
            "unit": "iter/sec",
            "range": "stddev: 0.000004949469394786881",
            "extra": "mean: 92.48269452535243 usec\nrounds: 8220"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1859.6492312175772,
            "unit": "iter/sec",
            "range": "stddev: 0.000010961357126243195",
            "extra": "mean: 537.7358177086252 usec\nrounds: 1152"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1844.5738591896718,
            "unit": "iter/sec",
            "range": "stddev: 0.00002077983993090019",
            "extra": "mean: 542.1306363082169 usec\nrounds: 1647"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1020.431685679904,
            "unit": "iter/sec",
            "range": "stddev: 0.00002906426000558235",
            "extra": "mean: 979.9774095937736 usec\nrounds: 813"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 379.73723429423103,
            "unit": "iter/sec",
            "range": "stddev: 0.00005435412533063557",
            "extra": "mean: 2.63339991364969 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.88874198744111,
            "unit": "iter/sec",
            "range": "stddev: 0.0005002609688624619",
            "extra": "mean: 47.87267709090512 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.4462811141199,
            "unit": "iter/sec",
            "range": "stddev: 0.00003537112591174283",
            "extra": "mean: 2.121132438751673 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.17118607698313,
            "unit": "iter/sec",
            "range": "stddev: 0.00004702962997406052",
            "extra": "mean: 2.672573509693654 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 303.31076931537956,
            "unit": "iter/sec",
            "range": "stddev: 0.00006866802550330592",
            "extra": "mean: 3.2969485463940447 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.89862904615225,
            "unit": "iter/sec",
            "range": "stddev: 0.00007768247340726094",
            "extra": "mean: 5.078755524324164 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 857.383432045342,
            "unit": "iter/sec",
            "range": "stddev: 0.000028549102607553934",
            "extra": "mean: 1.1663393093734469 msec\nrounds: 779"
          }
        ]
      }
    ]
  }
}