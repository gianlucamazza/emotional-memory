window.BENCHMARK_DATA = {
  "lastUpdate": 1782549963363,
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
          "id": "01780efec5b80f1f20bc38a5b335882197afd297",
          "message": "feat(bench): add Addendum T — retrieve-time query appraisal (production-reachable, Ht1 PASS) (#73)\n\n* feat(bench): add Addendum T — retrieve-time query appraisal (direction A)\n\nThe only untested lever on the state-injection boundary (A2). Tests whether\nappraising the query text (direct-VAD, Addendum V) and injecting that as the\nretrieve-time state can substitute for the oracle query.state that drives the\nheadline AFT advantage — i.e. whether the +0.205 is production-reachable.\n\n3 arms on realistic_recall_v2 (same embedder/top_k; only query-affect source differs):\ncosine / aft_oracle (headline upper bound) / aft_query_appraised (direct-VAD on query text).\nHt1: appraised beats cosine; Ht2: recovery fraction; diagnostic: corr(appraised, oracle state);\nsecondary: affect-favorable subset (Addendum U criterion).\n\n- preregistration_addendum_t_query_appraisal.md.\n- benchmarks/query_appraisal/runner.py: reuses realistic adapters + DIRECT_VAD_SCHEMA.\n- Makefile: bench-query-appraisal (llm-config-strict).\n\nResults + closure land in a follow-up commit once the confirmatory run completes.\n\nRefs #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): Addendum T — retrieve-time query appraisal is production-reachable (Ht1 PASS)\n\nConfirmatory run (realistic_recall_v2 N=200, SBERT, paired bootstrap seed=42). Replacing the\noracle query.state with the query's affect appraised at retrieve-time (direct-VAD on the query\ntext) makes AFT beat cosine WITHOUT any oracle:\n\n- aft_query_appraised vs cosine: Δ+0.115 [0.055, 0.180], p<0.001 — Ht1 PASS\n- aft_oracle vs cosine (upper bound): Δ+0.195\n- recovery fraction 0.59 overall; ~0.82 on the affect-discriminative subset (oracle Δ+0.304 →\n  appraised Δ+0.248)\n- diagnostic: appraised query affect tracks oracle state (valence r=0.80, arousal r=0.56)\n\nFirst mechanism to MOVE the state-injection boundary (A2) rather than only characterize it —\ntuning (Hj1), routing (Hl), gating (Hq), recalibration (Hp) all failed. The headline advantage\nis largely production-reachable (the oracle is not required for the affect-discriminative regime).\n\nCaveats: bounded to that regime (Addendum U); naturalistic QA (LoCoMo/DailyDialog) with query\nappraisal untested (next study). 08_limitations §2.4 updated; bounds downstream_value.\n\nRefs #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T10:40:45+02:00",
          "tree_id": "55e747adf4412e06600c38fc47f876cfe82e40bf",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/01780efec5b80f1f20bc38a5b335882197afd297"
        },
        "date": 1782549962608,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 550.2341315720448,
            "unit": "iter/sec",
            "range": "stddev: 0.0008415027516676465",
            "extra": "mean: 1.8174081588558544 msec\nrounds: 1643"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.05209921303674,
            "unit": "iter/sec",
            "range": "stddev: 0.0010718463859649928",
            "extra": "mean: 2.231883304991563 msec\nrounds: 1823"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 391.45422799108104,
            "unit": "iter/sec",
            "range": "stddev: 0.0013899916496251511",
            "extra": "mean: 2.554577083333442 msec\nrounds: 2196"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 247.52132155639555,
            "unit": "iter/sec",
            "range": "stddev: 0.002469054423214047",
            "extra": "mean: 4.040055998861329 msec\nrounds: 3513"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 462.4463393547439,
            "unit": "iter/sec",
            "range": "stddev: 0.0009437462874506109",
            "extra": "mean: 2.1624130518479405 msec\nrounds: 1813"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 388.9358449537071,
            "unit": "iter/sec",
            "range": "stddev: 0.0002555025591236857",
            "extra": "mean: 2.571118123913276 msec\nrounds: 460"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138137.31108045607,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012022038070286236",
            "extra": "mean: 7.239173777007752 usec\nrounds: 50513"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.647534156335926,
            "unit": "iter/sec",
            "range": "stddev: 0.000362561156391862",
            "extra": "mean: 34.90701833333295 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.838440897987935,
            "unit": "iter/sec",
            "range": "stddev: 0.0023995304229411625",
            "extra": "mean: 1.1926899110000118 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.038809207075655654,
            "unit": "iter/sec",
            "range": "stddev: 1.2617499485547106",
            "extra": "mean: 25.767081457 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3280.000126033519,
            "unit": "iter/sec",
            "range": "stddev: 0.00001104211828084502",
            "extra": "mean: 304.87803706559396 usec\nrounds: 2590"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 887.9973681764501,
            "unit": "iter/sec",
            "range": "stddev: 0.000017505345237386",
            "extra": "mean: 1.1261294637095076 msec\nrounds: 744"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 365.495271332501,
            "unit": "iter/sec",
            "range": "stddev: 0.00004785980942423939",
            "extra": "mean: 2.7360135094340876 msec\nrounds: 318"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 499.8322108302395,
            "unit": "iter/sec",
            "range": "stddev: 0.000787754755518619",
            "extra": "mean: 2.0006713819802924 msec\nrounds: 788"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10763.42943826306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047710463875876015",
            "extra": "mean: 92.90719149837936 usec\nrounds: 6940"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10519.68166366594,
            "unit": "iter/sec",
            "range": "stddev: 0.000004497402794404191",
            "extra": "mean: 95.0599107436789 usec\nrounds: 7204"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1873.4222598254764,
            "unit": "iter/sec",
            "range": "stddev: 0.000008947579717944125",
            "extra": "mean: 533.7824907093596 usec\nrounds: 1184"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1857.5169765658632,
            "unit": "iter/sec",
            "range": "stddev: 0.00000963421111998695",
            "extra": "mean: 538.3530878133765 usec\nrounds: 1674"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1026.5884619184067,
            "unit": "iter/sec",
            "range": "stddev: 0.00002087967468893815",
            "extra": "mean: 974.1001746028586 usec\nrounds: 819"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 383.9435812601584,
            "unit": "iter/sec",
            "range": "stddev: 0.000028264032022643413",
            "extra": "mean: 2.6045493369568917 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.52286004056071,
            "unit": "iter/sec",
            "range": "stddev: 0.0004363631222761113",
            "extra": "mean: 42.51183734782632 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 478.68758254985477,
            "unit": "iter/sec",
            "range": "stddev: 0.000020689152913883834",
            "extra": "mean: 2.0890452070497383 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 384.6844935068694,
            "unit": "iter/sec",
            "range": "stddev: 0.000025711365117997165",
            "extra": "mean: 2.599532907822142 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 311.3148648506858,
            "unit": "iter/sec",
            "range": "stddev: 0.000027068293292882903",
            "extra": "mean: 3.2121819832780045 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 199.12656486037034,
            "unit": "iter/sec",
            "range": "stddev: 0.000032927456822687044",
            "extra": "mean: 5.021931657894117 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 862.6653797321943,
            "unit": "iter/sec",
            "range": "stddev: 0.000024198256856615334",
            "extra": "mean: 1.1591980198746816 msec\nrounds: 805"
          }
        ]
      }
    ]
  }
}