window.BENCHMARK_DATA = {
  "lastUpdate": 1784039961670,
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
          "id": "d7a2731a2152b0c0d78568d0f90f31f065aa5eee",
          "message": "feat: excellence fixes, Addendum Z closure (Branch B), and bench DX\n\nCode fixes and APIs:\n- fix async retrieve_with_explanations routing and numpy precomputed_weights checks\n- fix SQLiteStore.update rowcount guard, mem0 get() scan, Redis strict mode\n- add retrieve_query_gated(), engine_shared helpers, max_content_length and gate tau config\n- coerce LLM VAD list outputs in appraisal_schema (unblocks scored bench-z-profile)\n\nResearch (Addendum Z, Branch B):\n- scored run: Hz1 FAIL 0/3 third-party corpora; Hz2 PASS curated (+0.185 vs fixed)\n- closure doc, results artifacts, claim matrix and evidence propagation\n\nDX/docs/CI:\n- install-scored-bench, check_bench_deps.py, demo-check CI job, fakeredis dev extra\n- troubleshooting and configuration guides, examples, basedpyright in typecheck\n- 12 regression tests (test_excellence_fixes, test_check_bench_deps)",
          "timestamp": "2026-07-14T16:33:26+02:00",
          "tree_id": "01a4e7dd9ec985c720010f36035f7376fd6eb641",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/d7a2731a2152b0c0d78568d0f90f31f065aa5eee"
        },
        "date": 1784039960338,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 528.3668413905567,
            "unit": "iter/sec",
            "range": "stddev: 0.0008708346881671595",
            "extra": "mean: 1.8926244451074907 msec\nrounds: 1676"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 457.1015798049211,
            "unit": "iter/sec",
            "range": "stddev: 0.001015690047960037",
            "extra": "mean: 2.18769753634799 msec\nrounds: 1747"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 397.09646376400286,
            "unit": "iter/sec",
            "range": "stddev: 0.0013508200912393186",
            "extra": "mean: 2.5182797915679926 msec\nrounds: 2111"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 266.337840183057,
            "unit": "iter/sec",
            "range": "stddev: 0.002243044473027733",
            "extra": "mean: 3.7546298314677657 msec\nrounds: 3216"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 461.2778362840562,
            "unit": "iter/sec",
            "range": "stddev: 0.0009411768231618113",
            "extra": "mean: 2.167890848725272 msec\nrounds: 1765"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.801054199673,
            "unit": "iter/sec",
            "range": "stddev: 0.0002403192056681134",
            "extra": "mean: 2.6123230044146895 msec\nrounds: 453"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136124.30492562646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011784316062246762",
            "extra": "mean: 7.346226675290388 usec\nrounds: 30665"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.490033630676024,
            "unit": "iter/sec",
            "range": "stddev: 0.0002156727151581992",
            "extra": "mean: 35.09999366667197 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8146939610608044,
            "unit": "iter/sec",
            "range": "stddev: 0.001643657574338968",
            "extra": "mean: 1.2274547840000072 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03481543376881437,
            "unit": "iter/sec",
            "range": "stddev: 1.4150566034496337",
            "extra": "mean: 28.722893606333333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3154.6955478679047,
            "unit": "iter/sec",
            "range": "stddev: 0.00001471941188946466",
            "extra": "mean: 316.9877995598809 usec\nrounds: 2275"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 865.5266790802161,
            "unit": "iter/sec",
            "range": "stddev: 0.000020400143253661088",
            "extra": "mean: 1.1553658878114388 msec\nrounds: 722"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.1295925117489,
            "unit": "iter/sec",
            "range": "stddev: 0.00009371366283367978",
            "extra": "mean: 2.8158734757282065 msec\nrounds: 309"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 499.7642134379573,
            "unit": "iter/sec",
            "range": "stddev: 0.00043803714110934367",
            "extra": "mean: 2.0009435912204303 msec\nrounds: 729"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10617.431558080105,
            "unit": "iter/sec",
            "range": "stddev: 0.000006137095601200068",
            "extra": "mean: 94.18473710235291 usec\nrounds: 6067"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10462.80874973382,
            "unit": "iter/sec",
            "range": "stddev: 0.000005763104569457651",
            "extra": "mean: 95.57662993939753 usec\nrounds: 8250"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1812.483935190576,
            "unit": "iter/sec",
            "range": "stddev: 0.000011246154769001726",
            "extra": "mean: 551.7290280947255 usec\nrounds: 1139"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1795.8060645465998,
            "unit": "iter/sec",
            "range": "stddev: 0.000014938223473328248",
            "extra": "mean: 556.8530030844267 usec\nrounds: 1621"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1002.50518615343,
            "unit": "iter/sec",
            "range": "stddev: 0.000022597739372288306",
            "extra": "mean: 997.5010741210804 usec\nrounds: 796"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 368.8745631764297,
            "unit": "iter/sec",
            "range": "stddev: 0.000044725260455805126",
            "extra": "mean: 2.7109486525415636 msec\nrounds: 354"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.904198758242483,
            "unit": "iter/sec",
            "range": "stddev: 0.00149372982752933",
            "extra": "mean: 45.653347608695476 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.4434770167447,
            "unit": "iter/sec",
            "range": "stddev: 0.000026569402647748354",
            "extra": "mean: 2.1301819046562893 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.038043470624,
            "unit": "iter/sec",
            "range": "stddev: 0.00008726198334981053",
            "extra": "mean: 2.68789715877258 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.4207371531442,
            "unit": "iter/sec",
            "range": "stddev: 0.00010347869491120143",
            "extra": "mean: 3.306651552448288 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.5060577912351,
            "unit": "iter/sec",
            "range": "stddev: 0.00006715237662726853",
            "extra": "mean: 5.141228048914076 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 841.1626878231689,
            "unit": "iter/sec",
            "range": "stddev: 0.000030209253376896174",
            "extra": "mean: 1.188830667926895 msec\nrounds: 792"
          }
        ]
      }
    ]
  }
}