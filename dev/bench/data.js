window.BENCHMARK_DATA = {
  "lastUpdate": 1783424360760,
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
          "id": "e27a442527ed1f8c0baa254557bf7185d287474e",
          "message": "docs(bench): pre-register addendum y query-affect-conditioned gate (#110)\n\nPre-registration for Hy: a per-query gate that appraises the query and routes\nneutral queries (|valence| < tau) to pure cosine, affect-carrying queries to\nthe Addendum-T affect-conditioned path. Composition of Q (gating) + T (query\nappraisal); the untested variant named in the X2 closure.\n\n- gated arm = exact per-query selection between the two already-computed arms\n  (naive_cosine, aft_query_appraised) -> no new retrieval, no new LLM calls,\n  exact cosine-equivalence on neutral queries; zero src/ change (harness-only)\n- 4 corpora each on its own metric: curated v2 (preserve), T2A/DailyDialog +\n  X2/ES-MemEval + X/MADial (recover spectrum)\n- confirmatory family Holm m=2: Hg1 (gated>aft on X2), Hg2 (gated>cosine on\n  curated); paired bootstrap n=10k seed=0 one-tailed\n- tau=0.2 primary (schema neutral=0, calibrated valence axis) + {0.1,0.2,0.3}\n  sensitivity; honest disclosure of 0.2's prior exposure via X2 post-hoc\n- honest expected outcome declared: partial safe wrapper, penalty decomposition\n  (gateable neutral component vs non-gateable affect-carrying), NOT AFT>cosine\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T13:31:22+02:00",
          "tree_id": "2fc126aea6d11152aa0a4f22cfc6c861f51c059d",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e27a442527ed1f8c0baa254557bf7185d287474e"
        },
        "date": 1783424359448,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 590.6218723577841,
            "unit": "iter/sec",
            "range": "stddev: 0.0008125271332224178",
            "extra": "mean: 1.693130659059346 msec\nrounds: 1446"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 374.31224681960316,
            "unit": "iter/sec",
            "range": "stddev: 0.0016082355040435965",
            "extra": "mean: 2.6715663419955966 msec\nrounds: 1924"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 269.9013266896637,
            "unit": "iter/sec",
            "range": "stddev: 0.0027297222103218753",
            "extra": "mean: 3.7050577418977046 msec\nrounds: 2561"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 143.48339872737472,
            "unit": "iter/sec",
            "range": "stddev: 0.0055393157465277775",
            "extra": "mean: 6.9694473985805665 msec\nrounds: 4087"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 392.75186148702414,
            "unit": "iter/sec",
            "range": "stddev: 0.0013408957965001162",
            "extra": "mean: 2.546136881984042 msec\nrounds: 1915"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 344.40861645999604,
            "unit": "iter/sec",
            "range": "stddev: 0.0006446410620796621",
            "extra": "mean: 2.9035278219183365 msec\nrounds: 438"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136530.59639525853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010077161124828517",
            "extra": "mean: 7.324365573742767 usec\nrounds: 30593"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.68498204499564,
            "unit": "iter/sec",
            "range": "stddev: 0.0005423122089816493",
            "extra": "mean: 30.59509099999976 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8233526368880497,
            "unit": "iter/sec",
            "range": "stddev: 0.004406318308540262",
            "extra": "mean: 1.2145464230000016 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.021182330193387482,
            "unit": "iter/sec",
            "range": "stddev: 1.553059466277088",
            "extra": "mean: 47.20915927899998 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3476.5443417461256,
            "unit": "iter/sec",
            "range": "stddev: 0.000009816006552434894",
            "extra": "mean: 287.6419518060112 usec\nrounds: 2241"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 880.3964206776835,
            "unit": "iter/sec",
            "range": "stddev: 0.000038769017371651495",
            "extra": "mean: 1.1358519599957617 msec\nrounds: 50"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 353.71186448274193,
            "unit": "iter/sec",
            "range": "stddev: 0.00005075505495510784",
            "extra": "mean: 2.827159901640199 msec\nrounds: 305"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 484.18331625219065,
            "unit": "iter/sec",
            "range": "stddev: 0.0005006212243540886",
            "extra": "mean: 2.065333452090989 msec\nrounds: 741"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12174.071362507619,
            "unit": "iter/sec",
            "range": "stddev: 0.000004018498716845161",
            "extra": "mean: 82.14178890717622 usec\nrounds: 5481"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12153.22589500722,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036533537144021463",
            "extra": "mean: 82.28268022326644 usec\nrounds: 6098"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1988.5446760473358,
            "unit": "iter/sec",
            "range": "stddev: 0.000018701032049437617",
            "extra": "mean: 502.8803285364034 usec\nrounds: 834"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1987.4110150716078,
            "unit": "iter/sec",
            "range": "stddev: 0.00003529731005764712",
            "extra": "mean: 503.16718203555354 usec\nrounds: 1258"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1176.6618485209679,
            "unit": "iter/sec",
            "range": "stddev: 0.000027150100870974462",
            "extra": "mean: 849.8618369049468 usec\nrounds: 840"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 363.10167372248736,
            "unit": "iter/sec",
            "range": "stddev: 0.0002472616192996108",
            "extra": "mean: 2.754049546916392 msec\nrounds: 373"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 18.779143417224397,
            "unit": "iter/sec",
            "range": "stddev: 0.0008905160143665798",
            "extra": "mean: 53.2505651499946 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 456.4530136926923,
            "unit": "iter/sec",
            "range": "stddev: 0.00015628866327256383",
            "extra": "mean: 2.1908059975549894 msec\nrounds: 410"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 342.869517507889,
            "unit": "iter/sec",
            "range": "stddev: 0.0003712088755748556",
            "extra": "mean: 2.9165613999996114 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 276.3609938427245,
            "unit": "iter/sec",
            "range": "stddev: 0.0004189889068780464",
            "extra": "mean: 3.618455651411843 msec\nrounds: 284"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 181.4476950443511,
            "unit": "iter/sec",
            "range": "stddev: 0.0007762392178699338",
            "extra": "mean: 5.511230108244532 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 948.7968384665952,
            "unit": "iter/sec",
            "range": "stddev: 0.000019752091519023723",
            "extra": "mean: 1.0539664124684027 msec\nrounds: 834"
          }
        ]
      }
    ]
  }
}