window.BENCHMARK_DATA = {
  "lastUpdate": 1781134755441,
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
          "id": "d7c395d35f1d1f938996f08b5c4ad227be6dfad9",
          "message": "docs: make state-based retrieval explicit, fix adaptive-weights direction, clarify affect-free senses and runner protocols (#54)\n\nFour conceptual gaps surfaced while designing Addendum Q:\n\n- mental_model.md: new callout 'The query is never appraised' — only the\n  semantic signal reads the query; s2-s4 compare memory tags against the\n  runtime affective state (Bower state-dependent recall). Also fixes a\n  factual error: the doc claimed high arousal -> semantic dominates and\n  low arousal -> emotional signals dominate, which is the opposite of\n  adaptive_weights() (calm -> semantic up; negative mood -> mood\n  congruence/affect proximity up; high arousal -> momentum up).\n- QueryClassifierConfig docstring: routed weights replace base_weights\n  but are still mood-modulated by adaptive_weights() and ablation-masked\n  — routing selects starting weights, it does not bypass adaptive\n  weighting (the trap behind Addendum Q Amendment 1).\n- 09_current_evidence.md: terminology box disambiguating the two senses\n  of 'affect-free' (noAF annotation vs semantically determinable query)\n  and what the falsified claim does and does not cover.\n- appraisal_confound/README.md: runner inventory + protocol note on\n  cross-scenario accumulation (runner_hg1/runner_hq reset once per\n  system; runner.py isolates per scenario) — absolute accuracies are not\n  comparable across the two families, only within-study deltas.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-06-11T01:31:44+02:00",
          "tree_id": "05e2917b61c83fe503d01d971c55fb856d30c5d2",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/d7c395d35f1d1f938996f08b5c4ad227be6dfad9"
        },
        "date": 1781134754924,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 579.5210463631682,
            "unit": "iter/sec",
            "range": "stddev: 0.000872882372098246",
            "extra": "mean: 1.7255628700209973 msec\nrounds: 1431"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 340.9327217971967,
            "unit": "iter/sec",
            "range": "stddev: 0.0018120865325292528",
            "extra": "mean: 2.9331300167628047 msec\nrounds: 1909"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 242.12095420598587,
            "unit": "iter/sec",
            "range": "stddev: 0.003232362675005699",
            "extra": "mean: 4.1301671029647595 msec\nrounds: 2496"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 147.1980320027514,
            "unit": "iter/sec",
            "range": "stddev: 0.004911364545290821",
            "extra": "mean: 6.793569087807561 msec\nrounds: 3576"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 409.350287881829,
            "unit": "iter/sec",
            "range": "stddev: 0.0013330108209067381",
            "extra": "mean: 2.4428955581647944 msec\nrounds: 1831"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 348.21904926324424,
            "unit": "iter/sec",
            "range": "stddev: 0.0003469928943052821",
            "extra": "mean: 2.8717555863637627 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134041.8617177637,
            "unit": "iter/sec",
            "range": "stddev: 8.180007533856072e-7",
            "extra": "mean: 7.460355945410421 usec\nrounds: 28392"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.95852025208568,
            "unit": "iter/sec",
            "range": "stddev: 0.0002821972214776183",
            "extra": "mean: 30.34116800000201 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8202644048539927,
            "unit": "iter/sec",
            "range": "stddev: 0.02019549072728343",
            "extra": "mean: 1.2191190963333345 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.021692250055038165,
            "unit": "iter/sec",
            "range": "stddev: 6.409448392889571",
            "extra": "mean: 46.099413268 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3450.725163688295,
            "unit": "iter/sec",
            "range": "stddev: 0.00001017817543735221",
            "extra": "mean: 289.79415994148707 usec\nrounds: 2057"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 878.4722634925616,
            "unit": "iter/sec",
            "range": "stddev: 0.000042239028918539716",
            "extra": "mean: 1.1383398674697798 msec\nrounds: 664"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.67970208083733,
            "unit": "iter/sec",
            "range": "stddev: 0.00004673960959333988",
            "extra": "mean: 2.8194452463256092 msec\nrounds: 272"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 476.6780252285479,
            "unit": "iter/sec",
            "range": "stddev: 0.0011270803315720824",
            "extra": "mean: 2.097852107867864 msec\nrounds: 788"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12473.953588362194,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034215118729048773",
            "extra": "mean: 80.16704510853467 usec\nrounds: 6163"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12473.715264689055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037489243952616776",
            "extra": "mean: 80.16857678568536 usec\nrounds: 7547"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2044.2616609160584,
            "unit": "iter/sec",
            "range": "stddev: 0.00001652359540018124",
            "extra": "mean: 489.1741693927224 usec\nrounds: 856"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2056.6475645221176,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134306503702674",
            "extra": "mean: 486.22817893077365 usec\nrounds: 1291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1185.5955343366304,
            "unit": "iter/sec",
            "range": "stddev: 0.000019319093528691694",
            "extra": "mean: 843.4579677794792 usec\nrounds: 869"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.72797552853433,
            "unit": "iter/sec",
            "range": "stddev: 0.00009289749524919703",
            "extra": "mean: 2.612821805406396 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.76928347984493,
            "unit": "iter/sec",
            "range": "stddev: 0.0008205267440707524",
            "extra": "mean: 48.14802595238429 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.79752182541665,
            "unit": "iter/sec",
            "range": "stddev: 0.00005808213462794548",
            "extra": "mean: 2.124055360620239 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 383.9430488517799,
            "unit": "iter/sec",
            "range": "stddev: 0.00008545561085959106",
            "extra": "mean: 2.6045529486485037 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 315.1141895044334,
            "unit": "iter/sec",
            "range": "stddev: 0.00011279625139307104",
            "extra": "mean: 3.1734527777776598 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 209.38752785502703,
            "unit": "iter/sec",
            "range": "stddev: 0.00011236125364373444",
            "extra": "mean: 4.775833643216642 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 967.6846531287224,
            "unit": "iter/sec",
            "range": "stddev: 0.00003367816213719381",
            "extra": "mean: 1.0333945017798882 msec\nrounds: 843"
          }
        ]
      }
    ]
  }
}