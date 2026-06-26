window.BENCHMARK_DATA = {
  "lastUpdate": 1782509659875,
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
          "id": "e9774017f8ceb1db635be3cc4fcca071c4a5dc78",
          "message": "feat(bench): add A3 downstream benchmark, ranking edge converts to answer quality (#61) (#65)\n\n* feat(bench): add A3 downstream generate→judge benchmark (Addendum R, #61)\n\nPre-registers and implements the encode→retrieve→generate→judge task that A3\nrecorded as future work: does AFT's retrieval-ranking edge on the\naffect-discriminative regime (realistic_recall_v2) convert to downstream answer\nquality once an LLM generator consumes the retrieved memories?\n\n- preregistration_addendum_r_downstream.md: Hr1 (Δ judge_correct > 0), Hr2\n  (Δ F1 > 0), pre-declared paired-bootstrap + Holm plan, decision rule.\n- benchmarks/downstream/runner.py: reuses AFTReplayAdapter / NaiveCosineReplayAdapter\n  (oracle affect, identical generator + LoCoMo judge), pairs by query_id, reports\n  Δ with bootstrap CI + McNemar and a retrieval-ranking reference row.\n- Makefile: bench-a3 (llm-config-strict) + bench-a3-dry.\n\nResults + closure + claim-matrix update land in a follow-up commit once the\nconfirmatory run completes.\n\nRefs #61\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): A3 downstream PASS — ranking edge converts to answer quality (#61)\n\nConfirmatory run of Addendum R (encode→retrieve→generate→judge) on\nrealistic_recall_v2 (N=200, sbert-bge, gpt-5-mini judge):\n\n- Hr1 judge_correct: AFT 0.595 vs cosine 0.440, Δ=+0.155 [+0.095, +0.220],\n  p<0.001, Holm<0.001, McNemar<0.001 — PASS.\n- Hr2 token-F1: Δ=+0.152 [+0.100, +0.205], p<0.001 — PASS.\n- Ranking reference top1 Δ=+0.205 matches Hd2; the ranking edge converts ~1:1\n  to a +0.155 answer-quality edge. Positive across all 5 challenge types.\n\nBounded by the A2 oracle-affect + state-injection regime; does NOT contradict\nthe LoCoMo oracle-free negative (AFT loses there). Both recorded.\n\n- results.{json,md} + Addendum R closure.\n- claim_validation_matrix.json: new claim downstream_value = early_controlled_evidence\n  (requires_oracle_affect=true).\n- problem_register §A3 updated: future-work item now executed, scoped by regime.\n- README \"When NOT to use\": in-regime downstream conversion documented honestly.\n\nCloses #61\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(research): list downstream_value claim in current-evidence matrix (#61)\n\nThe claim-matrix test requires every claim_id and its exact allowed_public_wording\nto appear in 09_current_evidence.md. Add the downstream_value row.\n\nRefs #61\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-26T23:28:34+02:00",
          "tree_id": "39856335dad4b60bc83e8e1e44988eca7f0fa237",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e9774017f8ceb1db635be3cc4fcca071c4a5dc78"
        },
        "date": 1782509658917,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 582.1618164533502,
            "unit": "iter/sec",
            "range": "stddev: 0.0008421920182077264",
            "extra": "mean: 1.7177354675925092 msec\nrounds: 1512"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 417.52510708585254,
            "unit": "iter/sec",
            "range": "stddev: 0.001253802286884799",
            "extra": "mean: 2.395065549421864 msec\nrounds: 1902"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 413.7472057869353,
            "unit": "iter/sec",
            "range": "stddev: 0.0013024680156441444",
            "extra": "mean: 2.4169347514940407 msec\nrounds: 2008"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 183.5001924887594,
            "unit": "iter/sec",
            "range": "stddev: 0.0033386477526039294",
            "extra": "mean: 5.449585564120085 msec\nrounds: 4398"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 425.53117628728023,
            "unit": "iter/sec",
            "range": "stddev: 0.0010877674287850586",
            "extra": "mean: 2.3500040789605747 msec\nrounds: 1963"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.8167767232129,
            "unit": "iter/sec",
            "range": "stddev: 0.00025232200772616116",
            "extra": "mean: 2.6190572572061734 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134284.70226690327,
            "unit": "iter/sec",
            "range": "stddev: 8.777532490296164e-7",
            "extra": "mean: 7.4468646325208905 usec\nrounds: 43851"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.85085789358731,
            "unit": "iter/sec",
            "range": "stddev: 0.0006049995232413769",
            "extra": "mean: 30.44060533333 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8483164751792632,
            "unit": "iter/sec",
            "range": "stddev: 0.0009036206408678181",
            "extra": "mean: 1.1788053506666643 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03659946312194111,
            "unit": "iter/sec",
            "range": "stddev: 0.5863081878160844",
            "extra": "mean: 27.322805164333335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3394.4544669818965,
            "unit": "iter/sec",
            "range": "stddev: 0.000009675297577296973",
            "extra": "mean: 294.598148164034 usec\nrounds: 2315"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 875.2268433730013,
            "unit": "iter/sec",
            "range": "stddev: 0.000016591819045313303",
            "extra": "mean: 1.1425609344271714 msec\nrounds: 61"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 353.2050567264065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000345347808713646",
            "extra": "mean: 2.8312165439199886 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 496.08821939896785,
            "unit": "iter/sec",
            "range": "stddev: 0.0004505851885537801",
            "extra": "mean: 2.0157705039066296 msec\nrounds: 768"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 11999.688478923686,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033657381130151957",
            "extra": "mean: 83.33549673030303 usec\nrounds: 7952"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12104.500396320982,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037532749602345383",
            "extra": "mean: 82.6139012151165 usec\nrounds: 8888"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1998.9259680743228,
            "unit": "iter/sec",
            "range": "stddev: 0.00001193826991158151",
            "extra": "mean: 500.26865225196707 usec\nrounds: 1110"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2007.2619794587333,
            "unit": "iter/sec",
            "range": "stddev: 0.000011264481856001844",
            "extra": "mean: 498.1910733294785 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1146.7472587414966,
            "unit": "iter/sec",
            "range": "stddev: 0.000014719326563885336",
            "extra": "mean: 872.0317335639022 usec\nrounds: 867"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.5065687431149,
            "unit": "iter/sec",
            "range": "stddev: 0.00005832237540253864",
            "extra": "mean: 2.614334188523658 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.705384521580076,
            "unit": "iter/sec",
            "range": "stddev: 0.0005083862762328858",
            "extra": "mean: 44.0424164166681 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 473.34750588616595,
            "unit": "iter/sec",
            "range": "stddev: 0.00006543019098119687",
            "extra": "mean: 2.112612800458037 msec\nrounds: 436"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 387.9154403106,
            "unit": "iter/sec",
            "range": "stddev: 0.00007806818056979493",
            "extra": "mean: 2.5778814042547777 msec\nrounds: 376"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 316.386935236171,
            "unit": "iter/sec",
            "range": "stddev: 0.0001023963152279711",
            "extra": "mean: 3.1606867687299967 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 207.50376128601218,
            "unit": "iter/sec",
            "range": "stddev: 0.00014488642729724476",
            "extra": "mean: 4.819189752525271 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 948.5846153299997,
            "unit": "iter/sec",
            "range": "stddev: 0.00002203521220685553",
            "extra": "mean: 1.0542022122634929 msec\nrounds: 848"
          }
        ]
      }
    ]
  }
}