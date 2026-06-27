window.BENCHMARK_DATA = {
  "lastUpdate": 1782521016509,
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
          "id": "0c919f94ce61828e3253683ab83c69371aa47b57",
          "message": "feat(bench): add Addendum V — direct-VAD beats SEC→projection appraisal (#71)\n\n* feat(bench): add Addendum V — direct-VAD vs SEC→projection appraisal (#62 follow-up)\n\nDirection C: tests whether asking the LLM for valence/arousal/dominance directly\nbeats the production path (5 Scherer SECs → Addendum-O M1 linear projection) on\nhuman-gold affect, paired per item against EmoBank. Addendum O is already live in\n`_scherer_project`, so this isolates whether the SEC→projection is the bottleneck on\nthe weak arousal/dominance axes (Addendum S: r 0.28/0.33; +0.15 valence bias).\n\n- preregistration_addendum_v_direct_vad.md: Hv1 (arousal r↑), Hv2 (dominance r↑),\n  Hv3 (valence |bias|↓), guard Gv (valence not regressed); paired-bootstrap plan.\n- benchmarks/appraisal_vad/runner.py: two LLMAppraisalEngine instances (default Scherer\n  schema vs a custom 3-dim direct-VAD AppraisalSchema, identity projection — no library\n  change), per-dim r/bias/MAE + paired Δr vs EmoBank. Reuses human_gold_appraisal helpers.\n- Makefile: bench-appraisal-vad (llm-config-strict).\n\nResults + closure land in a follow-up commit once the confirmatory run completes.\n\nRefs #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): Addendum V — direct-VAD beats SEC→projection on human gold\n\nConfirmatory run (EmoBank N=300, paired bootstrap seed=42). Direct-VAD (LLM rates\nV/A/D directly via a custom AppraisalSchema; no engine change) vs the production\nScherer-SEC→Addendum-O projection:\n\n- valence r 0.695→0.790, bias +0.157→−0.013 (Δr +0.095 [0.052, 0.141])\n- arousal r 0.228→0.582 (Δr +0.354 [0.251, 0.457]) — the weak axis recovered\n- dominance r 0.307→0.428 (Δr +0.122 [−0.008, 0.243])\n- Hv1 PASS, Hv3 PASS, Gv OK; Hv2 (dominance) positive but CI touches 0 → FAIL by rule.\n- Decision: adopt direct-VAD = YES. The SEC→linear-projection was the bottleneck.\n\nHonest caveat: arousal MAE worsens (0.112→0.193) — correlation up, absolute scale\nneeds a small affine calibration. This is appraisal quality vs human gold, not yet a\nretrieval-benefit claim (cf. Addendum P). Follow-up: expose the direct-VAD schema in the\nlibrary and feed it to Addendum T (retrieve-time query appraisal).\n\n- results.{json,md} + Addendum V closure.\n- claim_validation_matrix.json: bound appraisal_human_validated (not_yet_shown).\n- 08_limitations §1.1: dominance/axis estimation updated.\n\nRefs #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T02:37:30+02:00",
          "tree_id": "62911c75df70f8be5b6549d4fa7866d08494c1b4",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/0c919f94ce61828e3253683ab83c69371aa47b57"
        },
        "date": 1782521015264,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 569.9480947935058,
            "unit": "iter/sec",
            "range": "stddev: 0.0008140151793075412",
            "extra": "mean: 1.7545457369452275 msec\nrounds: 1532"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 451.4945974883231,
            "unit": "iter/sec",
            "range": "stddev: 0.0010384030825635112",
            "extra": "mean: 2.2148659265537787 msec\nrounds: 1770"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 366.8504298502993,
            "unit": "iter/sec",
            "range": "stddev: 0.0015524845404080922",
            "extra": "mean: 2.725906578351483 msec\nrounds: 2208"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 203.95458795080603,
            "unit": "iter/sec",
            "range": "stddev: 0.0032842544073216195",
            "extra": "mean: 4.9030522433807695 msec\nrounds: 3739"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 474.51206095225007,
            "unit": "iter/sec",
            "range": "stddev: 0.0009184474851983135",
            "extra": "mean: 2.10742799243754 msec\nrounds: 1719"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.39907866216396,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348604516258648",
            "extra": "mean: 2.6219255786031437 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135932.20123125438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011121457589342685",
            "extra": "mean: 7.356608595624461 usec\nrounds: 46512"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.956951550509714,
            "unit": "iter/sec",
            "range": "stddev: 0.0002886013520796982",
            "extra": "mean: 35.769279000012 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8235565578359688,
            "unit": "iter/sec",
            "range": "stddev: 0.0019218469173909293",
            "extra": "mean: 1.2142456889999949 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.0323347722403374,
            "unit": "iter/sec",
            "range": "stddev: 1.970754556012298",
            "extra": "mean: 30.92645875366665 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3195.935718660893,
            "unit": "iter/sec",
            "range": "stddev: 0.000012134021681978633",
            "extra": "mean: 312.89740721662673 usec\nrounds: 2134"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 869.6785553515693,
            "unit": "iter/sec",
            "range": "stddev: 0.000023762923051665193",
            "extra": "mean: 1.1498501300813933 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 357.24416138020064,
            "unit": "iter/sec",
            "range": "stddev: 0.000035781767616340204",
            "extra": "mean: 2.799205999998808 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 494.6470681935793,
            "unit": "iter/sec",
            "range": "stddev: 0.0008325685652168785",
            "extra": "mean: 2.021643438930991 msec\nrounds: 786"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10817.516592627982,
            "unit": "iter/sec",
            "range": "stddev: 0.000005406253495038305",
            "extra": "mean: 92.44265922194091 usec\nrounds: 6761"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10614.353551214219,
            "unit": "iter/sec",
            "range": "stddev: 0.000006069936426890761",
            "extra": "mean: 94.21204929485376 usec\nrounds: 7303"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1822.9424837437814,
            "unit": "iter/sec",
            "range": "stddev: 0.00002116905883590137",
            "extra": "mean: 548.5636595326351 usec\nrounds: 1028"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1834.7259813166684,
            "unit": "iter/sec",
            "range": "stddev: 0.00001336057583253976",
            "extra": "mean: 545.0405184115627 usec\nrounds: 1385"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1015.8490809887436,
            "unit": "iter/sec",
            "range": "stddev: 0.000016759755144618525",
            "extra": "mean: 984.3981933090718 usec\nrounds: 807"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 366.82318302024913,
            "unit": "iter/sec",
            "range": "stddev: 0.00010031646984182173",
            "extra": "mean: 2.7261090527770664 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.13573976901404,
            "unit": "iter/sec",
            "range": "stddev: 0.0011543686662960646",
            "extra": "mean: 45.175811173919556 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.08139538867255,
            "unit": "iter/sec",
            "range": "stddev: 0.00003074011179747567",
            "extra": "mean: 2.1318261816191146 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.7783272081425,
            "unit": "iter/sec",
            "range": "stddev: 0.000047869556544103885",
            "extra": "mean: 2.6682439388887738 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 303.57695473388236,
            "unit": "iter/sec",
            "range": "stddev: 0.00006131233946214155",
            "extra": "mean: 3.2940576825951986 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.12142762806457,
            "unit": "iter/sec",
            "range": "stddev: 0.00011689016334429957",
            "extra": "mean: 5.178089310347865 msec\nrounds: 174"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 834.927517378092,
            "unit": "iter/sec",
            "range": "stddev: 0.00005252295605023149",
            "extra": "mean: 1.1977087581689512 msec\nrounds: 765"
          }
        ]
      }
    ]
  }
}