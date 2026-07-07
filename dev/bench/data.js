window.BENCHMARK_DATA = {
  "lastUpdate": 1783465898271,
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
          "id": "fca0db5575e07ae1476107cb18b33a66ccb4ddcd",
          "message": "docs(bench): pre-register addendum z + learning-to-rank harness (#117)\n\n* docs(bench): pre-register addendum z (held-out learned retrieval profile)\n\nEvery third-party FAIL (K/T2A, X, X2) was run with a single fixed weight vector;\nAddendum J only swept hand-authored grids. No addendum has tested a *learned*\nprofile. Z fits a linear pairwise learning-to-rank over the 6 AFT signals per\ncorpus and evaluates it strictly out-of-sample via k-fold cross-fitting — which\nresolves the circularity that kept the \"support-mode profile\" residual (X)\nunscheduled (we test generalization, not fit).\n\nHz1 (break): held-out learned > cosine on >=1 of {MADial, ES-MemEval, DailyDialog}\n(Holm m=3) -> Branch A, provenance bound broken. Hz2 (preserve): non-inferior to\nthe fixed profile on curated. Honest ex-ante expectation is Branch B: even the\nheld-out optimal linear profile does not beat cosine on affect-orthogonal /\ncounter-congruent gold, hardening the boundary to its strongest form. s2 sign is\nfree (negative = counter-congruent readout). Harness-only, zero src/ (reuses\nbuild_retrieval_plan precomputed_weights + committed direct-VAD appraisals).\n\nCommitted before the harness runs a single scored evaluation.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): addendum z harness — held-out learned retrieval profile\n\nLearning-to-rank core + runner for Addendum Z (harness-only, zero src/).\n\n- benchmarks/common/ltr.py: pairwise logistic learning-to-rank over the 6 AFT\n  signals with k-fold cross-fitting (fit on train folds, score held-out folds by\n  an unseen w). Pure numpy, 6 parameters. score_fixed() for the cosine (e1) and\n  aft_fixed (base_weights) arms. Fully unit-tested, including recovery of a\n  NEGATIVE s2 weight on synthetic counter-congruent data (the Hz interpretability\n  readout) and held-out generalization beating chance.\n- benchmarks/learned_profile/runner.py: per query, extracts the 6-signal feature\n  matrix over the full candidate pool via build_retrieval_plan(precomputed_weights)\n  (raw signals are weight-independent), builds the three arms, and applies the\n  pre-registered Holm m=3 (Hz1) + non-inferiority (Hz2) verdict. Wires the two\n  dry-capable third-party break corpora (MADial X, ES-MemEval X2), validated\n  end-to-end by --dry-run (keyword appraiser, no LLM). DailyDialog + curated are\n  added in the run+closure PR (both stateful/LLM-only); until the full family is\n  present the Holm verdict is correctly reported not-evaluable.\n- make bench-z-profile[-dry]; tests/test_ltr.py.\n\nDry smoke confirms the mechanism and directionally matches X/X2 (aft_fixed < cosine\non both). The scored verdict requires the LLM key and the full 4-corpus family.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): wire dailydialog + curated into addendum z (all 4 corpora dry-capable)\n\nAdds the DailyDialog (break) and realistic_recall_v2 (preserve) feature\nextractors, completing the pre-registered 4-corpus family. Both are dry-capable:\ntheir encode side needs no LLM (DailyDialog uses oracle session PAD + keyword;\ncurated replays the timeline with oracle per-event PAD), so only the query-affect\nsource differs (keyword in dry, direct-VAD when scored). curated builds the pool\nas \"memories encoded so far\" by replaying scenario -> session -> events -> queries.\n\n--dry-run now validates the full pipeline end-to-end on all four corpora and\nexercises the Holm m=3 + non-inferiority verdict wiring (labelled non-scored).\nThe scored run (direct-VAD, full corpora) produces the real Hz1/Hz2 verdict.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): document addendum z (ladder, roadmap, changelog, harness index)\n\nPropagate the Addendum Z pre-registration + harness to the reference docs:\nbenchmarks/README ladder row + harness index, docs/research/index ladder entry,\nROADMAP item, CHANGELOG [Unreleased], CLAUDE.md commands + addendum-harness list.\nNo claim added to the matrix — Z is not yet executed (that is the closure's job).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-08T01:06:01+02:00",
          "tree_id": "3120e9d568c683d604423f59fdb138957d69af18",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/fca0db5575e07ae1476107cb18b33a66ccb4ddcd"
        },
        "date": 1783465897341,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 538.038971916556,
            "unit": "iter/sec",
            "range": "stddev: 0.000898813834592157",
            "extra": "mean: 1.8586014251679321 msec\nrounds: 1637"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 417.5630916770665,
            "unit": "iter/sec",
            "range": "stddev: 0.0011768584462636255",
            "extra": "mean: 2.394847676751509 msec\nrounds: 1884"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 380.1707110461812,
            "unit": "iter/sec",
            "range": "stddev: 0.0014393306080773022",
            "extra": "mean: 2.63039726876415 msec\nrounds: 2225"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 223.57923624888312,
            "unit": "iter/sec",
            "range": "stddev: 0.0027424515209860728",
            "extra": "mean: 4.472687252973812 msec\nrounds: 3783"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 456.13127541619866,
            "unit": "iter/sec",
            "range": "stddev: 0.0009545349002174551",
            "extra": "mean: 2.192351311774327 msec\nrounds: 1809"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.05530974825086,
            "unit": "iter/sec",
            "range": "stddev: 0.0003216219534837226",
            "extra": "mean: 2.61742207079633 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136525.5795245703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011673650483842668",
            "extra": "mean: 7.324634720338482 usec\nrounds: 49291"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.301021475372682,
            "unit": "iter/sec",
            "range": "stddev: 0.00042297974261766477",
            "extra": "mean: 35.33441366666542 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8209113299954897,
            "unit": "iter/sec",
            "range": "stddev: 0.006597774899069687",
            "extra": "mean: 1.2181583606666682 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03675155195198722,
            "unit": "iter/sec",
            "range": "stddev: 1.265371497245222",
            "extra": "mean: 27.20973528699999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3161.7070000638164,
            "unit": "iter/sec",
            "range": "stddev: 0.000013156282295163471",
            "extra": "mean: 316.2848423272036 usec\nrounds: 2372"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 866.533021482187,
            "unit": "iter/sec",
            "range": "stddev: 0.000014251543725967683",
            "extra": "mean: 1.1540241112675895 msec\nrounds: 710"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 352.6983954926739,
            "unit": "iter/sec",
            "range": "stddev: 0.00002157537313603573",
            "extra": "mean: 2.8352836666668977 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 495.5544761930158,
            "unit": "iter/sec",
            "range": "stddev: 0.0004418190355333595",
            "extra": "mean: 2.0179416149810048 msec\nrounds: 761"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10488.835743996788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050003600761252855",
            "extra": "mean: 95.33946611494446 usec\nrounds: 7127"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10386.828114263584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052214446252241165",
            "extra": "mean: 96.2757820769906 usec\nrounds: 8012"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1835.1059554195208,
            "unit": "iter/sec",
            "range": "stddev: 0.000008890919182603575",
            "extra": "mean: 544.927663193917 usec\nrounds: 1152"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1838.4535870647862,
            "unit": "iter/sec",
            "range": "stddev: 0.000010671174211912911",
            "extra": "mean: 543.9354069289107 usec\nrounds: 1703"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 999.4193815944211,
            "unit": "iter/sec",
            "range": "stddev: 0.000018362743092541616",
            "extra": "mean: 1.0005809557191623 msec\nrounds: 813"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.53386092524124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377031171566661",
            "extra": "mean: 2.691544715492881 msec\nrounds: 355"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.214943932223303,
            "unit": "iter/sec",
            "range": "stddev: 0.0015033086962849276",
            "extra": "mean: 43.075701708327564 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.15404827299324,
            "unit": "iter/sec",
            "range": "stddev: 0.000028100555097350212",
            "extra": "mean: 2.12696243640415 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.2843842887973,
            "unit": "iter/sec",
            "range": "stddev: 0.00003327097292752628",
            "extra": "mean: 2.657564442622478 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.4688811333199,
            "unit": "iter/sec",
            "range": "stddev: 0.00004587069986613785",
            "extra": "mean: 3.2844079049317463 msec\nrounds: 284"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.88290171708002,
            "unit": "iter/sec",
            "range": "stddev: 0.000053931949907590184",
            "extra": "mean: 5.13128648634216 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 850.4924499597854,
            "unit": "iter/sec",
            "range": "stddev: 0.000020365404819782055",
            "extra": "mean: 1.1757893912488981 msec\nrounds: 800"
          }
        ]
      }
    ]
  }
}