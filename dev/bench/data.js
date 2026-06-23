window.BENCHMARK_DATA = {
  "lastUpdate": 1782217380523,
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
          "id": "439f1f2b92dc220842f9ab87a7363de796be6cd3",
          "message": "docs: add external review response (June 2026) (#59)\n\n* docs: add June 2026 response to external AFT review\n\nReconcile an external critical review against the current repository.\nMost of the review's criticisms (LoCoMo downstream gap, mechanism\nablation, \"field\" framing, prior-art positioning, CIs/effect sizes)\nare already addressed by committed pre-registered studies and closures;\nthis doc maps each point to its artifact and isolates the genuinely-open\nresidue (human/ecological validation, construct validity vs human gold,\ndownstream end-to-end value, multi-seed robustness).\n\nAdd the page to the Research nav alongside the 2026-04 audit.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FnY4dRqa2xvyK4kQcLbqJQ\n\n* docs: add problem register and apply honest re-scoping of claims\n\nIdentify every open problem (research/evidence, technical/security, and the\nreview critiques that remain materially unresolved) and record the correct\nresolution for each. Resolution philosophy: honest re-scoping — bound or\ncorrect claims where a problem is not solvable short-term; flag the real fix\nwhere it is.\n\n- New docs/research/problem_register_2026-06.md (sections A-E + scoped future\n  work), added to the Research nav.\n- review_response_2026-06.md: replace \"Already addressed\" with a 3-state legend\n  (Resolved / Honestly scoped / Open); relabel the LoCoMo/downstream row as\n  honestly scoped, not solved.\n- README: footnote clarifying the comparison-table checkmarks are feature\n  implementation, not head-to-head results; add a \"When NOT to use\" section\n  surfacing the four committed FAIL regimes (propagates to docs/index.md via the\n  positioning include).\n- 08_limitations.md: new 2.9 on single-seed runs / uncharacterized cross-seed\n  variance.\n- SECURITY.md: Known advisories subsection for the two unpatched optional/dev\n  CVEs (chromadb, torch); runtime wheel unaffected.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FnY4dRqa2xvyK4kQcLbqJQ\n\n* bench: add multi-seed robustness sweep; verify retrieval determinism\n\nResolve register item A7 (single-seed convention, cross-run variance\nuncharacterized) with a real, runnable harness instead of a scoping note.\n\nbenchmarks/realistic/multiseed_runner.py runs the realistic replay benchmark\nacross seeds {0,1,7,42,123}, each in an isolated subprocess invoking the\ncanonical runner, and reports cross-seed mean/stdev/min/max of top1_accuracy\nand the AFT-baseline delta. Committed result (hash embedder, v2): cross-seed\nstdev = spread = 0.0000 — per-query outcomes are identical across seeds, so\nretrieval is deterministic given a fixed dataset + deterministic embedder; only\nthe bootstrap CI resampling is seed-sensitive.\n\nSubprocess isolation is deliberate: running several full benchmarks in one\nprocess leaks accumulated global state across runs (surfaced while building this\nharness). Isolating each seed matches the canonical one-run-per-process model\nand makes the determinism verdict trustworthy.\n\n- Makefile: add bench-multiseed target.\n- 08_limitations.md 2.9: rewrite from \"uncharacterized\" to the verified result.\n- problem_register_2026-06.md: mark A7 resolved; trim future-work list to the\n  items that genuinely need external resources (human raters, LLM judge).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FnY4dRqa2xvyK4kQcLbqJQ\n\n* docs: pin down root cause of in-process benchmark timing jitter\n\nReplace the vague \"leaks global state\" wording with the precise mechanism: the\nengine stamps encode/retrieve with real wall-clock datetime.now and ACT-R decay\ntracks now-encoded_at, so back-to-back in-process runs can tip a near-tie query.\nThis is correct production behaviour, not a defect, and stays within the\nbootstrap CIs; the subprocess-per-seed harness already yields 0.0000 cross-seed\nvariance. Documented why an injected clock is deliberately not threaded through\nthe core API (disproportionate for a sub-CI, benchmark-only effect).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FnY4dRqa2xvyK4kQcLbqJQ\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-06-23T12:17:30Z",
          "tree_id": "3f8f9d6ff093e64fdb421ee23e6bc4d585c88b37",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/439f1f2b92dc220842f9ab87a7363de796be6cd3"
        },
        "date": 1782217379052,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 993.0921625363098,
            "unit": "iter/sec",
            "range": "stddev: 0.0005085015938606947",
            "extra": "mean: 1.0069558876046791 msec\nrounds: 1557"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 353.4257130119714,
            "unit": "iter/sec",
            "range": "stddev: 0.0020331710209179643",
            "extra": "mean: 2.8294489143921666 msec\nrounds: 3224"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 236.89206130896588,
            "unit": "iter/sec",
            "range": "stddev: 0.0036066243066810847",
            "extra": "mean: 4.221331835581237 msec\nrounds: 4300"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 160.1481928708633,
            "unit": "iter/sec",
            "range": "stddev: 0.005035505878095588",
            "extra": "mean: 6.244216572623817 msec\nrounds: 5976"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 348.16355161551394,
            "unit": "iter/sec",
            "range": "stddev: 0.0023693626210653388",
            "extra": "mean: 2.872213347318808 msec\nrounds: 3599"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 542.0727117747525,
            "unit": "iter/sec",
            "range": "stddev: 0.0005688785724503926",
            "extra": "mean: 1.8447709657362905 msec\nrounds: 788"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 232581.46270833685,
            "unit": "iter/sec",
            "range": "stddev: 5.176094044229631e-7",
            "extra": "mean: 4.299568797767971 usec\nrounds: 33722"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 62.2115058064637,
            "unit": "iter/sec",
            "range": "stddev: 0.0006032731310027459",
            "extra": "mean: 16.07419700000416 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.4991504209623812,
            "unit": "iter/sec",
            "range": "stddev: 0.010657681226086735",
            "extra": "mean: 667.0444713333362 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.04583236261216851,
            "unit": "iter/sec",
            "range": "stddev: 3.6911563691135445",
            "extra": "mean: 21.818643923333326 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 6308.629444212609,
            "unit": "iter/sec",
            "range": "stddev: 0.000006912433726406046",
            "extra": "mean: 158.51303501703953 usec\nrounds: 3484"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1589.6212150024423,
            "unit": "iter/sec",
            "range": "stddev: 0.0001044323542172591",
            "extra": "mean: 629.0806832233071 usec\nrounds: 1067"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 643.3897662474077,
            "unit": "iter/sec",
            "range": "stddev: 0.00005034351772422393",
            "extra": "mean: 1.554267805396616 msec\nrounds: 519"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 664.4716484519918,
            "unit": "iter/sec",
            "range": "stddev: 0.0007185379164682092",
            "extra": "mean: 1.5049551058042625 msec\nrounds: 1361"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 23092.61561740162,
            "unit": "iter/sec",
            "range": "stddev: 0.000006362777023900406",
            "extra": "mean: 43.30388625385694 usec\nrounds: 9319"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 23349.982707991083,
            "unit": "iter/sec",
            "range": "stddev: 0.000002381309618090347",
            "extra": "mean: 42.826584178058916 usec\nrounds: 15093"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 3827.759277321531,
            "unit": "iter/sec",
            "range": "stddev: 0.000009113142090497336",
            "extra": "mean: 261.2494484500991 usec\nrounds: 1775"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 3774.260137958613,
            "unit": "iter/sec",
            "range": "stddev: 0.00004520640261591198",
            "extra": "mean: 264.9525902951858 usec\nrounds: 2741"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 2184.7895605001245,
            "unit": "iter/sec",
            "range": "stddev: 0.00002400416032346296",
            "extra": "mean: 457.70998638930155 usec\nrounds: 1396"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 692.9631716066688,
            "unit": "iter/sec",
            "range": "stddev: 0.00004663767281672773",
            "extra": "mean: 1.4430781331155758 msec\nrounds: 616"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 27.77394258454256,
            "unit": "iter/sec",
            "range": "stddev: 0.001892449801446709",
            "extra": "mean: 36.00497109677705 msec\nrounds: 31"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 881.2093577623452,
            "unit": "iter/sec",
            "range": "stddev: 0.000018958891390050505",
            "extra": "mean: 1.1348041089115304 msec\nrounds: 808"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 729.0338723580703,
            "unit": "iter/sec",
            "range": "stddev: 0.000024372795992077898",
            "extra": "mean: 1.3716783786266142 msec\nrounds: 655"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 592.9709948083607,
            "unit": "iter/sec",
            "range": "stddev: 0.00003889125135566625",
            "extra": "mean: 1.686423128205765 msec\nrounds: 546"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 387.4979301590166,
            "unit": "iter/sec",
            "range": "stddev: 0.00006345747677414272",
            "extra": "mean: 2.5806589459449043 msec\nrounds: 333"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1787.1417383370822,
            "unit": "iter/sec",
            "range": "stddev: 0.000018043347306323627",
            "extra": "mean: 559.5527084105204 usec\nrounds: 1629"
          }
        ]
      }
    ]
  }
}