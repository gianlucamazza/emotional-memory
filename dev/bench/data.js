window.BENCHMARK_DATA = {
  "lastUpdate": 1780229932699,
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
          "id": "66dd3cc28971ebd4d381bc5038d9ae6def721fec",
          "message": "docs(bench): Addendum P — Hg1 re-run FAIL on leakage-free set (recalibrated M1) (#47)\n\n* docs(bench): pre-register Addendum P (Hg1 re-run) + dataset generator\n\nPre-registration for the Hg1 re-run with the recalibrated SEC->affect mapping\n(M1, Addendum O). Hp1 confirmatory (aft_llm_dual.top1 > naive_cosine.top1 on a\nleakage-free affect-free set), identical statistic to Hg1 (paired bootstrap\nn=10000, seed=0, alpha=0.05, threshold 0.05); Hp2/Hp3 exploratory.\n\nLeakage constraint: realistic_recall_v3_noAF (s01-s50) is contained in v3, the\nAddendum O calibration set, so the re-run needs scenarios disjoint from v3.\n\nAdds benchmarks/datasets/generate_v4_noAF.py: scripted, author-blind generator\n(one structured-JSON scenario per gpt-5-mini call, pNN_ namespace, 5 challenge\ntypes, no preset affect). Python fixes ids + target/confound structure so the\nLLM only authors text; each scenario validated against the noAF schema +\n_validate_dataset before inclusion. Lazy imports keep --dry-run dependency-free.\n\nNo LLM calls yet (generation + run are gated). make check green.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* chore(bench): add realistic_recall_v4_noAF (Addendum P frozen dataset)\n\nLLM-generated (gpt-5-mini via generate_v4_noAF.py), schema-validated affect-free\nscenario set for the Hg1 re-run. 40 scenarios / 240 events / 160 queries, pNN_\nnamespace fully disjoint from realistic_recall_v3 (the Addendum O calibration\nset): zero overlap on scenario ids, memory ids, and event text — leakage-free.\n\nBalanced across the 5 challenge types (32 each); no preset valence/arousal on\nevents, no state on queries (affect inferred by the LLM). Dry-validated with the\nhash embedder (non-LLM systems): naive_cosine top1=0.344, aft_neutral top1=0.300\n(N=160) — non-trivial baseline comparable to the original Hg1 set (0.325).\nFrozen and committed before any LLM benchmark run (author-blind, per protocol).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): close Addendum P (Hg1 re-run) — FAIL on leakage-free set\n\nRe-ran Hg1 (AFT LLM dual-path appraisal vs naive cosine on affect-free\nretrieval) with the Addendum O recalibrated SEC->affect mapping (M1) on a\nnew leakage-free dataset realistic_recall_v4_noAF (40 scenarios / 160\nqueries, pNN_ namespace, zero overlap with realistic_recall_v3).\n\nHp1 FAIL: aft_llm_dual top1 0.800 vs naive_cosine 0.887, delta=-0.0875,\nCI[-0.144,-0.031], one-tailed p=0.0018, d=-0.242. Naive cosine is\nsignificantly ahead; recalibration did not rescue affect-free retrieval\n(gap widened vs Hg1's -0.010 and became significant).\n\nExploratory: Hp2 dual>neutral PASS (delta=+0.056, p=0.030) — LLM affect\ncarries signal; Hp3 dual>sync PASS (delta=+0.512, d=0.95) — deferred\ndual-path essential (synchronous appraisal collapses to top1=0.287). Net:\nbetter-calibrated affect is a distractor on affect-free queries even though\nit is not noise.\n\nClaim appraisal_llm_real_dual_path stays falsified; updated current_evidence\n+ allowed_public_wording in claim matrix and 09_current_evidence.md; Hd1/Hd2\n(oracle-affect) and Addendum O (mapping calibration) unaffected.\n\nAdds closure doc + result artifacts (results.hg1_v4.{json,md,protocol.json}).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: propagate Addendum P (Hg1 re-run FAIL) to limitations + roadmap\n\nThe Addendum P closure (ffb5181) updated the claim matrix and current-evidence\npage; this propagates the result to the two docs that still described the\nappraisal study as pending.\n\n- 08_limitations.md §2.4: replace the stale \"Addendum G (future study) will\n  address this / Until Addendum G is executed\" paragraph and the \"architecture\n  advantage under automatic appraisal is not yet established\" sentence with the\n  actual outcomes — Addendum G FAIL, Addendum O calibration PASS, Addendum P\n  leakage-free re-run FAIL (dual-path 0.800 vs naive_cosine 0.887, p=0.0018,\n  d=-0.242; Hp2/Hp3 isolate that the signal is real but a net distractor on\n  affect-free queries). Oracle-affect scope stated as a hard boundary.\n- 09_current_evidence.md: theory_faithful_operationalization no longer lists\n  \"Addendum G pending\"; the dual-path advantage is recorded as falsified.\n- ROADMAP.md: add a closed v0.11.x research section documenting Addenda N\n  (prompt, FAIL/reverted), O (mapping M1, calibration PASS), and P (Hg1 re-run,\n  FAIL); next angle noted as affect-aware routing.\n\nmake check green.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: clear remaining \"Addendum G pending\" refs from theory-fidelity claim\n\nFollow-up to 21eda56: the theory_faithful_operationalization claim still\ndescribed the automatic-appraisal dual-path study as pending in both the\ncurrent-evidence table (09) and the claim matrix JSON (not_yet_shown +\nnext_study). Updated to record it as falsified (Addendum G/P), pointing to\nthe appraisal_llm_real_dual_path claim. No status change.\n\nmake check green.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-31T14:13:19+02:00",
          "tree_id": "aec4d698ebb758175722cce387452e2ce4fef44f",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/66dd3cc28971ebd4d381bc5038d9ae6def721fec"
        },
        "date": 1780229931525,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 551.1873920027363,
            "unit": "iter/sec",
            "range": "stddev: 0.0008222278437225997",
            "extra": "mean: 1.8142650113358103 msec\nrounds: 1588"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 441.402159762738,
            "unit": "iter/sec",
            "range": "stddev: 0.0010800546512315721",
            "extra": "mean: 2.265507718715103 msec\nrounds: 1806"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 363.21737031630033,
            "unit": "iter/sec",
            "range": "stddev: 0.0015697938245004285",
            "extra": "mean: 2.75317229219839 msec\nrounds: 2269"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 213.69638731964875,
            "unit": "iter/sec",
            "range": "stddev: 0.0029742255934185257",
            "extra": "mean: 4.679536292320151 msec\nrounds: 3763"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 453.2194783875499,
            "unit": "iter/sec",
            "range": "stddev: 0.0009725236717790235",
            "extra": "mean: 2.2064365008268596 msec\nrounds: 1813"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.4964576440487,
            "unit": "iter/sec",
            "range": "stddev: 0.00024243017098522313",
            "extra": "mean: 2.614403297116546 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 140947.66036733764,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011323691203744177",
            "extra": "mean: 7.094832205045483 usec\nrounds: 51849"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.882342032136098,
            "unit": "iter/sec",
            "range": "stddev: 0.00039778964756565345",
            "extra": "mean: 34.623231000011856 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8264443416934257,
            "unit": "iter/sec",
            "range": "stddev: 0.00800922118301124",
            "extra": "mean: 1.210002839333318 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03615271755052742,
            "unit": "iter/sec",
            "range": "stddev: 2.278362553516635",
            "extra": "mean: 27.660437935333334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3248.4208818227844,
            "unit": "iter/sec",
            "range": "stddev: 0.000012727276266322693",
            "extra": "mean: 307.8418826808153 usec\nrounds: 2327"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 874.1314366932929,
            "unit": "iter/sec",
            "range": "stddev: 0.000123590547351315",
            "extra": "mean: 1.143992720114093 msec\nrounds: 711"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 367.90262372676045,
            "unit": "iter/sec",
            "range": "stddev: 0.00003070639319512166",
            "extra": "mean: 2.7181105420511904 msec\nrounds: 321"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 502.71884502498995,
            "unit": "iter/sec",
            "range": "stddev: 0.00043853117107136815",
            "extra": "mean: 1.989183437016948 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10487.634095342633,
            "unit": "iter/sec",
            "range": "stddev: 0.000006276125139922053",
            "extra": "mean: 95.35038988861002 usec\nrounds: 6725"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10597.242204132175,
            "unit": "iter/sec",
            "range": "stddev: 0.000007453802093165107",
            "extra": "mean: 94.36417331389016 usec\nrounds: 8199"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1824.830250930449,
            "unit": "iter/sec",
            "range": "stddev: 0.000014934710389258453",
            "extra": "mean: 547.996176351262 usec\nrounds: 482"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1820.737893101437,
            "unit": "iter/sec",
            "range": "stddev: 0.000010211243181062277",
            "extra": "mean: 549.227872825014 usec\nrounds: 1667"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 997.9653865825899,
            "unit": "iter/sec",
            "range": "stddev: 0.00014467337551803879",
            "extra": "mean: 1.0020387615089312 msec\nrounds: 717"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 379.48555496548653,
            "unit": "iter/sec",
            "range": "stddev: 0.00003203027029340708",
            "extra": "mean: 2.6351464157600097 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.035281979517386,
            "unit": "iter/sec",
            "range": "stddev: 0.0010262645547214292",
            "extra": "mean: 45.3817655217454 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 460.31812278550836,
            "unit": "iter/sec",
            "range": "stddev: 0.00010227805174307826",
            "extra": "mean: 2.1724106666683727 msec\nrounds: 447"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 377.5527907323628,
            "unit": "iter/sec",
            "range": "stddev: 0.000032306545151128035",
            "extra": "mean: 2.6486362292813075 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.62356261560035,
            "unit": "iter/sec",
            "range": "stddev: 0.00004744097818588852",
            "extra": "mean: 3.2827401512005956 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 189.592312109394,
            "unit": "iter/sec",
            "range": "stddev: 0.00021581382387847827",
            "extra": "mean: 5.274475472523401 msec\nrounds: 182"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 827.0109828510735,
            "unit": "iter/sec",
            "range": "stddev: 0.00002927816357278451",
            "extra": "mean: 1.209173784551877 msec\nrounds: 738"
          }
        ]
      }
    ]
  }
}