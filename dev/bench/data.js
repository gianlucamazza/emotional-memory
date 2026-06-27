window.BENCHMARK_DATA = {
  "lastUpdate": 1782569993999,
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
          "id": "7baabce4cc673f1511a44ceece916a3912ca585b",
          "message": "feat(bench): add Addendum W — affine arousal calibration (pre-registered) (#82)\n\n* feat(bench): add Addendum W — affine arousal calibration (pre-registered)\n\nFollows up the Addendum V arousal MAE caveat: direct-VAD arousal wins on\ncorrelation (r 0.58 vs 0.23) but loses on MAE (0.193 vs 0.112), driven by a\nsystematic under-prediction (bias -0.093) and a wider spread than EmoBank's\nnarrow arousal range. Pearson r is affine-invariant, so a fit\narousal_cal = a*arousal_direct + b can fix scale/offset while keeping r.\n\nThis is the pre-registration + deterministic offline harness (no result yet —\nregistered before observing, per project discipline):\n\n- preregistration_addendum_w_arousal_calibration.md: Hw1 (cal < raw MAE),\n  Hw2 (cal < scherer MAE → dominates on both axes), Gw (slope>0 & r preserved);\n  two pre-declared protocols (native EmoBank split; 5-fold CV); decision rule.\n- appraisal_vad/runner.py: add --dump-predictions so the single Addendum V LLM\n  pass persists paired per-item predictions; downstream analysis needs no LLM.\n- arousal_calibration/runner.py: deterministic affine fit + verdicts over the\n  dump; Makefile targets bench-arousal-calibration{,-dump}.\n- tests/test_arousal_calibration.py: synthetic, no-LLM unit tests (param\n  recovery, leakage-free OOF, MAE reduction, determinism).\n\nExecution awaits one LLM dump pass (EMOTIONAL_MEMORY_LLM_API_KEY). ruff clean;\nmake typecheck green (src scope); 5/5 new tests pass.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* fix(bench): make arousal_calibration self-contained (no LLM-dep import)\n\nThe offline calibration runner imported `_pearson` from\n`human_gold_appraisal.runner`, which transitively pulls `tqdm` (bench extra) and\n`httpx` (llm-test extra). In the core CI test env (neither extra installed) this\nbroke collection of `tests/test_arousal_calibration.py` → all Test jobs failed.\n\nInline a numpy-only `_pearson` so the deterministic module depends only on numpy\nand `benchmarks.common.statistics`. Import chain verified to pull neither tqdm nor\nhttpx; 5/5 tests pass; ruff clean.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T16:12:58+02:00",
          "tree_id": "934d3b64e4f24983206d09a8f83468b093b47a08",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7baabce4cc673f1511a44ceece916a3912ca585b"
        },
        "date": 1782569992964,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 564.308532318824,
            "unit": "iter/sec",
            "range": "stddev: 0.0008636775919009117",
            "extra": "mean: 1.772080241088785 msec\nrounds: 1543"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 388.7455476654405,
            "unit": "iter/sec",
            "range": "stddev: 0.0013865858982227187",
            "extra": "mean: 2.5723767281847127 msec\nrounds: 1994"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 303.72453291046685,
            "unit": "iter/sec",
            "range": "stddev: 0.0020076685514178985",
            "extra": "mean: 3.2924571170376415 msec\nrounds: 2606"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 166.20770392315268,
            "unit": "iter/sec",
            "range": "stddev: 0.004251821284728735",
            "extra": "mean: 6.016568284117306 msec\nrounds: 4294"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 454.0521779567642,
            "unit": "iter/sec",
            "range": "stddev: 0.0010105066421421682",
            "extra": "mean: 2.2023900523944238 msec\nrounds: 1775"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 376.75425823894443,
            "unit": "iter/sec",
            "range": "stddev: 0.00027565763420997545",
            "extra": "mean: 2.6542500267263915 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137544.155467734,
            "unit": "iter/sec",
            "range": "stddev: 9.137607309175262e-7",
            "extra": "mean: 7.270392526671818 usec\nrounds: 42150"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.561844466977874,
            "unit": "iter/sec",
            "range": "stddev: 0.0004531285525468374",
            "extra": "mean: 30.71079100000418 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8314448686768391,
            "unit": "iter/sec",
            "range": "stddev: 0.003284905826510355",
            "extra": "mean: 1.2027255656666682 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.030463551502010817,
            "unit": "iter/sec",
            "range": "stddev: 0.7307326433335978",
            "extra": "mean: 32.826113525666656 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3429.596293014388,
            "unit": "iter/sec",
            "range": "stddev: 0.000016331500310265524",
            "extra": "mean: 291.5795080712157 usec\nrounds: 2354"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 899.171063826158,
            "unit": "iter/sec",
            "range": "stddev: 0.00001958010337116818",
            "extra": "mean: 1.1121354325447197 msec\nrounds: 719"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.4330522758817,
            "unit": "iter/sec",
            "range": "stddev: 0.00010656224683598416",
            "extra": "mean: 2.789921280000461 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 493.1996375616929,
            "unit": "iter/sec",
            "range": "stddev: 0.0004505742688342463",
            "extra": "mean: 2.0275765102826395 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12315.572407806872,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037684508958576842",
            "extra": "mean: 81.19801231212749 usec\nrounds: 7391"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12070.931665845033,
            "unit": "iter/sec",
            "range": "stddev: 0.000003894324106071237",
            "extra": "mean: 82.84364684372474 usec\nrounds: 9030"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2018.913890107604,
            "unit": "iter/sec",
            "range": "stddev: 0.00001315583266563125",
            "extra": "mean: 495.31582545439915 usec\nrounds: 1100"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2011.3734105570252,
            "unit": "iter/sec",
            "range": "stddev: 0.000013346127487095056",
            "extra": "mean: 497.17272523905064 usec\nrounds: 1565"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1164.6498308149944,
            "unit": "iter/sec",
            "range": "stddev: 0.000017009931192994403",
            "extra": "mean: 858.6271800685565 usec\nrounds: 883"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.72963454028394,
            "unit": "iter/sec",
            "range": "stddev: 0.00008303007502863707",
            "extra": "mean: 2.6128104796513885 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.078518561437214,
            "unit": "iter/sec",
            "range": "stddev: 0.0014919536268636826",
            "extra": "mean: 47.44166422727083 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.8253676666792,
            "unit": "iter/sec",
            "range": "stddev: 0.00007911798996157221",
            "extra": "mean: 2.1284504175803822 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 385.42241568135654,
            "unit": "iter/sec",
            "range": "stddev: 0.00008482616312545691",
            "extra": "mean: 2.5945558932585238 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 318.0634345026067,
            "unit": "iter/sec",
            "range": "stddev: 0.00011089409256762189",
            "extra": "mean: 3.1440269189189194 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 206.86364095410593,
            "unit": "iter/sec",
            "range": "stddev: 0.00019656859020258916",
            "extra": "mean: 4.834102287805408 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 950.8025417045424,
            "unit": "iter/sec",
            "range": "stddev: 0.000019817507777715095",
            "extra": "mean: 1.051743086642637 msec\nrounds: 831"
          }
        ]
      }
    ]
  }
}