window.BENCHMARK_DATA = {
  "lastUpdate": 1784185678520,
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
          "id": "4fea2983e3b634c8371617801227678c08235914",
          "message": "fix: review defects + Claude Code collaboration guide (#121)\n\n* docs: add Claude Code collaboration guide\n\nAdd docs/contributing/claude-code-guide.md capturing the guiding\nprinciples (theory-fidelity over raw performance), a base system prompt,\nreusable prompts for common tasks (review, feature, refactor, debug),\nand the pre-PR checklist for LLM-assisted work on the project.\n\nWire the page into the docs nav under Contributing and note the addition\nin the CHANGELOG.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011m3w1uTjKtj5XVUSDLVsgh\n\n* docs: translate Claude Code guide to English\n\nRewrite the collaboration guide in English for consistency with the rest\nof the documentation site.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011m3w1uTjKtj5XVUSDLVsgh\n\n* docs: link Claude Code guide from CONTRIBUTING\n\nAdd a \"Working with Claude Code\" section to CONTRIBUTING.md and a\nContents entry pointing to docs/contributing/claude-code-guide.md, so the\ncollaboration guide is discoverable from the main contribution workflow.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011m3w1uTjKtj5XVUSDLVsgh\n\n* fix: correctness & fidelity defects from 2026-07 project review\n\nStructured review (per docs/contributing/claude-code-guide.md) surfaced a\nbatch of correctness and theory-fidelity defects. Fixes, each with a\nregression test in tests/test_review_fixes.py:\n\nCorrectness\n- encode_batch honored neither enable_appraisal nor enable_resonance,\n  diverging from single encode() and corrupting Layer-4/5 ablations\n  (both sync and async engines).\n- categorize_affect(CoreAffect.neutral()) returned high-confidence\n  disgust/loathing; near-baseline states are now uncategorised.\n- decay arousal floor could raise a weak memory above its initial\n  strength; the floor is now clamped to the initial value.\n- appraisal: reject NaN/Infinity JSON (bypassed clamping); missing SEC\n  dimensions are a degraded result, not silently neutral-filled; count\n  every neutral fallback via LLMAppraisalEngine.fallback_count.\n- GenericAppraisalVector.dimensions is now read-only (was a shared,\n  mutable dict returned from the LRU cache).\n- async retrieve reads a single state snapshot (no torn read); state\n  persistence saves the snapshot captured under the lock; close() uses\n  callable() uniformly; encode log ordering matches sync.\n- AffectiveState is now frozen.\n\nCleanups & docs\n- remove dead _APPRAISAL_JSON_SCHEMA; default appraisal prompt derives\n  from SCHERER_CPM_SCHEMA; NUL-delimit cache-key components.\n- correct docstrings/citations (Pearce-Hall LR, mood dominance EMA,\n  Mehrabian & Russell 1974, Anderson & Schooler 1991, Collins & Loftus\n  1975) and document known research-level limitations rather than\n  silently changing calibrated behaviour.\n\nGates: ruff clean, mypy strict clean, full suite green, fidelity\nbenchmark (127 invariants) green.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011m3w1uTjKtj5XVUSDLVsgh\n\n* fix(async): serialize state persist under lock\n\nAddress CodeRabbit review on PR #121: await _persist_state_async inside\n_state_lock so concurrent encode to_thread saves cannot reorder and\noverwrite a newer affective state with an older snapshot. Also skip LRU\ncaching of error fallbacks so a transient LLM/parse failure does not pin\nneutral for that key until eviction, with regression coverage.\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-07-16T09:01:30+02:00",
          "tree_id": "a7782018870ab8f2c4cbedb3ee9a309ff1e645ab",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/4fea2983e3b634c8371617801227678c08235914"
        },
        "date": 1784185676491,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 631.8111108868519,
            "unit": "iter/sec",
            "range": "stddev: 0.0007184190791367401",
            "extra": "mean: 1.5827515261584018 msec\nrounds: 1338"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 440.5705781199138,
            "unit": "iter/sec",
            "range": "stddev: 0.001093717710322829",
            "extra": "mean: 2.2697838885823685 msec\nrounds: 1813"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 368.52286036654124,
            "unit": "iter/sec",
            "range": "stddev: 0.0016386957616749622",
            "extra": "mean: 2.713535868590017 msec\nrounds: 2184"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 264.6886566607458,
            "unit": "iter/sec",
            "range": "stddev: 0.0026367604442137154",
            "extra": "mean: 3.7780236320505054 msec\nrounds: 3014"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 443.88808403038786,
            "unit": "iter/sec",
            "range": "stddev: 0.0010165619016306356",
            "extra": "mean: 2.252820104834221 msec\nrounds: 1841"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.7120200485183,
            "unit": "iter/sec",
            "range": "stddev: 0.0002908192280070936",
            "extra": "mean: 2.6266572825112244 msec\nrounds: 446"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135337.9931730346,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012137712051640334",
            "extra": "mean: 7.38890814437793 usec\nrounds: 45430"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.854691544794,
            "unit": "iter/sec",
            "range": "stddev: 0.000687424073869608",
            "extra": "mean: 35.9005950000153 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8216230947163331,
            "unit": "iter/sec",
            "range": "stddev: 0.00502280439607323",
            "extra": "mean: 1.2171030809999952 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.028415437601484385,
            "unit": "iter/sec",
            "range": "stddev: 2.8798628747787642",
            "extra": "mean: 35.1921379506667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3122.8642848935733,
            "unit": "iter/sec",
            "range": "stddev: 0.000016886846443350914",
            "extra": "mean: 320.2188467931067 usec\nrounds: 2043"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 850.0394365632496,
            "unit": "iter/sec",
            "range": "stddev: 0.00004955689195337341",
            "extra": "mean: 1.1764160072890832 msec\nrounds: 686"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 347.321837741618,
            "unit": "iter/sec",
            "range": "stddev: 0.000040078257963115615",
            "extra": "mean: 2.8791739860133028 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.1053968528445,
            "unit": "iter/sec",
            "range": "stddev: 0.000442029968594945",
            "extra": "mean: 2.0320850094213303 msec\nrounds: 743"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10368.184003545915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061569398886885075",
            "extra": "mean: 96.44890558057229 usec\nrounds: 4946"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10303.672062810097,
            "unit": "iter/sec",
            "range": "stddev: 0.000005767255635031162",
            "extra": "mean: 97.05277826236176 usec\nrounds: 7563"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1823.7146507907942,
            "unit": "iter/sec",
            "range": "stddev: 0.000013097730392912869",
            "extra": "mean: 548.3313957950509 usec\nrounds: 1094"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1812.9247200790373,
            "unit": "iter/sec",
            "range": "stddev: 0.00005650313912204367",
            "extra": "mean: 551.594883628926 usec\nrounds: 1521"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 994.9640026549706,
            "unit": "iter/sec",
            "range": "stddev: 0.00003156583676346639",
            "extra": "mean: 1.0050614869800227 msec\nrounds: 768"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.98021087576814,
            "unit": "iter/sec",
            "range": "stddev: 0.00005265818727577715",
            "extra": "mean: 2.7028472621087825 msec\nrounds: 351"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.496894736912516,
            "unit": "iter/sec",
            "range": "stddev: 0.0008590759678546107",
            "extra": "mean: 46.518346590909744 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 466.2548816702965,
            "unit": "iter/sec",
            "range": "stddev: 0.00004024639063282272",
            "extra": "mean: 2.1447496622826385 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 373.5800783652207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000485151443284655",
            "extra": "mean: 2.6768022651956733 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.96465651470777,
            "unit": "iter/sec",
            "range": "stddev: 0.00008255845099291507",
            "extra": "mean: 3.300715045457634 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.28039853827747,
            "unit": "iter/sec",
            "range": "stddev: 0.00016282036038127218",
            "extra": "mean: 5.1738303913004335 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 848.6437496746133,
            "unit": "iter/sec",
            "range": "stddev: 0.00003058164851438994",
            "extra": "mean: 1.178350751282172 msec\nrounds: 780"
          }
        ]
      }
    ]
  }
}