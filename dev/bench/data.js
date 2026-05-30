window.BENCHMARK_DATA = {
  "lastUpdate": 1780147826900,
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
          "id": "0ff4130dac8bdba5ceac140859b1121d628eebec",
          "message": "chore: benchmark tooling wiring and project hygiene (#39)\n\n* fix(bench): load .env in diagnostics, hg1 and locomo runners\n\nThe realistic/comparative runners already load .env via python-dotenv when\ninvoked outside make; the appraisal_diagnostics, appraisal_confound (Hg1) and\nlocomo runners did not, so a direct 'python -m ...' run failed with\n'EMOTIONAL_MEMORY_LLM_API_KEY is not set' despite the key being in .env. Mirror\nthe existing guarded-import pattern.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* build(make): add bench-appraisal-diagnostics target and gate locomo on llm-config-strict\n\nThe appraisal_diagnostics runner was the only benchmark without a Makefile\ntarget; add bench-appraisal-diagnostics (gated on llm-config-strict) plus a\nkeyless bench-appraisal-diagnostics-dry, wired into .PHONY and help. Add the\nllm-config-strict prerequisite to bench-locomo/-routing/-pareto for parity with\nthe other LLM-backed targets.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): add appraisal_diagnostics README and correct protocol dataset counts\n\nAdd the missing README (every other sub-benchmark has one). Append a post-freeze\nnote to protocol.md: the runner iterates all 125 scenarios / 750 events of\nrealistic_recall_v3.json, not the '50 / ~250' in the frozen prose. Frozen\nthresholds/dataset/decision-tree unchanged.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* chore(repo): add editorconfig and code of conduct, document LLM bench deps\n\nAdd .editorconfig (utf-8/LF/trim, 4-space Python, tab Makefiles) and\nCODE_OF_CONDUCT.md (Contributor Covenant 2.1, contact info@gianlucamazza.it).\nIgnore *.log. Note in CONTRIBUTING that real-LLM tests/benchmarks need\nmake install-llm-test (httpx) and, for direct module runs, make install-dotenv.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* ci(pre-commit): add mypy pre-push and conventional commit-msg hooks\n\nAdd a local mypy --strict hook on pre-push (kept off per-commit to stay fast)\nand conventional-pre-commit on commit-msg (mirrors pr-title.yml in CI).\ndefault_install_hook_types wires pre-commit, pre-push and commit-msg via a\nsingle 'pre-commit install'.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(changelog): record bench tooling and project-hygiene changes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* ci(pre-commit): align ruff hook to 0.15.x and fix CoC trailing newline\n\nThe ruff hook was pinned to v0.11.12 while the project's ruff (and CI's\n`uv run ruff`) is 0.15.x, where UP038 was removed — the stale pin failed on\ncode that `make lint` accepts. Pin the hook to v0.15.15 to match. Also drop the\ntrailing blank line in CODE_OF_CONDUCT.md flagged by end-of-file-fixer.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): add WP-1a appraisal diagnostics results (gpt-5-mini, N=750)\n\nFirst execution of the appraisal-diagnostics runner against oracle affect on\nrealistic_recall_v3.json. Verdict P1d (bias AND variance), but with nuance: the\nappraisal signal is not absent — valence Pearson r=0.81 and sign accuracy=0.86 —\nit is mis-calibrated (valence bias +0.19, std 0.36; arousal bias -0.14, r=0.37).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T15:22:50+02:00",
          "tree_id": "ae23b8fe533f9f66f005d66e126ea958fc17ffa9",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/0ff4130dac8bdba5ceac140859b1121d628eebec"
        },
        "date": 1780147826483,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 598.1330273094972,
            "unit": "iter/sec",
            "range": "stddev: 0.0007907135931604982",
            "extra": "mean: 1.6718688892639286 msec\nrounds: 1481"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 429.9703851960182,
            "unit": "iter/sec",
            "range": "stddev: 0.001198867740646803",
            "extra": "mean: 2.3257415729785955 msec\nrounds: 1843"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 313.3108114679392,
            "unit": "iter/sec",
            "range": "stddev: 0.0020718869817764984",
            "extra": "mean: 3.191718777002144 msec\nrounds: 2435"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 159.49378165164683,
            "unit": "iter/sec",
            "range": "stddev: 0.005022511753503915",
            "extra": "mean: 6.269836915548956 msec\nrounds: 4026"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 412.1610499202051,
            "unit": "iter/sec",
            "range": "stddev: 0.0012541819638854843",
            "extra": "mean: 2.4262360555263562 msec\nrounds: 1891"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 365.59782863758545,
            "unit": "iter/sec",
            "range": "stddev: 0.000505311529129545",
            "extra": "mean: 2.735246004404728 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137689.4769756669,
            "unit": "iter/sec",
            "range": "stddev: 9.276674666336702e-7",
            "extra": "mean: 7.262719141396146 usec\nrounds: 17799"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.80502624614114,
            "unit": "iter/sec",
            "range": "stddev: 0.00038346091794981926",
            "extra": "mean: 30.483133666677986 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8217039060854816,
            "unit": "iter/sec",
            "range": "stddev: 0.028619030424926917",
            "extra": "mean: 1.2169833836666346 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02308132017796448,
            "unit": "iter/sec",
            "range": "stddev: 2.6192221154796966",
            "extra": "mean: 43.325078127666664 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3550.487347879325,
            "unit": "iter/sec",
            "range": "stddev: 0.000008841635668094492",
            "extra": "mean: 281.65147542274474 usec\nrounds: 2360"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 910.7534727521564,
            "unit": "iter/sec",
            "range": "stddev: 0.000049914849699507136",
            "extra": "mean: 1.0979919702948313 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 365.2730867749379,
            "unit": "iter/sec",
            "range": "stddev: 0.00006452528513692247",
            "extra": "mean: 2.7376777436004955 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 496.92308999485556,
            "unit": "iter/sec",
            "range": "stddev: 0.00047969748458920905",
            "extra": "mean: 2.012383847992156 msec\nrounds: 796"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12348.8542880953,
            "unit": "iter/sec",
            "range": "stddev: 0.000003287116546272139",
            "extra": "mean: 80.97917237261701 usec\nrounds: 7536"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12158.084240665603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038125991598268365",
            "extra": "mean: 82.24980023211735 usec\nrounds: 7759"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2038.5078533136168,
            "unit": "iter/sec",
            "range": "stddev: 0.00001165941696584683",
            "extra": "mean: 490.5548920866255 usec\nrounds: 1112"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2015.770855991732,
            "unit": "iter/sec",
            "range": "stddev: 0.00004362197026352023",
            "extra": "mean: 496.0881327495994 usec\nrounds: 1710"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1199.5491353140724,
            "unit": "iter/sec",
            "range": "stddev: 0.000019566434993782733",
            "extra": "mean: 833.6465514921778 usec\nrounds: 903"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 393.5893342374837,
            "unit": "iter/sec",
            "range": "stddev: 0.00005306984977505678",
            "extra": "mean: 2.540719254848051 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.573199437782844,
            "unit": "iter/sec",
            "range": "stddev: 0.00040857777414953693",
            "extra": "mean: 51.09026774997574 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 463.8981623997841,
            "unit": "iter/sec",
            "range": "stddev: 0.00006339841347068589",
            "extra": "mean: 2.1556455296716766 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 391.03993223375744,
            "unit": "iter/sec",
            "range": "stddev: 0.00005687389646937798",
            "extra": "mean: 2.5572835855602998 msec\nrounds: 374"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 309.3623354432025,
            "unit": "iter/sec",
            "range": "stddev: 0.00016875417059068984",
            "extra": "mean: 3.2324555559336843 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.7261543944836,
            "unit": "iter/sec",
            "range": "stddev: 0.00010839930759007779",
            "extra": "mean: 4.790966435907416 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 978.3777531366796,
            "unit": "iter/sec",
            "range": "stddev: 0.000019205830484943842",
            "extra": "mean: 1.0221001006962795 msec\nrounds: 864"
          }
        ]
      }
    ]
  }
}