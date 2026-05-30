window.BENCHMARK_DATA = {
  "lastUpdate": 1780150992135,
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
          "id": "5a1a45b38aaf7d78b1697e7f2ec2fe405eac4670",
          "message": "docs(appraisal): diagnose appraisal mis-calibration; prompt-calibration attempt fails (Addendum N) (#40)\n\n* fix(bench): load .env in diagnostics, hg1 and locomo runners\n\nThe realistic/comparative runners already load .env via python-dotenv when\ninvoked outside make; the appraisal_diagnostics, appraisal_confound (Hg1) and\nlocomo runners did not, so a direct 'python -m ...' run failed with\n'EMOTIONAL_MEMORY_LLM_API_KEY is not set' despite the key being in .env. Mirror\nthe existing guarded-import pattern.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* build(make): add bench-appraisal-diagnostics target and gate locomo on llm-config-strict\n\nThe appraisal_diagnostics runner was the only benchmark without a Makefile\ntarget; add bench-appraisal-diagnostics (gated on llm-config-strict) plus a\nkeyless bench-appraisal-diagnostics-dry, wired into .PHONY and help. Add the\nllm-config-strict prerequisite to bench-locomo/-routing/-pareto for parity with\nthe other LLM-backed targets.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): add appraisal_diagnostics README and correct protocol dataset counts\n\nAdd the missing README (every other sub-benchmark has one). Append a post-freeze\nnote to protocol.md: the runner iterates all 125 scenarios / 750 events of\nrealistic_recall_v3.json, not the '50 / ~250' in the frozen prose. Frozen\nthresholds/dataset/decision-tree unchanged.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* chore(repo): add editorconfig and code of conduct, document LLM bench deps\n\nAdd .editorconfig (utf-8/LF/trim, 4-space Python, tab Makefiles) and\nCODE_OF_CONDUCT.md (Contributor Covenant 2.1, contact info@gianlucamazza.it).\nIgnore *.log. Note in CONTRIBUTING that real-LLM tests/benchmarks need\nmake install-llm-test (httpx) and, for direct module runs, make install-dotenv.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* ci(pre-commit): add mypy pre-push and conventional commit-msg hooks\n\nAdd a local mypy --strict hook on pre-push (kept off per-commit to stay fast)\nand conventional-pre-commit on commit-msg (mirrors pr-title.yml in CI).\ndefault_install_hook_types wires pre-commit, pre-push and commit-msg via a\nsingle 'pre-commit install'.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(changelog): record bench tooling and project-hygiene changes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* ci(pre-commit): align ruff hook to 0.15.x and fix CoC trailing newline\n\nThe ruff hook was pinned to v0.11.12 while the project's ruff (and CI's\n`uv run ruff`) is 0.15.x, where UP038 was removed — the stale pin failed on\ncode that `make lint` accepts. Pin the hook to v0.15.15 to match. Also drop the\ntrailing blank line in CODE_OF_CONDUCT.md flagged by end-of-file-fixer.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): add WP-1a appraisal diagnostics results (gpt-5-mini, N=750)\n\nFirst execution of the appraisal-diagnostics runner against oracle affect on\nrealistic_recall_v3.json. Verdict P1d (bias AND variance), but with nuance: the\nappraisal signal is not absent — valence Pearson r=0.81 and sign accuracy=0.86 —\nit is mis-calibrated (valence bias +0.19, std 0.36; arousal bias -0.14, r=0.37).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): pre-register Addendum N (appraisal prompt calibration)\n\nFrozen exploratory protocol: calibrate the Scherer CPM prompt to reduce the\nmis-calibration measured in WP-1a (valence bias +0.19 with r=0.81; arousal\nbias -0.14). Validation is leakage-free — diagnostic bias on v3 + the\nindependent 15-phrase gold set; Hg1 is NOT re-run (same scenarios as v3).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs(bench): close Addendum N (FAIL) — prompt calibration reverted\n\nPre-registered prompt calibration failed both hypotheses and was reverted; the\nappraisal prompt and to_core_affect mapping are unchanged on this branch. What\nremains is the evidence trail.\n\nHn1 FAIL: at N=150 (seed 42, gpt-5-mini) prompt anchoring nearly eliminated the\nvalence bias (+0.169 -> +0.044) but left arousal bias unchanged (-0.115 ->\n-0.118); the criterion required both axes to fall. Hn2 FAIL: the same anchoring\nover-saturated self_relevance (mean 0.84 -> 0.92), regressing 1/15 gold-set\ncases (routine_lunch).\n\nTakeaway: the Hg1 null is, on valence, a calibration problem (recoverable), but\nprompt-only tuning is insufficient and arousal needs the SEC->arousal mapping\nrecalibrated with a train/test split (next study). claim matrix + 08_limitations\n+ CHANGELOG updated accordingly.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T16:16:55+02:00",
          "tree_id": "1352c992eec184a7417a8ab83e3690f8d30247d6",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/5a1a45b38aaf7d78b1697e7f2ec2fe405eac4670"
        },
        "date": 1780150991451,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 578.865914126597,
            "unit": "iter/sec",
            "range": "stddev: 0.0007910776442168367",
            "extra": "mean: 1.7275157779998458 msec\nrounds: 1500"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 460.393570679658,
            "unit": "iter/sec",
            "range": "stddev: 0.0009947717296577003",
            "extra": "mean: 2.1720546586342326 msec\nrounds: 1743"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 383.1398633005318,
            "unit": "iter/sec",
            "range": "stddev: 0.0014222047173689518",
            "extra": "mean: 2.610012937274575 msec\nrounds: 2216"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 208.451141394087,
            "unit": "iter/sec",
            "range": "stddev: 0.003369705206848184",
            "extra": "mean: 4.797287236290308 msec\nrounds: 3720"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 471.08348080275096,
            "unit": "iter/sec",
            "range": "stddev: 0.0009165272799597892",
            "extra": "mean: 2.122766008045851 msec\nrounds: 1740"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 388.0879035423971,
            "unit": "iter/sec",
            "range": "stddev: 0.00023425240228537502",
            "extra": "mean: 2.5767358139024137 msec\nrounds: 446"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 141433.49426209627,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011459086677664158",
            "extra": "mean: 7.07046096271127 usec\nrounds: 50106"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.685044843432642,
            "unit": "iter/sec",
            "range": "stddev: 0.0006274385236627412",
            "extra": "mean: 34.86137133332553 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8259670007835686,
            "unit": "iter/sec",
            "range": "stddev: 0.002776960803555496",
            "extra": "mean: 1.2107021213333364 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02644918497293195,
            "unit": "iter/sec",
            "range": "stddev: 2.7227720693037596",
            "extra": "mean: 37.808348386666665 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3203.677094835897,
            "unit": "iter/sec",
            "range": "stddev: 0.000013693217595805373",
            "extra": "mean: 312.1413208628079 usec\nrounds: 2272"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 874.9508207278518,
            "unit": "iter/sec",
            "range": "stddev: 0.00003154576452687967",
            "extra": "mean: 1.142921380618996 msec\nrounds: 712"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 348.694184299477,
            "unit": "iter/sec",
            "range": "stddev: 0.00023617553152388438",
            "extra": "mean: 2.8678424964528433 msec\nrounds: 282"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 495.75069268856834,
            "unit": "iter/sec",
            "range": "stddev: 0.0008600226547284606",
            "extra": "mean: 2.0171429203190283 msec\nrounds: 753"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10155.076462839701,
            "unit": "iter/sec",
            "range": "stddev: 0.000004821549679286005",
            "extra": "mean: 98.47291683713885 usec\nrounds: 5387"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10419.29191708622,
            "unit": "iter/sec",
            "range": "stddev: 0.000005140880481991852",
            "extra": "mean: 95.97581178814427 usec\nrounds: 7024"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1838.9626834547644,
            "unit": "iter/sec",
            "range": "stddev: 0.00000814493734447581",
            "extra": "mean: 543.7848244540512 usec\nrounds: 1145"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1835.9866271257135,
            "unit": "iter/sec",
            "range": "stddev: 0.000013087653207926172",
            "extra": "mean: 544.6662765542726 usec\nrounds: 1591"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1010.5315236628896,
            "unit": "iter/sec",
            "range": "stddev: 0.00007125842790348318",
            "extra": "mean: 989.5782334184728 usec\nrounds: 784"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 354.60230446748517,
            "unit": "iter/sec",
            "range": "stddev: 0.0002743114006065701",
            "extra": "mean: 2.8200606352565143 msec\nrounds: 329"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.839764244964133,
            "unit": "iter/sec",
            "range": "stddev: 0.00030062107768032923",
            "extra": "mean: 40.25803103999806 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 473.4204348686043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000231732383068929",
            "extra": "mean: 2.1122873588622038 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.33262067034934,
            "unit": "iter/sec",
            "range": "stddev: 0.000030153640030961322",
            "extra": "mean: 2.6292774946242203 msec\nrounds: 372"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 308.08112270546013,
            "unit": "iter/sec",
            "range": "stddev: 0.00007536399852703764",
            "extra": "mean: 3.24589832450087 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 198.4745260333398,
            "unit": "iter/sec",
            "range": "stddev: 0.00004395273823580369",
            "extra": "mean: 5.03842996875085 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 856.4176100349694,
            "unit": "iter/sec",
            "range": "stddev: 0.000014328461282986496",
            "extra": "mean: 1.1676546445129357 msec\nrounds: 647"
          }
        ]
      }
    ]
  }
}