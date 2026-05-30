window.BENCHMARK_DATA = {
  "lastUpdate": 1780181244624,
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
          "id": "781ffbf893125cac16fe9f9f31abb156f6f10c95",
          "message": "feat(release): hybrid Zenodo flow + DOI gate, docs consolidation, paper refresh (#45)\n\nRe-applies the work that PR #44 failed to land (the merge captured an outdated\ndivergent ref instead of the rebuilt branch). Net effect over main:\n\n- Hybrid release: PyPI + GitHub release stay on the on-tag OIDC workflow;\n  release.py gains --skip-pypi/--skip-github-release, Makefile passes both by\n  default via RELEASE_FLAGS. 'make release' now does reserve-DOI + Zenodo + HF + SWH.\n- DOI-freshness gate: scripts/check_doi_freshness.py (pure evaluate/parse logic,\n  6 in-process unit tests, no noqa, pyproject unchanged) wired into release.yml\n  before publish. Catches a tag reusing the previous release's version_doi —\n  the v0.11.2/v0.11.3 bug.\n- Docs consolidation: detailed content moved from README/CONTRIBUTING/CLAUDE into\n  MkDocs pages (getting-started, comparison, benchmarks, production-readiness,\n  architecture); LLM env vars centralized in docs/contributing/llm-environment.md.\n- Paper/metadata: fidelity count 126 -> 127 (dominance retrieval-gap test is now\n  a real case), abstract rewrite, 38 resolved citations, regenerated arXiv bundle.\n\nmake check green (857 passed), check-arxiv-bundle OK, mkdocs --strict OK.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-31T00:41:44+02:00",
          "tree_id": "67af82cbc70c3f825a1a3ceecee9bbbd25308694",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/781ffbf893125cac16fe9f9f31abb156f6f10c95"
        },
        "date": 1780181243319,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 540.8328056167219,
            "unit": "iter/sec",
            "range": "stddev: 0.0009171327079017734",
            "extra": "mean: 1.8490002633247828 msec\nrounds: 1576"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 441.0601819249328,
            "unit": "iter/sec",
            "range": "stddev: 0.00109674521742868",
            "extra": "mean: 2.267264289502781 msec\nrounds: 1810"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 351.5555226099371,
            "unit": "iter/sec",
            "range": "stddev: 0.0016028245480096354",
            "extra": "mean: 2.8445008986803324 msec\nrounds: 2349"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 235.38365008892382,
            "unit": "iter/sec",
            "range": "stddev: 0.0029295338352697473",
            "extra": "mean: 4.248383435392465 msec\nrounds: 3351"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 453.41225255117,
            "unit": "iter/sec",
            "range": "stddev: 0.0009900585069148877",
            "extra": "mean: 2.2054984054211118 msec\nrounds: 1734"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 372.31668595226324,
            "unit": "iter/sec",
            "range": "stddev: 0.0003358005528706023",
            "extra": "mean: 2.6858855316740096 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135093.2950267821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013539820625337032",
            "extra": "mean: 7.40229187393609 usec\nrounds: 43625"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.40667515518113,
            "unit": "iter/sec",
            "range": "stddev: 0.0005336045010965465",
            "extra": "mean: 36.48746133333702 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.802600977859548,
            "unit": "iter/sec",
            "range": "stddev: 0.014914767068648657",
            "extra": "mean: 1.245949142333335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.034114521470513186,
            "unit": "iter/sec",
            "range": "stddev: 2.7049307705278496",
            "extra": "mean: 29.313030254999997 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3170.7561909556857,
            "unit": "iter/sec",
            "range": "stddev: 0.000014389962188212605",
            "extra": "mean: 315.38218007818307 usec\nrounds: 2560"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 869.747009685402,
            "unit": "iter/sec",
            "range": "stddev: 0.000022340097181274187",
            "extra": "mean: 1.1497596299430934 msec\nrounds: 708"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 357.32063107836916,
            "unit": "iter/sec",
            "range": "stddev: 0.00002727424142772083",
            "extra": "mean: 2.798606945762042 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 489.5469440892156,
            "unit": "iter/sec",
            "range": "stddev: 0.000850822866526825",
            "extra": "mean: 2.0427050195573457 msec\nrounds: 767"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10491.005929372883,
            "unit": "iter/sec",
            "range": "stddev: 0.000005857965817485613",
            "extra": "mean: 95.31974404858397 usec\nrounds: 7099"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10455.75230003111,
            "unit": "iter/sec",
            "range": "stddev: 0.000005024416835962903",
            "extra": "mean: 95.64113334982358 usec\nrounds: 7934"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1805.7554717467713,
            "unit": "iter/sec",
            "range": "stddev: 0.000013875820148866728",
            "extra": "mean: 553.7848372308487 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1809.0621678744717,
            "unit": "iter/sec",
            "range": "stddev: 0.00003682055543021482",
            "extra": "mean: 552.77260105159 usec\nrounds: 1712"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1000.8685226164275,
            "unit": "iter/sec",
            "range": "stddev: 0.000027206581955023694",
            "extra": "mean: 999.1322310605223 usec\nrounds: 792"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.5945412676199,
            "unit": "iter/sec",
            "range": "stddev: 0.00004040546221512705",
            "extra": "mean: 2.691105193818783 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.936938580217113,
            "unit": "iter/sec",
            "range": "stddev: 0.0014106821698115595",
            "extra": "mean: 43.59779734783308 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 463.0666136887195,
            "unit": "iter/sec",
            "range": "stddev: 0.000055731155078343246",
            "extra": "mean: 2.1595165154191305 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.92548455134795,
            "unit": "iter/sec",
            "range": "stddev: 0.00003702910776938605",
            "extra": "mean: 2.6815008397805284 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 301.33994078759196,
            "unit": "iter/sec",
            "range": "stddev: 0.00006756107305304388",
            "extra": "mean: 3.3185113044967327 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.51054723991243,
            "unit": "iter/sec",
            "range": "stddev: 0.00005227656212437392",
            "extra": "mean: 5.167676978145332 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 839.7426148066672,
            "unit": "iter/sec",
            "range": "stddev: 0.000028035430131201197",
            "extra": "mean: 1.1908410772153424 msec\nrounds: 790"
          }
        ]
      }
    ]
  }
}