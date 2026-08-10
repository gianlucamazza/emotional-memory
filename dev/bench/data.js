window.BENCHMARK_DATA = {
  "lastUpdate": 1786353222143,
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
          "id": "290ce1a1a3e9e6062379d6f74d8d2f91e7c9137b",
          "message": "fix: restore chromadb CVE pin, cosine SQLite index, keyword appraisal scaling (#126)\n\n* fix: restore chromadb CVE pin, cosine SQLite index, keyword appraisal scaling\n\nRepository analysis pass over AFT v0.17.0: baseline reproduction, source review\nof the engine/retrieval/store/appraisal layers, and the highest-impact safe\ncorrections. No committed benchmark artefact is regenerated and no calibrated\nretrieval weight is touched.\n\nSecurity\n- The [chroma] extra was widened to chromadb<2.0 by an automated dependency PR\n  (#122), re-admitting the CVE-2026-45829 range (>=1.0.0,<=1.5.9) that v0.17.0\n  had excluded. PyPI's newest chromadb is still 1.5.9, so installs of the extra\n  resolved to a vulnerable version while SECURITY.md described the CVE as\n  resolved. Pin restored to <1.0 with the rationale inline, plus a dependabot\n  ignore rule so it cannot be reverted automatically again.\n\nCorrectness\n- SQLiteStore created its sqlite-vec index without distance_metric, so ANN\n  candidate prefiltering ranked by L2 while the retrieval scorer, the store's\n  brute-force fallback and InMemoryStore all rank by cosine. New indexes use\n  distance_metric=cosine; legacy databases are detected on open, warned about,\n  and migratable via the new rebuild_vector_index(). Zero vectors (NULL cosine\n  distance) no longer sort first.\n- KeywordAppraisalEngine divided each dimension by every matching rule, including\n  rules that left it at its neutral default, so the self-reference rule halved\n  the goal_relevance of \"I succeeded at the project\". Only contributing rules\n  count now, as the surrounding comment already documented. Disclosed as a\n  behaviour change: the published keyword-engine numbers (Addendum S r=0.07,\n  the aft_keyword_synchronous ablation arm) predate the fix.\n- as_async() did not forward query_classifier, silently disabling per-query-type\n  weight routing for engines configured with mode=\"llm\".\n- LLMQueryClassifier cached error fallbacks (pinning default_type for a query\n  after a transient failure) and treated cache_size=0 as unbounded rather than\n  disabled.\n- search_by_embedding(top_k<=0) raised out of numpy instead of returning [].\n\nDeveloper loop\n- make check was red on a fresh clone: ruff >=0.16 (unpinned, and make install\n  bypasses uv.lock) flags the S310 noqa directives as unused and formats Markdown\n  code blocks. S310 moved to the scripts/** per-file ignores and *.md excluded\n  from the formatter, so both 0.15.x and 0.16.x agree.\n- test_figure_inventory required gitignored PDF renders that only make figures\n  produces. Committed assets stay strictly required; generated ones are skipped\n  when absent entirely and failed when the build is partial.\n- make install failed without a pre-existing virtualenv; every install target now\n  bootstraps .venv when no environment is active.\n\nDocs\n- Signal indices in both engines aligned to retrieval.py's 1-based s1..s6.\n- SQLiteStore migration note, keyword-engine provenance note in the limitations\n  page and the claim matrix, 08_limitations last-updated stamp, CLAUDE.md's\n  make check description.\n\nmake check: lint + typecheck + meta-check + 1053 tests + 127 fidelity, all green.\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01CnnPbzWW7W13a8FbE39n95\n\n* fix(build): make check-arxiv-bundle POSIX-sh compatible\n\n`diff <(tar ...)` is a bashism; make runs recipes with /bin/sh, which is dash on\nmost Linux distributions, so `make check-all` aborted before the bundle check\ncould run. Pipe the extracted main.tex into diff instead. Also records the\nMakefile install/venv bootstrap in the changelog.\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01CnnPbzWW7W13a8FbE39n95\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-08-10T09:06:56Z",
          "tree_id": "b61aa051febe607b2f72f4b832ca26311fc2e770",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/290ce1a1a3e9e6062379d6f74d8d2f91e7c9137b"
        },
        "date": 1786353221010,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 535.5670647732043,
            "unit": "iter/sec",
            "range": "stddev: 0.000871505332822915",
            "extra": "mean: 1.8671797908698295 msec\nrounds: 1621"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 467.1734068383388,
            "unit": "iter/sec",
            "range": "stddev: 0.0009564241702957839",
            "extra": "mean: 2.1405327986617206 msec\nrounds: 1644"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 374.392751784594,
            "unit": "iter/sec",
            "range": "stddev: 0.0013676573631793353",
            "extra": "mean: 2.670991880140211 msec\nrounds: 2286"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 235.24711499221036,
            "unit": "iter/sec",
            "range": "stddev: 0.002521553167829304",
            "extra": "mean: 4.250849155081509 msec\nrounds: 3611"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 456.55037561725993,
            "unit": "iter/sec",
            "range": "stddev: 0.0009596040203356271",
            "extra": "mean: 2.190338795906129 msec\nrounds: 1710"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 371.10370279431544,
            "unit": "iter/sec",
            "range": "stddev: 0.00027735122173668404",
            "extra": "mean: 2.6946645707662227 msec\nrounds: 431"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136746.48118060498,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014412205088001223",
            "extra": "mean: 7.31280243093986 usec\nrounds: 46318"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.740021040622267,
            "unit": "iter/sec",
            "range": "stddev: 0.00029117869490705156",
            "extra": "mean: 36.04899933333172 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7994141886430324,
            "unit": "iter/sec",
            "range": "stddev: 0.006458975685982778",
            "extra": "mean: 1.2509160009999978 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.032835037642718896,
            "unit": "iter/sec",
            "range": "stddev: 0.3947814507930018",
            "extra": "mean: 30.455271922666668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3129.4144720838804,
            "unit": "iter/sec",
            "range": "stddev: 0.000020597729027699922",
            "extra": "mean: 319.5485957263114 usec\nrounds: 2340"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 863.1700486760274,
            "unit": "iter/sec",
            "range": "stddev: 0.00002465451323538017",
            "extra": "mean: 1.158520272493061 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 352.7697151851653,
            "unit": "iter/sec",
            "range": "stddev: 0.00003783934304229145",
            "extra": "mean: 2.8347104554457285 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 486.74427688700547,
            "unit": "iter/sec",
            "range": "stddev: 0.0010212960676032254",
            "extra": "mean: 2.054466888435842 msec\nrounds: 735"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10444.580042741658,
            "unit": "iter/sec",
            "range": "stddev: 0.000005229160758804252",
            "extra": "mean: 95.74343783165688 usec\nrounds: 6973"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10599.173471128664,
            "unit": "iter/sec",
            "range": "stddev: 0.000005792021021828006",
            "extra": "mean: 94.34697929267064 usec\nrounds: 8258"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1801.5637268646763,
            "unit": "iter/sec",
            "range": "stddev: 0.000015004791935986143",
            "extra": "mean: 555.0733427233988 usec\nrounds: 1065"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1781.5348220234928,
            "unit": "iter/sec",
            "range": "stddev: 0.000015151271020216678",
            "extra": "mean: 561.3137546557668 usec\nrounds: 1235"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 987.2345459628159,
            "unit": "iter/sec",
            "range": "stddev: 0.000027315268575534766",
            "extra": "mean: 1.0129305179700072 msec\nrounds: 946"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 366.1221081978317,
            "unit": "iter/sec",
            "range": "stddev: 0.00006250456572659567",
            "extra": "mean: 2.7313291866539138 msec\nrounds: 1034"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.20482766042012,
            "unit": "iter/sec",
            "range": "stddev: 0.0016692583486468027",
            "extra": "mean: 43.09448079658331 msec\nrounds: 644"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 454.41762205931366,
            "unit": "iter/sec",
            "range": "stddev: 0.00004360219653266356",
            "extra": "mean: 2.2006188832823765 msec\nrounds: 2596"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 361.74455567804875,
            "unit": "iter/sec",
            "range": "stddev: 0.0000779043837878736",
            "extra": "mean: 2.764381617646227 msec\nrounds: 1020"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 292.67514185050635,
            "unit": "iter/sec",
            "range": "stddev: 0.00009820014716978482",
            "extra": "mean: 3.416757547898562 msec\nrounds: 595"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 186.10488822883073,
            "unit": "iter/sec",
            "range": "stddev: 0.00014585027112403197",
            "extra": "mean: 5.37331399253963 msec\nrounds: 268"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 826.0926725600757,
            "unit": "iter/sec",
            "range": "stddev: 0.00002648753065382146",
            "extra": "mean: 1.2105179397136914 msec\nrounds: 1045"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5145.848312132287,
            "unit": "iter/sec",
            "range": "stddev: 0.000009667446811812344",
            "extra": "mean: 194.33141813417149 usec\nrounds: 4599"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 74.51597300253145,
            "unit": "iter/sec",
            "range": "stddev: 0.0002156559585870324",
            "extra": "mean: 13.419941520001734 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1818.7056814066837,
            "unit": "iter/sec",
            "range": "stddev: 0.000019390251112384656",
            "extra": "mean: 549.8415770200635 usec\nrounds: 1584"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 25.12447707157548,
            "unit": "iter/sec",
            "range": "stddev: 0.005094703589519572",
            "extra": "mean: 39.80182342307724 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17672.9664790741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067126166206965995",
            "extra": "mean: 56.58359626178563 usec\nrounds: 13749"
          }
        ]
      }
    ]
  }
}