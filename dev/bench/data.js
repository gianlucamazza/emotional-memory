window.BENCHMARK_DATA = {
  "lastUpdate": 1782503991419,
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
          "id": "990633770f157e04d56ad9169825d519b2509991",
          "message": "docs(research): full project validation report 2026-06 + claim-scope fixes (#63)\n\n* docs(research): full project validation report 2026-06 + claim-scope fixes\n\nRun the complete validation suite and re-verify every problem_register entry.\n\nFindings:\n- Fix broken docs link (review_response: comparison.md -> ../comparison.md)\n  that was failing the strict mkdocs build / docs deploy.\n- Add a discoverable Limitations link to the README validation section.\n- Record A7 multi-seed determinism claim as non-reproducing (fresh sweep\n  shows genuine cross-seed variance; flagged for maintainer decision).\n- Note register is stale on C1/C3 (already implemented) and D2 (torch 2.12.1\n  now patched).\n\nQuality gates green: lint, mypy strict, 91.52% coverage, 127 fidelity,\npaper-table reproduction, preflight. LLM suites blocked (no API key).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01EYzreMkdLAzroufjVcjibp\n\n* fix(security): bump torch 2.12.0 -> 2.12.1 (CVE-2025-3000) + file gap issues\n\n- uv.lock: torch 2.12.1 / triton 3.7.1 (targeted; clears CVE-2025-3000,\n  pip-audit clean). Test suite already ran against 2.12.1.\n- SECURITY.md: mark torch advisory resolved; chromadb CVE-2026-45829 still\n  unpatched (optional [chroma] extra only, not in runtime wheel).\n- validation report: cross-reference new tracking issues #60 (A7 multi-seed\n  reproducibility), #61 (A3 downstream task), #62 (A5 human-gold appraisal).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01EYzreMkdLAzroufjVcjibp\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-06-26T21:54:26+02:00",
          "tree_id": "be8c75365ae12af6be8dfb8b031ab0f93d46c6a0",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/990633770f157e04d56ad9169825d519b2509991"
        },
        "date": 1782503990176,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 561.3666077960637,
            "unit": "iter/sec",
            "range": "stddev: 0.000818447255422501",
            "extra": "mean: 1.7813670890151796 msec\nrounds: 1584"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 441.8164689048239,
            "unit": "iter/sec",
            "range": "stddev: 0.0010700275157431707",
            "extra": "mean: 2.263383260653011 msec\nrounds: 1807"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 382.8267096252689,
            "unit": "iter/sec",
            "range": "stddev: 0.0013974214280673023",
            "extra": "mean: 2.6121479375847447 msec\nrounds: 2211"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 245.8556336183237,
            "unit": "iter/sec",
            "range": "stddev: 0.0025143443713684247",
            "extra": "mean: 4.067427641509491 msec\nrounds: 3445"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 460.5979904332581,
            "unit": "iter/sec",
            "range": "stddev: 0.0009673654556146803",
            "extra": "mean: 2.171090670759022 msec\nrounds: 1792"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 386.7513654883419,
            "unit": "iter/sec",
            "range": "stddev: 0.0002678886613212395",
            "extra": "mean: 2.58564051541828 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137223.77183413468,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014905147891738719",
            "extra": "mean: 7.287367098528099 usec\nrounds: 47396"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.31966726883041,
            "unit": "iter/sec",
            "range": "stddev: 0.000298232498120577",
            "extra": "mean: 35.31114933333394 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8276653791976731,
            "unit": "iter/sec",
            "range": "stddev: 0.001196579551614533",
            "extra": "mean: 1.2082177473333313 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.036312564470187124,
            "unit": "iter/sec",
            "range": "stddev: 1.9767200531833862",
            "extra": "mean: 27.538677440999994 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3213.374316924448,
            "unit": "iter/sec",
            "range": "stddev: 0.000014358601183704915",
            "extra": "mean: 311.19935039410836 usec\nrounds: 2286"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 875.3424335995769,
            "unit": "iter/sec",
            "range": "stddev: 0.000021622125452028473",
            "extra": "mean: 1.1424100576134613 msec\nrounds: 729"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.1811214571449,
            "unit": "iter/sec",
            "range": "stddev: 0.00003063951276704917",
            "extra": "mean: 2.791883603278199 msec\nrounds: 305"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.9368403362199,
            "unit": "iter/sec",
            "range": "stddev: 0.0008629008142094168",
            "extra": "mean: 2.0286574631304184 msec\nrounds: 773"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10569.988883186126,
            "unit": "iter/sec",
            "range": "stddev: 0.000004992639517182635",
            "extra": "mean: 94.60747887736363 usec\nrounds: 7196"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10509.412468820497,
            "unit": "iter/sec",
            "range": "stddev: 0.00000507638112323184",
            "extra": "mean: 95.15279783401944 usec\nrounds: 8587"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1846.8741253379121,
            "unit": "iter/sec",
            "range": "stddev: 0.00001195938912632945",
            "extra": "mean: 541.455417172535 usec\nrounds: 1153"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1822.7004110835182,
            "unit": "iter/sec",
            "range": "stddev: 0.000011676518225553413",
            "extra": "mean: 548.6365142176833 usec\nrounds: 1688"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1007.6893905416539,
            "unit": "iter/sec",
            "range": "stddev: 0.000021688849291365072",
            "extra": "mean: 992.3692850060467 usec\nrounds: 807"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 376.67483913148044,
            "unit": "iter/sec",
            "range": "stddev: 0.00008226056387386888",
            "extra": "mean: 2.654809655738499 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.970668270675926,
            "unit": "iter/sec",
            "range": "stddev: 0.0008526283098482093",
            "extra": "mean: 41.71765212000082 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 465.19517091708985,
            "unit": "iter/sec",
            "range": "stddev: 0.0002426112667100283",
            "extra": "mean: 2.149635384280949 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 379.51067134733853,
            "unit": "iter/sec",
            "range": "stddev: 0.000028814647957226237",
            "extra": "mean: 2.6349720192315034 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.81869708450995,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325456881778672",
            "extra": "mean: 3.2592537857122856 msec\nrounds: 294"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.46274019545424,
            "unit": "iter/sec",
            "range": "stddev: 0.000037658409243960535",
            "extra": "mean: 5.090023680852325 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 851.3942630727612,
            "unit": "iter/sec",
            "range": "stddev: 0.000024791545329545874",
            "extra": "mean: 1.1745439726019609 msec\nrounds: 803"
          }
        ]
      }
    ]
  }
}