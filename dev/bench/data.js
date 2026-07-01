window.BENCHMARK_DATA = {
  "lastUpdate": 1782950074181,
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
          "id": "8f1c0da1dcc371df1e0485580bb861fab9a7a2a9",
          "message": "feat(bench): run Addendum X — Hx1 FAIL, construct boundary on MADial-Bench (#93)\n\n* feat(bench): Addendum X scored run — Hx1 FAIL on MADial-Bench (D1 0.996, D2 76.9%)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n* docs(bench): Addendum X closure — Hx1 FAIL, construct boundary (counter-congruent recall)\n\nClosure of the pre-registered third-party retrieval study on MADial-Bench EN:\n\n- Hx1 FAIL, decisive: cosine significantly beats aft_query_appraised\n  (nDCG@5 0.304 vs 0.221, delta=-0.083 [-0.123, -0.043], p_one=0.9998,\n  d=-0.317; MDE 0.051 < |delta| -> powered negative)\n- Diagnostics exonerate appraisal (D1 AUC=0.996) and regime (D2=76.9%\n  affect-discriminative, above U's 62.5%)\n- Post-hoc (exploratory, labeled): 84/160 negative-valence queries, 73.8%\n  of them with positive gold sets -> the benchmark rewards counter-congruent\n  supportive recall (interpersonal emotion regulation), the opposite of\n  AFT's mood-congruence prior. New CONSTRUCT bound on top of regime (U/T2A)\n  and provenance bounds\n- Exploratory arms dropped per pre-declared conditions (full-stack decay,\n  mem0)\n\nPropagation: 08_limitations §2.4 update; claim matrix\ncross_domain_affect_replication wording+evidence (mirrored verbatim in\n09_current_evidence, matrix test green); paper abstract requalified\n(1919<=1920 chars), Addendum X limitations paragraph + conclusion + refs\n(he2025madialbench), addenda range A--X; bundle regenerated (19pp);\nSUBMISSION/ARXIV_CHECKLIST counts updated; ROADMAP + CHANGELOG entries.\n\nmake check-all green.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T01:48:31+02:00",
          "tree_id": "6c36b286cc0f3932ea58a559cd18a84d172ddedd",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8f1c0da1dcc371df1e0485580bb861fab9a7a2a9"
        },
        "date": 1782950072720,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 567.047348795423,
            "unit": "iter/sec",
            "range": "stddev: 0.0007972368532973152",
            "extra": "mean: 1.7635211629580791 msec\nrounds: 1528"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 432.1891491102472,
            "unit": "iter/sec",
            "range": "stddev: 0.0012185416748223743",
            "extra": "mean: 2.3138017279210077 msec\nrounds: 1823"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 360.0126982764594,
            "unit": "iter/sec",
            "range": "stddev: 0.0016178531052665297",
            "extra": "mean: 2.7776798007054864 msec\nrounds: 2268"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 223.97246729716133,
            "unit": "iter/sec",
            "range": "stddev: 0.0029299462295573116",
            "extra": "mean: 4.464834504292994 msec\nrounds: 3494"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 460.4892246711753,
            "unit": "iter/sec",
            "range": "stddev: 0.0009579020772100792",
            "extra": "mean: 2.171603473922928 msec\nrounds: 1764"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 376.06929186736784,
            "unit": "iter/sec",
            "range": "stddev: 0.0002867980243307974",
            "extra": "mean: 2.6590844337077115 msec\nrounds: 445"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135447.64536554087,
            "unit": "iter/sec",
            "range": "stddev: 0.000001198040499797513",
            "extra": "mean: 7.382926423720684 usec\nrounds: 48059"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.954730184485452,
            "unit": "iter/sec",
            "range": "stddev: 0.00008639784144752703",
            "extra": "mean: 35.772121333333 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.810829407950285,
            "unit": "iter/sec",
            "range": "stddev: 0.011782211582104463",
            "extra": "mean: 1.2333050456666637 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03257869253722787,
            "unit": "iter/sec",
            "range": "stddev: 1.9806213026657395",
            "extra": "mean: 30.694908915000003 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3128.1485976392337,
            "unit": "iter/sec",
            "range": "stddev: 0.00004116353431403504",
            "extra": "mean: 319.6779081258112 usec\nrounds: 2449"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.970864696064,
            "unit": "iter/sec",
            "range": "stddev: 0.00008986473307787583",
            "extra": "mean: 1.1587876728053819 msec\nrounds: 706"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.0380282004285,
            "unit": "iter/sec",
            "range": "stddev: 0.000029666964974601222",
            "extra": "mean: 2.792999405750842 msec\nrounds: 313"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 489.4357531119133,
            "unit": "iter/sec",
            "range": "stddev: 0.0009428743716475909",
            "extra": "mean: 2.0431690853024835 msec\nrounds: 762"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10578.207104246214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050494989031663535",
            "extra": "mean: 94.53397822005097 usec\nrounds: 6887"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10437.86591072498,
            "unit": "iter/sec",
            "range": "stddev: 0.00000567309282330009",
            "extra": "mean: 95.80502456661117 usec\nrounds: 8019"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1827.0451545563956,
            "unit": "iter/sec",
            "range": "stddev: 0.000012235166212516209",
            "extra": "mean: 547.3318475496566 usec\nrounds: 1102"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1798.1525760079326,
            "unit": "iter/sec",
            "range": "stddev: 0.00001679181886293693",
            "extra": "mean: 556.1263339622124 usec\nrounds: 1590"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1001.752016871675,
            "unit": "iter/sec",
            "range": "stddev: 0.000022256401490133093",
            "extra": "mean: 998.251047322923 usec\nrounds: 803"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.903914053919,
            "unit": "iter/sec",
            "range": "stddev: 0.000050478956201807635",
            "extra": "mean: 2.6961160616240574 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.804879357277542,
            "unit": "iter/sec",
            "range": "stddev: 0.0008172497472946809",
            "extra": "mean: 43.85026486364103 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 467.9906772487309,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377885458051344",
            "extra": "mean: 2.136794702575909 msec\nrounds: 427"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 371.71028928865655,
            "unit": "iter/sec",
            "range": "stddev: 0.00010864190073676228",
            "extra": "mean: 2.690267202217361 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 297.173666515851,
            "unit": "iter/sec",
            "range": "stddev: 0.0003437766262316376",
            "extra": "mean: 3.3650357103450177 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 192.22809104724345,
            "unit": "iter/sec",
            "range": "stddev: 0.00013166269233323865",
            "extra": "mean: 5.2021533093944745 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 838.9371766585591,
            "unit": "iter/sec",
            "range": "stddev: 0.0000468748009541721",
            "extra": "mean: 1.1919843676292252 msec\nrounds: 797"
          }
        ]
      }
    ]
  }
}