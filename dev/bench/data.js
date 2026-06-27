window.BENCHMARK_DATA = {
  "lastUpdate": 1782550624348,
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
          "id": "fb24ec4fe5574e71a948aec4fdc390f996fd184b",
          "message": "docs(paper): fold Addenda U/V/T into the paper (honest scope + production-reachability) (#74)\n\nBrings the paper current with the three results landed after the R/S fold (#68), before\nthe arXiv submission (#31). All on the pre-arXiv bundle.\n\n- Limitations / Controlled benchmark scope: add the Addendum U circularity bound — 62.5%\n  of v2 queries are affect-favorable by construction; the advantage concentrates there\n  (Δ+0.304) and is null on the neutral 37.5% (Δ+0.013, ns). The +0.205/+0.18 headline is\n  scoped to a ~62%-affect-discriminative benchmark, not regime-independent. (Integrity: the\n  paper cited +0.205 without this bound.)\n- State-injection boundary: Addendum T MOVES it — retrieve-time query appraisal (direct-VAD,\n  no oracle) beats cosine (Δ+0.115, p<0.001), recovering ~59% (~82% on the affect-discriminative\n  subset). The advantage is largely production-reachable; naturalistic QA untested.\n- Human validation: Addendum V — direct-VAD beats the SEC→projection on EmoBank (valence 0.79,\n  arousal 0.58, dominance 0.43); ships opt-in (DIRECT_VAD_SCHEMA), SEC stays default.\n- Abstract recondensed to fold U+T (1885 ≤ 1920 arXiv limit); conclusion addenda A--S → A--V.\n- Regenerated arxiv-submission bundle; tables unchanged.\n\nRefs #62 #31\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T10:51:34+02:00",
          "tree_id": "4e50f80df0ef97babad3cfb78d7d198e198a60bb",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/fb24ec4fe5574e71a948aec4fdc390f996fd184b"
        },
        "date": 1782550623037,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 524.4331255371121,
            "unit": "iter/sec",
            "range": "stddev: 0.0008767378380107727",
            "extra": "mean: 1.9068208152866455 msec\nrounds: 1727"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 429.7628586679344,
            "unit": "iter/sec",
            "range": "stddev: 0.0011453831454107395",
            "extra": "mean: 2.3268646413502005 msec\nrounds: 1896"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 388.1837702669036,
            "unit": "iter/sec",
            "range": "stddev: 0.001382548921200419",
            "extra": "mean: 2.5760994575132026 msec\nrounds: 2236"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 220.30512422893582,
            "unit": "iter/sec",
            "range": "stddev: 0.00279049295973974",
            "extra": "mean: 4.539159057239285 msec\nrounds: 3861"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 468.25094704406405,
            "unit": "iter/sec",
            "range": "stddev: 0.0009179319635729311",
            "extra": "mean: 2.135606999436344 msec\nrounds: 1774"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 387.87474241634175,
            "unit": "iter/sec",
            "range": "stddev: 0.0002678857180372796",
            "extra": "mean: 2.578151889371048 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136589.17791205103,
            "unit": "iter/sec",
            "range": "stddev: 0.000001129772231371767",
            "extra": "mean: 7.321224238159585 usec\nrounds: 49220"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.28858955412064,
            "unit": "iter/sec",
            "range": "stddev: 0.00031133542943330493",
            "extra": "mean: 35.34994200000104 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.811310162787266,
            "unit": "iter/sec",
            "range": "stddev: 0.011173581368266797",
            "extra": "mean: 1.232574231000001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.037611243109896615,
            "unit": "iter/sec",
            "range": "stddev: 2.222241890394933",
            "extra": "mean: 26.58779442833334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3197.0109153929984,
            "unit": "iter/sec",
            "range": "stddev: 0.000012222095560405332",
            "extra": "mean: 312.7921757117533 usec\nrounds: 2635"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 869.4377643206961,
            "unit": "iter/sec",
            "range": "stddev: 0.00008454525111079669",
            "extra": "mean: 1.1501685813950282 msec\nrounds: 731"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.58833501647115,
            "unit": "iter/sec",
            "range": "stddev: 0.000022291132484636113",
            "extra": "mean: 2.7579486250007834 msec\nrounds: 256"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 500.98549145055614,
            "unit": "iter/sec",
            "range": "stddev: 0.0004216183726411472",
            "extra": "mean: 1.9960657884614472 msec\nrounds: 780"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10475.5589197534,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053007388678458475",
            "extra": "mean: 95.46030027231622 usec\nrounds: 7713"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10138.39603803687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052234121855668915",
            "extra": "mean: 98.6349316251048 usec\nrounds: 8234"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1825.9393861393537,
            "unit": "iter/sec",
            "range": "stddev: 0.00001017263513617024",
            "extra": "mean: 547.6633055790172 usec\nrounds: 1165"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1826.406349932575,
            "unit": "iter/sec",
            "range": "stddev: 0.000022670210764080645",
            "extra": "mean: 547.5232825580773 usec\nrounds: 1720"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1013.1486053525603,
            "unit": "iter/sec",
            "range": "stddev: 0.000019447094253639895",
            "extra": "mean: 987.022036764306 usec\nrounds: 816"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 377.25770375309173,
            "unit": "iter/sec",
            "range": "stddev: 0.00003563012593155128",
            "extra": "mean: 2.6507079644806453 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.317908160108416,
            "unit": "iter/sec",
            "range": "stddev: 0.0019913997234822186",
            "extra": "mean: 46.9089177272689 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 468.0915649293599,
            "unit": "iter/sec",
            "range": "stddev: 0.00003810846088650592",
            "extra": "mean: 2.1363341596444085 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.6459634349475,
            "unit": "iter/sec",
            "range": "stddev: 0.000048407465340204716",
            "extra": "mean: 2.6691866391178594 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.71959627362816,
            "unit": "iter/sec",
            "range": "stddev: 0.000058667757325140105",
            "extra": "mean: 3.2817055818820164 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.2042199407529,
            "unit": "iter/sec",
            "range": "stddev: 0.0001347053282016372",
            "extra": "mean: 5.09673033690084 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 856.4881763212733,
            "unit": "iter/sec",
            "range": "stddev: 0.000020342708455100116",
            "extra": "mean: 1.1675584411394078 msec\nrounds: 807"
          }
        ]
      }
    ]
  }
}