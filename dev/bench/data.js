window.BENCHMARK_DATA = {
  "lastUpdate": 1782514059149,
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
          "id": "0cf56ae27a9e08d1fc970e6795dce36d54f8def5",
          "message": "docs: close residual gaps from the Addendum R/S propagation (#69)\n\nAudit follow-up — surfaces left stale after #64–#68:\n\n- benchmarks/datasets/README.md: document emobank_v1.json (the only third-party,\n  human-labeled set) with source, citation, and its CC-BY-SA 4.0 license (share-alike;\n  distinct from the repo's MIT code) — attribution + compliance.\n- README \"Validation & Benchmarks\": add the two positive in-regime results\n  (Addendum R downstream PASS; Addendum S human-gold valence r=0.70), keeping the\n  per-regime scoping.\n- CLAUDE.md: add `bench-a3` / `bench-human-gold` to Commands and the two new benchmark\n  suites to Conventions.\n- 08_limitations §2.2: note that downstream answer quality has since been measured\n  (Addendum R), while the comparative benchmark itself still reports recall@k.\n\n(Left paper/SUBMISSION.md untouched: the markdown formatter re-pads its metadata\ntable in a way that breaks scripts/sync_release_metadata.py's DOI-row matcher; the\nabstract-length checkbox is a submission-prep artifact finalized at arXiv time.)\n\nRefs #60 #61 #62; supports #31\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T00:41:21+02:00",
          "tree_id": "71f222aaad7872571fcb6e8ac1c2332aaa853ccb",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/0cf56ae27a9e08d1fc970e6795dce36d54f8def5"
        },
        "date": 1782514057708,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 572.4324312843839,
            "unit": "iter/sec",
            "range": "stddev: 0.000819017953254686",
            "extra": "mean: 1.7469310705479593 msec\nrounds: 1460"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 428.39413137556124,
            "unit": "iter/sec",
            "range": "stddev: 0.0012353235013119856",
            "extra": "mean: 2.3342990175636364 msec\nrounds: 1765"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 388.9239066723376,
            "unit": "iter/sec",
            "range": "stddev: 0.0014087165023915834",
            "extra": "mean: 2.5711970461164904 msec\nrounds: 2060"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 194.0786496326199,
            "unit": "iter/sec",
            "range": "stddev: 0.0035150868580824217",
            "extra": "mean: 5.152550277389833 msec\nrounds: 3724"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 458.4613009216894,
            "unit": "iter/sec",
            "range": "stddev: 0.0009823730530002341",
            "extra": "mean: 2.1812091838277357 msec\nrounds: 1719"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.3985352693065,
            "unit": "iter/sec",
            "range": "stddev: 0.00028086065534765225",
            "extra": "mean: 2.6219293141592623 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136264.44621314775,
            "unit": "iter/sec",
            "range": "stddev: 0.000001245777171469135",
            "extra": "mean: 7.338671442114686 usec\nrounds: 37065"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.06013785141521,
            "unit": "iter/sec",
            "range": "stddev: 0.00023027967640982998",
            "extra": "mean: 35.63774366666431 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8103252323945768,
            "unit": "iter/sec",
            "range": "stddev: 0.016969924113308478",
            "extra": "mean: 1.2340723946666685 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.030619989527549185,
            "unit": "iter/sec",
            "range": "stddev: 1.035709906316833",
            "extra": "mean: 32.658404376666674 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3135.058038320951,
            "unit": "iter/sec",
            "range": "stddev: 0.000013625028592037838",
            "extra": "mean: 318.9733611871415 usec\nrounds: 2190"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 860.8829485534366,
            "unit": "iter/sec",
            "range": "stddev: 0.000020979653678129383",
            "extra": "mean: 1.1615981030642148 msec\nrounds: 718"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 352.46683427105614,
            "unit": "iter/sec",
            "range": "stddev: 0.00003216216865833195",
            "extra": "mean: 2.837146371709328 msec\nrounds: 304"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 491.7419211847946,
            "unit": "iter/sec",
            "range": "stddev: 0.000961271383853755",
            "extra": "mean: 2.033587044176785 msec\nrounds: 747"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10678.91173013809,
            "unit": "iter/sec",
            "range": "stddev: 0.000005110641658073052",
            "extra": "mean: 93.6425007782201 usec\nrounds: 7067"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10566.137693707667,
            "unit": "iter/sec",
            "range": "stddev: 0.00000523259218241839",
            "extra": "mean: 94.64196180175833 usec\nrounds: 8325"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1818.8003699300082,
            "unit": "iter/sec",
            "range": "stddev: 0.000010436334307876652",
            "extra": "mean: 549.8129517306412 usec\nrounds: 1098"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1814.3008221137165,
            "unit": "iter/sec",
            "range": "stddev: 0.000011732173374828866",
            "extra": "mean: 551.1765126330974 usec\nrounds: 1504"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1002.807700807071,
            "unit": "iter/sec",
            "range": "stddev: 0.000024390543060164724",
            "extra": "mean: 997.2001603051001 usec\nrounds: 786"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.0309354726876,
            "unit": "iter/sec",
            "range": "stddev: 0.00016918828966668598",
            "extra": "mean: 2.7097999215678525 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.303083541643165,
            "unit": "iter/sec",
            "range": "stddev: 0.0003031524680413984",
            "extra": "mean: 49.25360219047142 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 466.1409590389562,
            "unit": "iter/sec",
            "range": "stddev: 0.000030285020238080135",
            "extra": "mean: 2.1452738288900894 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 327.6525372075582,
            "unit": "iter/sec",
            "range": "stddev: 0.0007348739140919214",
            "extra": "mean: 3.052013601123221 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 300.9857870768446,
            "unit": "iter/sec",
            "range": "stddev: 0.00006316181807426857",
            "extra": "mean: 3.3224160174204185 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 187.90848237128927,
            "unit": "iter/sec",
            "range": "stddev: 0.00022016802882834447",
            "extra": "mean: 5.3217395371439125 msec\nrounds: 175"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 836.8776721877476,
            "unit": "iter/sec",
            "range": "stddev: 0.000029463022250058328",
            "extra": "mean: 1.1949177678331666 msec\nrounds: 771"
          }
        ]
      }
    ]
  }
}