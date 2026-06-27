window.BENCHMARK_DATA = {
  "lastUpdate": 1782557289197,
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
          "id": "c8531fed69af8dc2a414668de1ab7d4ea6ed64a0",
          "message": "docs(paper): fold Addendum T2A — query-appraisal recovery is regime-bound (#78)\n\nBoundary section: the Addendum T production-reachable recovery is now qualified\nas regime-bound. The pre-registered naturalistic re-test (Addendum T2A) applied\nthe same retrieve-time query appraisal to DailyDialog (N=120, 396 q) and did NOT\nbeat cosine (delta=-0.008, p_holm=1.000, 0/3 types), reproducing the Hk1 null,\ndespite a faithful diagnostic (valence r=0.69, arousal r=0.74). So the recovery\nis confined to the affect-discriminative regime, not naturalistic dialogue.\n\nAbstract qualified (\"within the affect-discriminative regime\", 1903<=1920 chars);\naddenda count A--V and T2A. arXiv bundle regenerated; check-arxiv-bundle and\nreproduce-paper-check green.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T12:42:08+02:00",
          "tree_id": "17be9b8cff9e16206ab727c57a7198c2dced710a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/c8531fed69af8dc2a414668de1ab7d4ea6ed64a0"
        },
        "date": 1782557287821,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 520.1597147351847,
            "unit": "iter/sec",
            "range": "stddev: 0.0009481075099399777",
            "extra": "mean: 1.9224864434361353 msec\nrounds: 1653"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 462.6128583892331,
            "unit": "iter/sec",
            "range": "stddev: 0.0009892808237136528",
            "extra": "mean: 2.161634684089607 msec\nrounds: 1741"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 356.45686169218897,
            "unit": "iter/sec",
            "range": "stddev: 0.0015752001646727567",
            "extra": "mean: 2.8053885546002744 msec\nrounds: 2326"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 221.84037658141753,
            "unit": "iter/sec",
            "range": "stddev: 0.0029074550427436144",
            "extra": "mean: 4.507745683676256 msec\nrounds: 3645"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 465.1738252190549,
            "unit": "iter/sec",
            "range": "stddev: 0.0009473388896216734",
            "extra": "mean: 2.149734025832365 msec\nrounds: 1742"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 378.58926441898706,
            "unit": "iter/sec",
            "range": "stddev: 0.00032146922252890216",
            "extra": "mean: 2.6413849889131926 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138326.06572914217,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014437754572898562",
            "extra": "mean: 7.229295467407505 usec\nrounds: 42492"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.457039403607933,
            "unit": "iter/sec",
            "range": "stddev: 0.0001757197359524187",
            "extra": "mean: 35.14069000000101 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8248086803890505,
            "unit": "iter/sec",
            "range": "stddev: 0.0029608628026687057",
            "extra": "mean: 1.2124023713333305 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03295476903334235,
            "unit": "iter/sec",
            "range": "stddev: 1.393505238893422",
            "extra": "mean: 30.344621714333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3175.8618294397825,
            "unit": "iter/sec",
            "range": "stddev: 0.000026121065479865446",
            "extra": "mean: 314.87515947014566 usec\nrounds: 2339"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 861.8812866915256,
            "unit": "iter/sec",
            "range": "stddev: 0.000016796626369909417",
            "extra": "mean: 1.1602525956198282 msec\nrounds: 685"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.949851419793,
            "unit": "iter/sec",
            "range": "stddev: 0.000040639649564614724",
            "extra": "mean: 2.8172993903223733 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.74674808177883,
            "unit": "iter/sec",
            "range": "stddev: 0.0007932054803560099",
            "extra": "mean: 2.0294400803108594 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10577.734965575919,
            "unit": "iter/sec",
            "range": "stddev: 0.0000057589672674353286",
            "extra": "mean: 94.53819775730727 usec\nrounds: 6599"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10466.867291916533,
            "unit": "iter/sec",
            "range": "stddev: 0.000006295952865584172",
            "extra": "mean: 95.53956996973592 usec\nrounds: 7932"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1810.766558832048,
            "unit": "iter/sec",
            "range": "stddev: 0.000008596895569745625",
            "extra": "mean: 552.2523017240854 usec\nrounds: 1160"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1825.580679452249,
            "unit": "iter/sec",
            "range": "stddev: 0.00004067357385697944",
            "extra": "mean: 547.7709154437601 usec\nrounds: 1703"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1007.8921664783545,
            "unit": "iter/sec",
            "range": "stddev: 0.000025770337503179032",
            "extra": "mean: 992.1696320887876 usec\nrounds: 723"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 372.10590024409004,
            "unit": "iter/sec",
            "range": "stddev: 0.00008971232586292366",
            "extra": "mean: 2.687406997158687 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.51505073492024,
            "unit": "iter/sec",
            "range": "stddev: 0.0006761201449728198",
            "extra": "mean: 46.47909095454462 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.3810770113153,
            "unit": "iter/sec",
            "range": "stddev: 0.00003406824121364789",
            "extra": "mean: 2.1214258458151796 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 377.6342536146188,
            "unit": "iter/sec",
            "range": "stddev: 0.000039669825631934696",
            "extra": "mean: 2.6480648681316774 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.9041289318939,
            "unit": "iter/sec",
            "range": "stddev: 0.00007320848815945556",
            "extra": "mean: 3.2797194432987453 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.0240066422362,
            "unit": "iter/sec",
            "range": "stddev: 0.00013098972822727777",
            "extra": "mean: 5.1540013903738995 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 849.9732929725931,
            "unit": "iter/sec",
            "range": "stddev: 0.0000249045894181553",
            "extra": "mean: 1.1765075541405798 msec\nrounds: 785"
          }
        ]
      }
    ]
  }
}