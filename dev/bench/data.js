window.BENCHMARK_DATA = {
  "lastUpdate": 1781023192463,
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
          "id": "8b630d7a2b679688efcddb0242c0f3d8395a2be4",
          "message": "docs(paper): add Addendum O/P to Hg1 affect-free scope discussion (#48)\n\nRecord the recalibrated SEC->affect mapping (Addendum O, held-out\ncalibration PASS) and the leakage-free Hg1 re-run (Addendum P,\nrealistic_recall_v4_noAF, N=160): Hp1 FAIL (naive cosine 0.887 vs\ndual 0.800, delta -0.087, p=0.0018, d=-0.24), with dual>neutral and\ndual>sync PASS showing the inferred affect carries signal and the\ndeferred dual-path schedule is essential. Updates abstract, oracle-\naffect operationalization paragraph, and conclusion study list.\nRegenerates the arXiv bundle to match.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-09T18:33:22+02:00",
          "tree_id": "189a7c16a2c8bb4899dea0a8a6267be52f1f5e03",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8b630d7a2b679688efcddb0242c0f3d8395a2be4"
        },
        "date": 1781023191894,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 631.9086414583851,
            "unit": "iter/sec",
            "range": "stddev: 0.000742549667648003",
            "extra": "mean: 1.5825072397998783 msec\nrounds: 1397"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 400.5411066613754,
            "unit": "iter/sec",
            "range": "stddev: 0.001377112044902194",
            "extra": "mean: 2.4966226521299792 msec\nrounds: 1972"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 335.22606337301187,
            "unit": "iter/sec",
            "range": "stddev: 0.0017743363174485857",
            "extra": "mean: 2.9830616090470348 msec\nrounds: 2476"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 183.45451744202356,
            "unit": "iter/sec",
            "range": "stddev: 0.0036384971439372368",
            "extra": "mean: 5.450942358593193 msec\nrounds: 4236"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 425.43159716625746,
            "unit": "iter/sec",
            "range": "stddev: 0.0010998143792895546",
            "extra": "mean: 2.3505541352848853 msec\nrounds: 1981"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 385.2833129972232,
            "unit": "iter/sec",
            "range": "stddev: 0.00025711811324539215",
            "extra": "mean: 2.5954926316967355 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135771.40044817203,
            "unit": "iter/sec",
            "range": "stddev: 9.071087250842225e-7",
            "extra": "mean: 7.365321390948822 usec\nrounds: 49432"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.405356023250356,
            "unit": "iter/sec",
            "range": "stddev: 0.0005285358639987191",
            "extra": "mean: 29.93531933334263 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8465217886725714,
            "unit": "iter/sec",
            "range": "stddev: 0.0014362400354424778",
            "extra": "mean: 1.1813045019999986 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03319926392837175,
            "unit": "iter/sec",
            "range": "stddev: 1.6802261448641347",
            "extra": "mean: 30.121149738666656 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3536.3835879687676,
            "unit": "iter/sec",
            "range": "stddev: 0.000010361136248450591",
            "extra": "mean: 282.77475424389166 usec\nrounds: 2474"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 913.8980536940055,
            "unit": "iter/sec",
            "range": "stddev: 0.00002109158146586833",
            "extra": "mean: 1.0942139508427309 msec\nrounds: 712"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 366.6088646285928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000860009423186279",
            "extra": "mean: 2.7277027275734 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 510.5208170882506,
            "unit": "iter/sec",
            "range": "stddev: 0.00042204547500947464",
            "extra": "mean: 1.958783983978338 msec\nrounds: 749"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12257.432757183837,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036347248327778765",
            "extra": "mean: 81.58315201965272 usec\nrounds: 6611"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12460.35148302888,
            "unit": "iter/sec",
            "range": "stddev: 0.000003900905497815636",
            "extra": "mean: 80.25455793618741 usec\nrounds: 8587"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2059.0639998188094,
            "unit": "iter/sec",
            "range": "stddev: 0.000011751730906393846",
            "extra": "mean: 485.65756095390753 usec\nrounds: 1132"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2076.888903356775,
            "unit": "iter/sec",
            "range": "stddev: 0.000012995349725633036",
            "extra": "mean: 481.4894038789213 usec\nrounds: 1753"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1204.0262998669023,
            "unit": "iter/sec",
            "range": "stddev: 0.000029416429092704796",
            "extra": "mean: 830.5466418055352 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 393.1954699080863,
            "unit": "iter/sec",
            "range": "stddev: 0.00007372347215798593",
            "extra": "mean: 2.5432642960860177 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.92738158826687,
            "unit": "iter/sec",
            "range": "stddev: 0.0008995427667569128",
            "extra": "mean: 47.78428661905125 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 472.63851866612515,
            "unit": "iter/sec",
            "range": "stddev: 0.00006127403207623236",
            "extra": "mean: 2.1157818512595803 msec\nrounds: 437"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 391.11794485369194,
            "unit": "iter/sec",
            "range": "stddev: 0.00007517587363712493",
            "extra": "mean: 2.5567735082420637 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 322.88854179510315,
            "unit": "iter/sec",
            "range": "stddev: 0.00008568186225863677",
            "extra": "mean: 3.097043934852834 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 212.12489702897176,
            "unit": "iter/sec",
            "range": "stddev: 0.00015954348025478614",
            "extra": "mean: 4.714203820513445 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 976.7768351691354,
            "unit": "iter/sec",
            "range": "stddev: 0.000027206472547077938",
            "extra": "mean: 1.0237753026020968 msec\nrounds: 846"
          }
        ]
      }
    ]
  }
}