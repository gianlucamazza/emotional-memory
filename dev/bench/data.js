window.BENCHMARK_DATA = {
  "lastUpdate": 1790646549723,
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
          "id": "79c5a675eca7d47b65eeebd9b6072509f23addc2",
          "message": "fix(ci): Dependabot ecosystem is github-actions not github_actions (#142)",
          "timestamp": "2026-09-29T03:41:55+02:00",
          "tree_id": "6f9edcffb2d0c91c3b761d4c324ffb1677a89fce",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/79c5a675eca7d47b65eeebd9b6072509f23addc2"
        },
        "date": 1790646548136,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 463.0859562253522,
            "unit": "iter/sec",
            "range": "stddev: 0.000864601406525805",
            "extra": "mean: 2.159426315043267 msec\nrounds: 1622"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 483.8537615160774,
            "unit": "iter/sec",
            "range": "stddev: 0.0009048949160248458",
            "extra": "mean: 2.0667401589824617 msec\nrounds: 1258"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 482.4015734040014,
            "unit": "iter/sec",
            "range": "stddev: 0.0007704431543218839",
            "extra": "mean: 2.0729617296718903 msec\nrounds: 1402"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 406.2612032964063,
            "unit": "iter/sec",
            "range": "stddev: 0.0010571592342383843",
            "extra": "mean: 2.4614705807150496 msec\nrounds: 1846"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 494.94233559505983,
            "unit": "iter/sec",
            "range": "stddev: 0.0006874009454645618",
            "extra": "mean: 2.0204373885247033 msec\nrounds: 1220"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 337.9216863819749,
            "unit": "iter/sec",
            "range": "stddev: 0.000258034553006327",
            "extra": "mean: 2.9592655348838277 msec\nrounds: 387"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137870.08796379698,
            "unit": "iter/sec",
            "range": "stddev: 0.000001219372485077336",
            "extra": "mean: 7.253204917535035 usec\nrounds: 50513"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.961098010893664,
            "unit": "iter/sec",
            "range": "stddev: 0.0005350422468556303",
            "extra": "mean: 91.23173600000219 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5451591110449318,
            "unit": "iter/sec",
            "range": "stddev: 0.013539690051858854",
            "extra": "mean: 1.8343268593333306 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03423823141120983,
            "unit": "iter/sec",
            "range": "stddev: 0.9799169004715635",
            "extra": "mean: 29.207116103333338 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3147.0846607553,
            "unit": "iter/sec",
            "range": "stddev: 0.000013414230683866004",
            "extra": "mean: 317.7544005949939 usec\nrounds: 2354"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 861.1325417098553,
            "unit": "iter/sec",
            "range": "stddev: 0.000020672670535471655",
            "extra": "mean: 1.1612614220970108 msec\nrounds: 706"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 352.6142000411808,
            "unit": "iter/sec",
            "range": "stddev: 0.000032835257948506156",
            "extra": "mean: 2.835960661491264 msec\nrounds: 322"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 445.7101025968061,
            "unit": "iter/sec",
            "range": "stddev: 0.0003793412358227206",
            "extra": "mean: 2.2436108003246455 msec\nrounds: 616"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9708.800098113727,
            "unit": "iter/sec",
            "range": "stddev: 0.000005280937424309394",
            "extra": "mean: 102.99933976334366 usec\nrounds: 6843"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9704.374253977865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000068889992445938114",
            "extra": "mean: 103.0463143556212 usec\nrounds: 7746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1732.498522542008,
            "unit": "iter/sec",
            "range": "stddev: 0.000008539159757340453",
            "extra": "mean: 577.2010694316497 usec\nrounds: 1109"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1727.2275384528443,
            "unit": "iter/sec",
            "range": "stddev: 0.000009598227012962814",
            "extra": "mean: 578.962515208474 usec\nrounds: 1611"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1020.5491170381218,
            "unit": "iter/sec",
            "range": "stddev: 0.00003267163339371312",
            "extra": "mean: 979.8646466935759 usec\nrounds: 968"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.95497619533035,
            "unit": "iter/sec",
            "range": "stddev: 0.00003629588455880503",
            "extra": "mean: 2.703031623696868 msec\nrounds: 1055"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 26.664632561890794,
            "unit": "iter/sec",
            "range": "stddev: 0.0009736301074277221",
            "extra": "mean: 37.502860678050524 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 461.38351307372403,
            "unit": "iter/sec",
            "range": "stddev: 0.000027960924977084578",
            "extra": "mean: 2.1673943079110654 msec\nrounds: 2465"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 375.60799046997545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000349096682472609",
            "extra": "mean: 2.662350177238671 msec\nrounds: 1072"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.11012116398825,
            "unit": "iter/sec",
            "range": "stddev: 0.00003747294209263541",
            "extra": "mean: 3.2667982234546353 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 198.71397929143546,
            "unit": "iter/sec",
            "range": "stddev: 0.000042962791217989896",
            "extra": "mean: 5.032358586777592 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 861.8482581313339,
            "unit": "iter/sec",
            "range": "stddev: 0.00002375234536713962",
            "extra": "mean: 1.1602970599119244 msec\nrounds: 1135"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5130.7363909624055,
            "unit": "iter/sec",
            "range": "stddev: 0.00001099811718356186",
            "extra": "mean: 194.90379621947866 usec\nrounds: 4549"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 77.70121233229,
            "unit": "iter/sec",
            "range": "stddev: 0.00010542410992848554",
            "extra": "mean: 12.869812065782066 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1921.865391016086,
            "unit": "iter/sec",
            "range": "stddev: 0.000018236157363327192",
            "extra": "mean: 520.327804785174 usec\nrounds: 1588"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 29.796419137431453,
            "unit": "iter/sec",
            "range": "stddev: 0.004041348776310995",
            "extra": "mean: 33.561079785716935 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17864.57984380391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053462215959697556",
            "extra": "mean: 55.97668731889245 usec\nrounds: 13800"
          }
        ]
      }
    ]
  }
}