window.BENCHMARK_DATA = {
  "lastUpdate": 1791016083088,
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
            "email": "info@gianlucamazza.it",
            "name": "Gianluca Mazza",
            "username": "gianlucamazza"
          },
          "distinct": true,
          "id": "93a02ba845b56770c52bdb206880964aed631da6",
          "message": "feat(bench): AA wrong-person memory retrieval stress test\n\nDry-run target and checkpoint ignore for the exploratory false-memory bench.",
          "timestamp": "2026-10-03T10:19:59+02:00",
          "tree_id": "44631945527f61284cbdd452e063d6de30aa921b",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/93a02ba845b56770c52bdb206880964aed631da6"
        },
        "date": 1791016082205,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 504.63155805453766,
            "unit": "iter/sec",
            "range": "stddev: 0.0007798576044445074",
            "extra": "mean: 1.9816438033626225 msec\nrounds: 1368"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 503.3728305638761,
            "unit": "iter/sec",
            "range": "stddev: 0.0010115017909989488",
            "extra": "mean: 1.9865990758376932 msec\nrounds: 1134"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 485.6253116883932,
            "unit": "iter/sec",
            "range": "stddev: 0.0007707113097932745",
            "extra": "mean: 2.0592007375465244 msec\nrounds: 1345"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 415.21255925410264,
            "unit": "iter/sec",
            "range": "stddev: 0.0010173148829533735",
            "extra": "mean: 2.408404990919405 msec\nrounds: 1762"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 497.6255760190968,
            "unit": "iter/sec",
            "range": "stddev: 0.001092633945427766",
            "extra": "mean: 2.009543014247371 msec\nrounds: 1123"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 323.9765542810174,
            "unit": "iter/sec",
            "range": "stddev: 0.00040837108815344677",
            "extra": "mean: 3.0866431128611844 msec\nrounds: 381"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136374.70749769086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011933422595270984",
            "extra": "mean: 7.3327380006804574 usec\nrounds: 42565"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.42115887364812,
            "unit": "iter/sec",
            "range": "stddev: 0.000717825188151464",
            "extra": "mean: 95.95861766666758 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.525339694404587,
            "unit": "iter/sec",
            "range": "stddev: 0.04541851187782386",
            "extra": "mean: 1.9035302503333327 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.023482379718214,
            "unit": "iter/sec",
            "range": "stddev: 6.677357388165351",
            "extra": "mean: 42.58512178066666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3174.3954477701473,
            "unit": "iter/sec",
            "range": "stddev: 0.00001695879992405363",
            "extra": "mean: 315.02061304380004 usec\nrounds: 2070"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 875.3429354215124,
            "unit": "iter/sec",
            "range": "stddev: 0.000037497202003341514",
            "extra": "mean: 1.1424094026856575 msec\nrounds: 596"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 351.664026323362,
            "unit": "iter/sec",
            "range": "stddev: 0.00004526722488085748",
            "extra": "mean: 2.8436232459002797 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 427.0072817207013,
            "unit": "iter/sec",
            "range": "stddev: 0.0005143263493211803",
            "extra": "mean: 2.341880438128182 msec\nrounds: 598"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10228.35290036394,
            "unit": "iter/sec",
            "range": "stddev: 0.000005817918614255662",
            "extra": "mean: 97.76745188019652 usec\nrounds: 4468"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10084.32469145819,
            "unit": "iter/sec",
            "range": "stddev: 0.000006546223865220067",
            "extra": "mean: 99.16380428002664 usec\nrounds: 6448"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1769.807705562617,
            "unit": "iter/sec",
            "range": "stddev: 0.000014112175800692639",
            "extra": "mean: 565.0331371351458 usec\nrounds: 824"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1788.1765912341507,
            "unit": "iter/sec",
            "range": "stddev: 0.000015189648439308155",
            "extra": "mean: 559.2288842735758 usec\nrounds: 1011"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1014.581913984937,
            "unit": "iter/sec",
            "range": "stddev: 0.00004505145027338793",
            "extra": "mean: 985.6276622085011 usec\nrounds: 897"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 346.16158246825813,
            "unit": "iter/sec",
            "range": "stddev: 0.00015238594368563433",
            "extra": "mean: 2.8888243255350172 msec\nrounds: 983"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 25.90246941848194,
            "unit": "iter/sec",
            "range": "stddev: 0.0010030791127327299",
            "extra": "mean: 38.60635771222954 msec\nrounds: 278"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 462.5112058600109,
            "unit": "iter/sec",
            "range": "stddev: 0.000043605172235323795",
            "extra": "mean: 2.1621097766497615 msec\nrounds: 2561"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 368.90681464570616,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389217558744852",
            "extra": "mean: 2.710711649391428 msec\nrounds: 984"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 303.49087044646626,
            "unit": "iter/sec",
            "range": "stddev: 0.00003506654634998782",
            "extra": "mean: 3.2949920323102213 msec\nrounds: 619"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.71377028395577,
            "unit": "iter/sec",
            "range": "stddev: 0.000168421575185298",
            "extra": "mean: 5.1095025074072575 msec\nrounds: 270"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 854.5842461130982,
            "unit": "iter/sec",
            "range": "stddev: 0.00002167729856625747",
            "extra": "mean: 1.170159647276785 msec\nrounds: 1083"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5082.554488154251,
            "unit": "iter/sec",
            "range": "stddev: 0.00000930158120481207",
            "extra": "mean: 196.75145683743645 usec\nrounds: 4483"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 77.08167129149466,
            "unit": "iter/sec",
            "range": "stddev: 0.0001381911513955421",
            "extra": "mean: 12.973252697367785 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1879.104875948719,
            "unit": "iter/sec",
            "range": "stddev: 0.000029241994486841134",
            "extra": "mean: 532.1682747989901 usec\nrounds: 1492"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.74954124317334,
            "unit": "iter/sec",
            "range": "stddev: 0.005433105221433683",
            "extra": "mean: 34.78316372221949 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 16930.15728775389,
            "unit": "iter/sec",
            "range": "stddev: 0.000005719670316168544",
            "extra": "mean: 59.06619666926137 usec\nrounds: 13210"
          }
        ]
      }
    ]
  }
}