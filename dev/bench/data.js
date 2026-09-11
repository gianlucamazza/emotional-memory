window.BENCHMARK_DATA = {
  "lastUpdate": 1789117063865,
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
          "id": "98197ec73baa7e54f2e2ade9209bb747b8db85b8",
          "message": "docs: record arXiv endorsement block; keep issue #31 open\n\nThe 2026-01-21 policy requires endorsement in every category, including\ncs.LG. SUBMISSION.md, ARXIV_CHECKLIST.md, and ROADMAP.md no longer claim\na no-endorsement path. Bundle stays ready; software citation is Zenodo.",
          "timestamp": "2026-09-11T10:48:53+02:00",
          "tree_id": "9946e806039b2f227c469bfea8cebb3ca5207dd7",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/98197ec73baa7e54f2e2ade9209bb747b8db85b8"
        },
        "date": 1789117062595,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 526.2103290108134,
            "unit": "iter/sec",
            "range": "stddev: 0.000933135854447119",
            "extra": "mean: 1.9003807885714274 msec\nrounds: 1575"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 370.5202605472214,
            "unit": "iter/sec",
            "range": "stddev: 0.001510340944645924",
            "extra": "mean: 2.6989077426511034 msec\nrounds: 1803"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 329.34224458940884,
            "unit": "iter/sec",
            "range": "stddev: 0.002012917198424719",
            "extra": "mean: 3.03635508784092 msec\nrounds: 2163"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 175.75296286600826,
            "unit": "iter/sec",
            "range": "stddev: 0.003922446870794562",
            "extra": "mean: 5.689804505670762 msec\nrounds: 3615"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 400.48933687645615,
            "unit": "iter/sec",
            "range": "stddev: 0.0013676584201855035",
            "extra": "mean: 2.496945381366002 msec\nrounds: 1728"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 345.0479681982452,
            "unit": "iter/sec",
            "range": "stddev: 0.00047872153341633713",
            "extra": "mean: 2.8981477712265677 msec\nrounds: 424"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136557.3694648621,
            "unit": "iter/sec",
            "range": "stddev: 0.000001889130193645557",
            "extra": "mean: 7.322929578379966 usec\nrounds: 30999"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.233870481488196,
            "unit": "iter/sec",
            "range": "stddev: 0.0005729584543279525",
            "extra": "mean: 36.71898199999646 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7742324919817423,
            "unit": "iter/sec",
            "range": "stddev: 0.009903628311383217",
            "extra": "mean: 1.2916016963333306 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02135575853054308,
            "unit": "iter/sec",
            "range": "stddev: 0.5585077235728689",
            "extra": "mean: 46.82577762666667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3150.7905440734826,
            "unit": "iter/sec",
            "range": "stddev: 0.000014877285735382261",
            "extra": "mean: 317.3806655859629 usec\nrounds: 2159"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.1807641815539,
            "unit": "iter/sec",
            "range": "stddev: 0.000017966946953554423",
            "extra": "mean: 1.1598495832243187 msec\nrounds: 763"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 347.88393287812914,
            "unit": "iter/sec",
            "range": "stddev: 0.0000580792810798611",
            "extra": "mean: 2.8745219468078176 msec\nrounds: 282"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 446.2681797829765,
            "unit": "iter/sec",
            "range": "stddev: 0.0010513872036294184",
            "extra": "mean: 2.240805070364433 msec\nrounds: 739"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10660.057239375143,
            "unit": "iter/sec",
            "range": "stddev: 0.00000531062912930245",
            "extra": "mean: 93.80812668681473 usec\nrounds: 4594"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10516.788709541379,
            "unit": "iter/sec",
            "range": "stddev: 0.0000057424510759028754",
            "extra": "mean: 95.08605978674345 usec\nrounds: 6105"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1780.4310174397162,
            "unit": "iter/sec",
            "range": "stddev: 0.00001548825101169191",
            "extra": "mean: 561.6617494330185 usec\nrounds: 882"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1801.5876932089545,
            "unit": "iter/sec",
            "range": "stddev: 0.000015114422601157415",
            "extra": "mean: 555.0659586371944 usec\nrounds: 1233"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1002.0002239317158,
            "unit": "iter/sec",
            "range": "stddev: 0.000029396657714184515",
            "extra": "mean: 998.003768977349 usec\nrounds: 909"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 354.2042064809001,
            "unit": "iter/sec",
            "range": "stddev: 0.00016916525558471562",
            "extra": "mean: 2.8232301641339297 msec\nrounds: 658"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.363160158953423,
            "unit": "iter/sec",
            "range": "stddev: 0.0007318913448353982",
            "extra": "mean: 46.8095540434777 msec\nrounds: 598"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 428.2328675249652,
            "unit": "iter/sec",
            "range": "stddev: 0.00014416751032423697",
            "extra": "mean: 2.33517806743711 msec\nrounds: 2595"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 321.7140131244939,
            "unit": "iter/sec",
            "range": "stddev: 0.0003109204497415367",
            "extra": "mean: 3.108350768709069 msec\nrounds: 1029"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 265.67817377668365,
            "unit": "iter/sec",
            "range": "stddev: 0.00019552823132937885",
            "extra": "mean: 3.76395240069872 msec\nrounds: 574"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 158.80180918951743,
            "unit": "iter/sec",
            "range": "stddev: 0.0003644023451567645",
            "extra": "mean: 6.2971574763772304 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 815.7833139439163,
            "unit": "iter/sec",
            "range": "stddev: 0.000034604216080664425",
            "extra": "mean: 1.2258157073175298 msec\nrounds: 1066"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5158.294969615676,
            "unit": "iter/sec",
            "range": "stddev: 0.00001258243963564899",
            "extra": "mean: 193.86250803615948 usec\nrounds: 4604"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 70.61608254426794,
            "unit": "iter/sec",
            "range": "stddev: 0.00016165276010595835",
            "extra": "mean: 14.161080082191166 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1837.6057715327265,
            "unit": "iter/sec",
            "range": "stddev: 0.00002655082576222509",
            "extra": "mean: 544.1863622173494 usec\nrounds: 1535"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 24.431066576020555,
            "unit": "iter/sec",
            "range": "stddev: 0.006039814656036405",
            "extra": "mean: 40.931491749996475 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17458.796656087146,
            "unit": "iter/sec",
            "range": "stddev: 0.000005756612048484205",
            "extra": "mean: 57.27771619651358 usec\nrounds: 13009"
          }
        ]
      }
    ]
  }
}