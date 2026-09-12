window.BENCHMARK_DATA = {
  "lastUpdate": 1789208413009,
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
          "id": "7d886549974a48d3857e6fe0d182d1980030d623",
          "message": "chore(release): prepare v0.18.1 (#134)",
          "timestamp": "2026-09-12T12:13:13+02:00",
          "tree_id": "36745c68e2b45dae043dc5a46f1ad21362fbbfcc",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7d886549974a48d3857e6fe0d182d1980030d623"
        },
        "date": 1789208411508,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 502.53854511233436,
            "unit": "iter/sec",
            "range": "stddev: 0.0007932760803791809",
            "extra": "mean: 1.9898971128204028 msec\nrounds: 1365"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 488.23598310223093,
            "unit": "iter/sec",
            "range": "stddev: 0.0011683969405487016",
            "extra": "mean: 2.0481898807335788 msec\nrounds: 1199"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 476.5461829732988,
            "unit": "iter/sec",
            "range": "stddev: 0.0008163205625962127",
            "extra": "mean: 2.0984325039825795 msec\nrounds: 1381"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 370.5051021475654,
            "unit": "iter/sec",
            "range": "stddev: 0.0012880219335033999",
            "extra": "mean: 2.6990181625129637 msec\nrounds: 1926"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 488.6593786573763,
            "unit": "iter/sec",
            "range": "stddev: 0.0009484457773871569",
            "extra": "mean: 2.046415240709317 msec\nrounds: 1184"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 315.02542416051034,
            "unit": "iter/sec",
            "range": "stddev: 0.00047726012990187736",
            "extra": "mean: 3.1743469679148326 msec\nrounds: 374"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136856.88004079217,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011289417006307945",
            "extra": "mean: 7.306903384776385 usec\nrounds: 38141"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.609299083889573,
            "unit": "iter/sec",
            "range": "stddev: 0.0008764315844519757",
            "extra": "mean: 94.25693366666603 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5289175186017854,
            "unit": "iter/sec",
            "range": "stddev: 0.01926806257296466",
            "extra": "mean: 1.890653957999992 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.025125142351126444,
            "unit": "iter/sec",
            "range": "stddev: 1.2403089162127192",
            "extra": "mean: 39.80076952499999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3184.5512637010866,
            "unit": "iter/sec",
            "range": "stddev: 0.000014247267304151789",
            "extra": "mean: 314.01598441778566 usec\nrounds: 2182"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 871.9129169572634,
            "unit": "iter/sec",
            "range": "stddev: 0.00002457374225021972",
            "extra": "mean: 1.1469035273496409 msec\nrounds: 713"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 356.24551463213476,
            "unit": "iter/sec",
            "range": "stddev: 0.00003132534134293978",
            "extra": "mean: 2.807052886076663 msec\nrounds: 237"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 434.2815316251289,
            "unit": "iter/sec",
            "range": "stddev: 0.0006117301768574989",
            "extra": "mean: 2.302653756096629 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9888.028207704734,
            "unit": "iter/sec",
            "range": "stddev: 0.000005848626942191204",
            "extra": "mean: 101.1323975816333 usec\nrounds: 5541"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9972.369057109849,
            "unit": "iter/sec",
            "range": "stddev: 0.000005972231406072053",
            "extra": "mean: 100.27707501328837 usec\nrounds: 7652"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1815.3280312560262,
            "unit": "iter/sec",
            "range": "stddev: 0.00001008013707545831",
            "extra": "mean: 550.8646276497475 usec\nrounds: 1085"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1833.0046607611391,
            "unit": "iter/sec",
            "range": "stddev: 0.000012288961445706703",
            "extra": "mean: 545.552349869508 usec\nrounds: 1532"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1016.1105475586443,
            "unit": "iter/sec",
            "range": "stddev: 0.00003608243508065445",
            "extra": "mean: 984.1448869934947 usec\nrounds: 938"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.3243675536604,
            "unit": "iter/sec",
            "range": "stddev: 0.0002032515376130598",
            "extra": "mean: 2.707646957128293 msec\nrounds: 1003"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.300760015498067,
            "unit": "iter/sec",
            "range": "stddev: 0.0024775185649592753",
            "extra": "mean: 44.841521064979084 msec\nrounds: 277"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 455.7737094831905,
            "unit": "iter/sec",
            "range": "stddev: 0.0001611833305948917",
            "extra": "mean: 2.1940712664930078 msec\nrounds: 2304"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 366.20185295921857,
            "unit": "iter/sec",
            "range": "stddev: 0.0002566569418578385",
            "extra": "mean: 2.730734407592862 msec\nrounds: 1001"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 296.8831066877281,
            "unit": "iter/sec",
            "range": "stddev: 0.00025765855430699957",
            "extra": "mean: 3.3683290745533547 msec\nrounds: 617"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.13768559257062,
            "unit": "iter/sec",
            "range": "stddev: 0.00010004058023645515",
            "extra": "mean: 5.124586760180743 msec\nrounds: 221"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 854.6095826595713,
            "unit": "iter/sec",
            "range": "stddev: 0.00002347510225872991",
            "extra": "mean: 1.1701249556411117 msec\nrounds: 1037"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5044.740270041304,
            "unit": "iter/sec",
            "range": "stddev: 0.0000084852152351393",
            "extra": "mean: 198.22626071328196 usec\nrounds: 4457"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 70.46607577568834,
            "unit": "iter/sec",
            "range": "stddev: 0.000628018523956664",
            "extra": "mean: 14.191225905402444 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1880.6395146809054,
            "unit": "iter/sec",
            "range": "stddev: 0.000029465376075070194",
            "extra": "mean: 531.7340150484255 usec\nrounds: 1462"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 24.85256447255955,
            "unit": "iter/sec",
            "range": "stddev: 0.006652228616341304",
            "extra": "mean: 40.237296279992734 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 18285.96171071352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000066730244581516645",
            "extra": "mean: 54.686760030461635 usec\nrounds: 13235"
          }
        ]
      }
    ]
  }
}