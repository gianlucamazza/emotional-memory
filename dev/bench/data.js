window.BENCHMARK_DATA = {
  "lastUpdate": 1790939731143,
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
          "id": "1c18efd995e572e1269f20e201f28ede541838c3",
          "message": "docs: add social card for emotional-memory (#143)",
          "timestamp": "2026-10-02T13:09:26+02:00",
          "tree_id": "6dfaae98fc0eee137de40317fb1c64a536ec69ad",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1c18efd995e572e1269f20e201f28ede541838c3"
        },
        "date": 1790939729847,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 484.2808858287666,
            "unit": "iter/sec",
            "range": "stddev: 0.0008081705289553656",
            "extra": "mean: 2.064917342935527 msec\nrounds: 1458"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 504.45050644993825,
            "unit": "iter/sec",
            "range": "stddev: 0.0009041460885997158",
            "extra": "mean: 1.982355032285492 msec\nrounds: 1177"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 480.1737435137679,
            "unit": "iter/sec",
            "range": "stddev: 0.0007930120719516261",
            "extra": "mean: 2.0825795110792584 msec\nrounds: 1399"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 413.50165881214497,
            "unit": "iter/sec",
            "range": "stddev: 0.0010147197878384495",
            "extra": "mean: 2.4183699839866977 msec\nrounds: 1811"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 493.2307344575052,
            "unit": "iter/sec",
            "range": "stddev: 0.0008614577558866802",
            "extra": "mean: 2.027448676935918 msec\nrounds: 1201"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 339.27872307043225,
            "unit": "iter/sec",
            "range": "stddev: 0.00026448826692377915",
            "extra": "mean: 2.947429154855685 msec\nrounds: 381"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137502.98048356097,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011322078256484654",
            "extra": "mean: 7.272569630732869 usec\nrounds: 48312"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.80416058716537,
            "unit": "iter/sec",
            "range": "stddev: 0.0005380850480234216",
            "extra": "mean: 92.55693600000114 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5412397164442604,
            "unit": "iter/sec",
            "range": "stddev: 0.009929013251232662",
            "extra": "mean: 1.8476101616666654 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03314843582963133,
            "unit": "iter/sec",
            "range": "stddev: 0.941183824214836",
            "extra": "mean: 30.167335953333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3156.1760720342018,
            "unit": "iter/sec",
            "range": "stddev: 0.00001384190849672767",
            "extra": "mean: 316.83910440252635 usec\nrounds: 2385"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 856.5576271064323,
            "unit": "iter/sec",
            "range": "stddev: 0.000016694916908994185",
            "extra": "mean: 1.1674637740114877 msec\nrounds: 708"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 350.8397210044625,
            "unit": "iter/sec",
            "range": "stddev: 0.00003519162343077232",
            "extra": "mean: 2.850304398649549 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 443.17652199239546,
            "unit": "iter/sec",
            "range": "stddev: 0.00040118606692535647",
            "extra": "mean: 2.2564372216837767 msec\nrounds: 618"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9924.977443849173,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052561145872195205",
            "extra": "mean: 100.75589649019626 usec\nrounds: 6724"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9752.2192503946,
            "unit": "iter/sec",
            "range": "stddev: 0.000005622407317667543",
            "extra": "mean: 102.54076270481075 usec\nrounds: 7320"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1777.8558526412237,
            "unit": "iter/sec",
            "range": "stddev: 0.000009751408345187107",
            "extra": "mean: 562.475297710091 usec\nrounds: 1048"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1793.222765232837,
            "unit": "iter/sec",
            "range": "stddev: 0.000010513348645313609",
            "extra": "mean: 557.6552001168451 usec\nrounds: 1709"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 999.4802595055122,
            "unit": "iter/sec",
            "range": "stddev: 0.00003342295621043661",
            "extra": "mean: 1.00052001076514 msec\nrounds: 929"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 366.9076220433275,
            "unit": "iter/sec",
            "range": "stddev: 0.00003762613389783865",
            "extra": "mean: 2.7254816741907635 msec\nrounds: 1019"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 25.226300268652125,
            "unit": "iter/sec",
            "range": "stddev: 0.0012284745801429532",
            "extra": "mean: 39.64116772377701 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 459.0462417813637,
            "unit": "iter/sec",
            "range": "stddev: 0.000040834843364526194",
            "extra": "mean: 2.1784297723894315 msec\nrounds: 2289"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 371.4259030554195,
            "unit": "iter/sec",
            "range": "stddev: 0.00002972165858902857",
            "extra": "mean: 2.6923270342046997 msec\nrounds: 994"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 301.23650194193516,
            "unit": "iter/sec",
            "range": "stddev: 0.00003092759749938481",
            "extra": "mean: 3.3196508177244572 msec\nrounds: 598"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 194.59759181643307,
            "unit": "iter/sec",
            "range": "stddev: 0.00003719941580617389",
            "extra": "mean: 5.138809738937137 msec\nrounds: 226"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 846.0621600131367,
            "unit": "iter/sec",
            "range": "stddev: 0.0000384148614550318",
            "extra": "mean: 1.181946253197842 msec\nrounds: 1094"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5022.013455268272,
            "unit": "iter/sec",
            "range": "stddev: 0.000012861159496771311",
            "extra": "mean: 199.1233215337096 usec\nrounds: 4407"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 75.9213770612573,
            "unit": "iter/sec",
            "range": "stddev: 0.00005793023098599048",
            "extra": "mean: 13.171520837841866 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1855.3072596386567,
            "unit": "iter/sec",
            "range": "stddev: 0.000017480279369420293",
            "extra": "mean: 538.9942796832272 usec\nrounds: 1516"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.418358571192112,
            "unit": "iter/sec",
            "range": "stddev: 0.005018315297821678",
            "extra": "mean: 35.18852074073367 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17424.3894042745,
            "unit": "iter/sec",
            "range": "stddev: 0.000005630941186694035",
            "extra": "mean: 57.390820234692576 usec\nrounds: 13551"
          }
        ]
      }
    ]
  }
}