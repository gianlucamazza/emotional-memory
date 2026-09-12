window.BENCHMARK_DATA = {
  "lastUpdate": 1789204463787,
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
          "id": "7ca002bdd361c7a12bd3aa6328a8fee20309685c",
          "message": "fix(security): remove vulnerable embedded chromadb",
          "timestamp": "2026-09-12T11:08:39+02:00",
          "tree_id": "61912d478934225f6043d58aebcaaef2fa1caba8",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7ca002bdd361c7a12bd3aa6328a8fee20309685c"
        },
        "date": 1789204460983,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 500.9154595743238,
            "unit": "iter/sec",
            "range": "stddev: 0.0008172167024448807",
            "extra": "mean: 1.9963448539795448 msec\nrounds: 1671"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 508.4249789773638,
            "unit": "iter/sec",
            "range": "stddev: 0.0008882699978304485",
            "extra": "mean: 1.9668585166908612 msec\nrounds: 1378"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 461.48506070027423,
            "unit": "iter/sec",
            "range": "stddev: 0.0009009879386265443",
            "extra": "mean: 2.1669173829431525 msec\nrounds: 1794"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 364.3631771181224,
            "unit": "iter/sec",
            "range": "stddev: 0.0014068543113861707",
            "extra": "mean: 2.74451443724186 msec\nrounds: 2422"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 503.7202576644518,
            "unit": "iter/sec",
            "range": "stddev: 0.0007446246894880897",
            "extra": "mean: 1.9852288741306487 msec\nrounds: 1438"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 366.5984509079369,
            "unit": "iter/sec",
            "range": "stddev: 0.0012235757425311283",
            "extra": "mean: 2.7277802116275933 msec\nrounds: 430"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 175838.77274654031,
            "unit": "iter/sec",
            "range": "stddev: 8.281737572788629e-7",
            "extra": "mean: 5.68702786296986 usec\nrounds: 43678"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 13.797908546138393,
            "unit": "iter/sec",
            "range": "stddev: 0.0009158319739956329",
            "extra": "mean: 72.4747519999956 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.6259749757168095,
            "unit": "iter/sec",
            "range": "stddev: 0.01759588110939845",
            "extra": "mean: 1.5975079496666638 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03910923603968221,
            "unit": "iter/sec",
            "range": "stddev: 0.5887306487294188",
            "extra": "mean: 25.569407671000004 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4195.332605794406,
            "unit": "iter/sec",
            "range": "stddev: 0.000009641359874905029",
            "extra": "mean: 238.3601239670115 usec\nrounds: 2904"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1086.4957417062358,
            "unit": "iter/sec",
            "range": "stddev: 0.000013656170979325419",
            "extra": "mean: 920.3901696196226 usec\nrounds: 790"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 421.9089232544227,
            "unit": "iter/sec",
            "range": "stddev: 0.000026111660719657173",
            "extra": "mean: 2.3701797826090836 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 476.31345754811554,
            "unit": "iter/sec",
            "range": "stddev: 0.00041553462674698895",
            "extra": "mean: 2.099457792243847 msec\nrounds: 722"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13999.741266692261,
            "unit": "iter/sec",
            "range": "stddev: 0.000003857753349555379",
            "extra": "mean: 71.42989152086462 usec\nrounds: 8527"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13397.763286342122,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029770169000318895",
            "extra": "mean: 74.63932438778156 usec\nrounds: 8533"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2401.693825815634,
            "unit": "iter/sec",
            "range": "stddev: 0.000011936833091461909",
            "extra": "mean: 416.3728070793504 usec\nrounds: 1130"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2383.4503983703044,
            "unit": "iter/sec",
            "range": "stddev: 0.000012126277287917952",
            "extra": "mean: 419.55981155880346 usec\nrounds: 1661"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1414.3709117481606,
            "unit": "iter/sec",
            "range": "stddev: 0.00003067508994679879",
            "extra": "mean: 707.0281152516078 usec\nrounds: 1154"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 436.12279013559765,
            "unit": "iter/sec",
            "range": "stddev: 0.00004231454722886549",
            "extra": "mean: 2.2929322260115867 msec\nrounds: 1261"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 31.39531021790916,
            "unit": "iter/sec",
            "range": "stddev: 0.003132480346838133",
            "extra": "mean: 31.85189103274283 msec\nrounds: 733"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 538.2519054372173,
            "unit": "iter/sec",
            "range": "stddev: 0.00003927356153380243",
            "extra": "mean: 1.8578661587601977 msec\nrounds: 2551"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 447.61946883672135,
            "unit": "iter/sec",
            "range": "stddev: 0.00004329897106803751",
            "extra": "mean: 2.234040450918749 msec\nrounds: 1253"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 366.5090477180123,
            "unit": "iter/sec",
            "range": "stddev: 0.00005713825446145552",
            "extra": "mean: 2.72844560380236 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 240.4647695508132,
            "unit": "iter/sec",
            "range": "stddev: 0.00004380249604039843",
            "extra": "mean: 4.15861334642906 msec\nrounds: 280"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1150.2795591709914,
            "unit": "iter/sec",
            "range": "stddev: 0.00003325224325343861",
            "extra": "mean: 869.3538818691186 usec\nrounds: 1456"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 6675.841941200387,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055500348707251744",
            "extra": "mean: 149.79384005909967 usec\nrounds: 5402"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 98.90033764743664,
            "unit": "iter/sec",
            "range": "stddev: 0.00005785157116412164",
            "extra": "mean: 10.111188938149379 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2700.806598432101,
            "unit": "iter/sec",
            "range": "stddev: 0.000010751239435859578",
            "extra": "mean: 370.25975891073796 usec\nrounds: 2020"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 36.64372765477008,
            "unit": "iter/sec",
            "range": "stddev: 0.005733320418845199",
            "extra": "mean: 27.289800028568475 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 30877.214295946418,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024677974796827113",
            "extra": "mean: 32.38634128115892 usec\nrounds: 19251"
          }
        ]
      }
    ]
  }
}