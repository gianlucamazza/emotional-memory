window.BENCHMARK_DATA = {
  "lastUpdate": 1789207644737,
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
          "id": "bd170a6e1b05254684058442ebf715a081458140",
          "message": "feat: add public memory update API (#133)",
          "timestamp": "2026-09-12T12:01:18+02:00",
          "tree_id": "cbbbb4d1ebbb22ee6fb6cd180e81174822d1a3e0",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/bd170a6e1b05254684058442ebf715a081458140"
        },
        "date": 1789207643750,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 472.6378197496764,
            "unit": "iter/sec",
            "range": "stddev: 0.0008159997368478534",
            "extra": "mean: 2.115784979986644 msec\nrounds: 1499"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 494.9545418637941,
            "unit": "iter/sec",
            "range": "stddev: 0.0008831568965572205",
            "extra": "mean: 2.020387561723171 msec\nrounds: 1207"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 473.57767797967904,
            "unit": "iter/sec",
            "range": "stddev: 0.0007932001887404529",
            "extra": "mean: 2.1115860111187703 msec\nrounds: 1439"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 426.0649892219683,
            "unit": "iter/sec",
            "range": "stddev: 0.0009739031716885518",
            "extra": "mean: 2.3470597803074287 msec\nrounds: 1757"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 486.27406637491003,
            "unit": "iter/sec",
            "range": "stddev: 0.0008697638391666913",
            "extra": "mean: 2.056453488163226 msec\nrounds: 1225"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 337.22945854920755,
            "unit": "iter/sec",
            "range": "stddev: 0.00027175285929451754",
            "extra": "mean: 2.965339992247691 msec\nrounds: 387"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138289.95908834634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013198485709245037",
            "extra": "mean: 7.231182991103147 usec\nrounds: 47822"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.930436541268502,
            "unit": "iter/sec",
            "range": "stddev: 0.000774229079593273",
            "extra": "mean: 91.48765433333257 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5435591370532616,
            "unit": "iter/sec",
            "range": "stddev: 0.01520212561588134",
            "extra": "mean: 1.8397262263333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03104307603677531,
            "unit": "iter/sec",
            "range": "stddev: 2.092630466326335",
            "extra": "mean: 32.21330253533335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3175.201207808443,
            "unit": "iter/sec",
            "range": "stddev: 0.000012476023724239343",
            "extra": "mean: 314.9406713315691 usec\nrounds: 2358"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 857.3645890926484,
            "unit": "iter/sec",
            "range": "stddev: 0.000016961254011185088",
            "extra": "mean: 1.1663649428982168 msec\nrounds: 683"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 347.4978021502679,
            "unit": "iter/sec",
            "range": "stddev: 0.00004668609316380814",
            "extra": "mean: 2.877716042553765 msec\nrounds: 47"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 442.392450798177,
            "unit": "iter/sec",
            "range": "stddev: 0.0003966368216933003",
            "extra": "mean: 2.2604364025556305 msec\nrounds: 626"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9841.343347657143,
            "unit": "iter/sec",
            "range": "stddev: 0.00000608996299828002",
            "extra": "mean: 101.61214426464075 usec\nrounds: 5684"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9802.554807487939,
            "unit": "iter/sec",
            "range": "stddev: 0.000005561459899899919",
            "extra": "mean: 102.01422176554664 usec\nrounds: 7544"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1740.6594148015856,
            "unit": "iter/sec",
            "range": "stddev: 0.000011428694118959738",
            "extra": "mean: 574.4949250247143 usec\nrounds: 1027"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1764.2066396247535,
            "unit": "iter/sec",
            "range": "stddev: 0.000011972779372301534",
            "extra": "mean: 566.827024419713 usec\nrounds: 1638"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1007.2080740786482,
            "unit": "iter/sec",
            "range": "stddev: 0.00003506733799495912",
            "extra": "mean: 992.8435104283274 usec\nrounds: 911"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 366.58871931215145,
            "unit": "iter/sec",
            "range": "stddev: 0.00003491321716515813",
            "extra": "mean: 2.7278526242606413 msec\nrounds: 1014"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.97790543415306,
            "unit": "iter/sec",
            "range": "stddev: 0.0015145895360852476",
            "extra": "mean: 40.035382575861185 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 458.7466432222303,
            "unit": "iter/sec",
            "range": "stddev: 0.00003750488292593383",
            "extra": "mean: 2.179852462736323 msec\nrounds: 2375"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 371.8016416063171,
            "unit": "iter/sec",
            "range": "stddev: 0.00005762628658127236",
            "extra": "mean: 2.6896061988312896 msec\nrounds: 1026"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 298.6944544479511,
            "unit": "iter/sec",
            "range": "stddev: 0.00007959737796548561",
            "extra": "mean: 3.347902798691747 msec\nrounds: 611"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.4551662919757,
            "unit": "iter/sec",
            "range": "stddev: 0.0000781317012146143",
            "extra": "mean: 5.116262818585084 msec\nrounds: 226"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 849.9909931732684,
            "unit": "iter/sec",
            "range": "stddev: 0.000036354389952204755",
            "extra": "mean: 1.1764830545635587 msec\nrounds: 1063"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5083.683463356451,
            "unit": "iter/sec",
            "range": "stddev: 0.000009263882402125556",
            "extra": "mean: 196.7077626307913 usec\nrounds: 4394"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 71.97793665780253,
            "unit": "iter/sec",
            "range": "stddev: 0.0007069036519346868",
            "extra": "mean: 13.893146239440004 msec\nrounds: 71"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1859.5446736507924,
            "unit": "iter/sec",
            "range": "stddev: 0.000020212599194184774",
            "extra": "mean: 537.7660532547077 usec\nrounds: 1521"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.301933517341123,
            "unit": "iter/sec",
            "range": "stddev: 0.004484820236021118",
            "extra": "mean: 35.33327500000243 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17218.798631556947,
            "unit": "iter/sec",
            "range": "stddev: 0.000006103326568553727",
            "extra": "mean: 58.07606101898983 usec\nrounds: 13422"
          }
        ]
      }
    ]
  }
}