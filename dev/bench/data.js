window.BENCHMARK_DATA = {
  "lastUpdate": 1786354777149,
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
          "id": "f23c87f5debc27c3aa18d7a123fa58ae002115e8",
          "message": "chore(release): v0.18.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.21870707\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-08-10T11:30:28+02:00",
          "tree_id": "761e82a7dc64606d7868c3707ce14b0301df0edc",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/f23c87f5debc27c3aa18d7a123fa58ae002115e8"
        },
        "date": 1786354775489,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 540.5233771921091,
            "unit": "iter/sec",
            "range": "stddev: 0.0008280213516884939",
            "extra": "mean: 1.8500587434252393 msec\nrounds: 1559"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 447.97073316188664,
            "unit": "iter/sec",
            "range": "stddev: 0.0010077318284976214",
            "extra": "mean: 2.2322886875706285 msec\nrounds: 1770"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 375.81785525431746,
            "unit": "iter/sec",
            "range": "stddev: 0.001445690042998664",
            "extra": "mean: 2.660863463560814 msec\nrounds: 2168"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 193.7999824817338,
            "unit": "iter/sec",
            "range": "stddev: 0.00362305608965121",
            "extra": "mean: 5.159959186757166 msec\nrounds: 3534"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 445.4998024961668,
            "unit": "iter/sec",
            "range": "stddev: 0.0010200078099954278",
            "extra": "mean: 2.2446699064666906 msec\nrounds: 1732"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 371.9867155040947,
            "unit": "iter/sec",
            "range": "stddev: 0.0002602592591970141",
            "extra": "mean: 2.6882680437790856 msec\nrounds: 434"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135250.1557809793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011792199748593534",
            "extra": "mean: 7.393706825886212 usec\nrounds: 36845"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.377201910703715,
            "unit": "iter/sec",
            "range": "stddev: 0.00029393969518166474",
            "extra": "mean: 36.526742333336415 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7912305902897547,
            "unit": "iter/sec",
            "range": "stddev: 0.002618281307070949",
            "extra": "mean: 1.2638540676666612 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.024453601746374003,
            "unit": "iter/sec",
            "range": "stddev: 0.9872848465188719",
            "extra": "mean: 40.89377141133333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3119.3849721079546,
            "unit": "iter/sec",
            "range": "stddev: 0.000013289034471467284",
            "extra": "mean: 320.57601384295964 usec\nrounds: 2095"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 856.572940390008,
            "unit": "iter/sec",
            "range": "stddev: 0.00002518871521967502",
            "extra": "mean: 1.167442902813026 msec\nrounds: 782"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 351.99139808135016,
            "unit": "iter/sec",
            "range": "stddev: 0.00004721515059376946",
            "extra": "mean: 2.840978516665018 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 484.48596087292236,
            "unit": "iter/sec",
            "range": "stddev: 0.0008880016206782539",
            "extra": "mean: 2.0640432969373363 msec\nrounds: 751"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10592.495011500505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075791226785466245",
            "extra": "mean: 94.40646409691749 usec\nrounds: 6434"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10850.218456587118,
            "unit": "iter/sec",
            "range": "stddev: 0.00000562496526757727",
            "extra": "mean: 92.16404296384508 usec\nrounds: 7518"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1839.8533163578325,
            "unit": "iter/sec",
            "range": "stddev: 0.000013331679653620294",
            "extra": "mean: 543.5215900687108 usec\nrounds: 1027"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1822.6397744377139,
            "unit": "iter/sec",
            "range": "stddev: 0.00001358037437781911",
            "extra": "mean: 548.6547665780536 usec\nrounds: 1508"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 995.8326927231248,
            "unit": "iter/sec",
            "range": "stddev: 0.00002172996408161577",
            "extra": "mean: 1.0041847464010039 msec\nrounds: 903"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 361.5023311777073,
            "unit": "iter/sec",
            "range": "stddev: 0.00009535922529477174",
            "extra": "mean: 2.76623389050407 msec\nrounds: 1032"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.331874648767553,
            "unit": "iter/sec",
            "range": "stddev: 0.0032565616980310573",
            "extra": "mean: 44.77904411196343 msec\nrounds: 652"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 458.06647711704363,
            "unit": "iter/sec",
            "range": "stddev: 0.00005507622136817504",
            "extra": "mean: 2.1830892456783806 msec\nrounds: 2430"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 361.4269403411473,
            "unit": "iter/sec",
            "range": "stddev: 0.00023503795713150375",
            "extra": "mean: 2.7668109052858925 msec\nrounds: 1003"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 284.1667679431107,
            "unit": "iter/sec",
            "range": "stddev: 0.00026379034749760195",
            "extra": "mean: 3.5190603293914964 msec\nrounds: 592"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 179.32052076118416,
            "unit": "iter/sec",
            "range": "stddev: 0.0005791778577318748",
            "extra": "mean: 5.5766066022738245 msec\nrounds: 264"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 831.9413596382304,
            "unit": "iter/sec",
            "range": "stddev: 0.000029045134455761155",
            "extra": "mean: 1.2020077958798079 msec\nrounds: 1068"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5054.0610887530875,
            "unit": "iter/sec",
            "range": "stddev: 0.000014149235811538415",
            "extra": "mean: 197.86068716607357 usec\nrounds: 4488"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 72.46224826784864,
            "unit": "iter/sec",
            "range": "stddev: 0.00022212946773924965",
            "extra": "mean: 13.800289445941718 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1817.6398105796688,
            "unit": "iter/sec",
            "range": "stddev: 0.000025412107297927487",
            "extra": "mean: 550.1640061905813 usec\nrounds: 1454"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 24.316441654334675,
            "unit": "iter/sec",
            "range": "stddev: 0.005855462260463755",
            "extra": "mean: 41.12443811538268 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 16939.095736076823,
            "unit": "iter/sec",
            "range": "stddev: 0.0000062766460572265126",
            "extra": "mean: 59.03502852694809 usec\nrounds: 13496"
          }
        ]
      }
    ]
  }
}