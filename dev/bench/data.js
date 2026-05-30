window.BENCHMARK_DATA = {
  "lastUpdate": 1780175297363,
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
          "id": "a26e4cca059c89762c2fa57bf4ee80a5e49500c1",
          "message": "chore(release): v0.11.3\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T22:59:11+02:00",
          "tree_id": "abe40895b68afd3506f2c94006c7b7e613037863",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/a26e4cca059c89762c2fa57bf4ee80a5e49500c1"
        },
        "date": 1780175296173,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 533.0161681108118,
            "unit": "iter/sec",
            "range": "stddev: 0.0008918192332865301",
            "extra": "mean: 1.8761156974737478 msec\nrounds: 1623"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 429.093092785095,
            "unit": "iter/sec",
            "range": "stddev: 0.0011741342118466228",
            "extra": "mean: 2.3304966144044537 msec\nrounds: 1805"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 345.6556124668134,
            "unit": "iter/sec",
            "range": "stddev: 0.00181197247742025",
            "extra": "mean: 2.893052980865487 msec\nrounds: 2195"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 198.46140366690508,
            "unit": "iter/sec",
            "range": "stddev: 0.0036799774024059884",
            "extra": "mean: 5.038763112239125 msec\nrounds: 3350"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 435.75803713462113,
            "unit": "iter/sec",
            "range": "stddev: 0.0010535032524364963",
            "extra": "mean: 2.294851534066059 msec\nrounds: 1820"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 369.6002688166325,
            "unit": "iter/sec",
            "range": "stddev: 0.0003342714131711189",
            "extra": "mean: 2.70562573777814 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 133617.6469333268,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012489816671530719",
            "extra": "mean: 7.484041389375649 usec\nrounds: 43876"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.977830404490216,
            "unit": "iter/sec",
            "range": "stddev: 0.00028348788214350345",
            "extra": "mean: 35.74258566666799 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.798265379123219,
            "unit": "iter/sec",
            "range": "stddev: 0.01718384776320042",
            "extra": "mean: 1.252716234666669 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02695566728901283,
            "unit": "iter/sec",
            "range": "stddev: 2.48837118360658",
            "extra": "mean: 37.09795009999999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3193.3335640653663,
            "unit": "iter/sec",
            "range": "stddev: 0.000015151490894726906",
            "extra": "mean: 313.1523782084703 usec\nrounds: 2221"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 880.2622772751279,
            "unit": "iter/sec",
            "range": "stddev: 0.00002540462076338845",
            "extra": "mean: 1.136025052778046 msec\nrounds: 720"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.9299123454826,
            "unit": "iter/sec",
            "range": "stddev: 0.00004258692491918837",
            "extra": "mean: 2.770621014760336 msec\nrounds: 271"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.1705100327442,
            "unit": "iter/sec",
            "range": "stddev: 0.0009363217971244206",
            "extra": "mean: 2.031816168614958 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10519.206639787697,
            "unit": "iter/sec",
            "range": "stddev: 0.000005329639287349558",
            "extra": "mean: 95.06420343694117 usec\nrounds: 6808"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10394.249786408522,
            "unit": "iter/sec",
            "range": "stddev: 0.000005331690995899899",
            "extra": "mean: 96.20703952175518 usec\nrounds: 7439"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1828.3990927467396,
            "unit": "iter/sec",
            "range": "stddev: 0.000012220420658842852",
            "extra": "mean: 546.9265457235243 usec\nrounds: 1017"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1804.4362198611298,
            "unit": "iter/sec",
            "range": "stddev: 0.000011574603838527855",
            "extra": "mean: 554.1897180920921 usec\nrounds: 1426"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1009.5853098660806,
            "unit": "iter/sec",
            "range": "stddev: 0.00005084320362470269",
            "extra": "mean: 990.5056959799145 usec\nrounds: 796"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 372.3654579200152,
            "unit": "iter/sec",
            "range": "stddev: 0.0003384160440667747",
            "extra": "mean: 2.685533737704537 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.25081905160739,
            "unit": "iter/sec",
            "range": "stddev: 0.0007322810907794378",
            "extra": "mean: 47.05700978261169 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 460.51311882919686,
            "unit": "iter/sec",
            "range": "stddev: 0.00010329094051381041",
            "extra": "mean: 2.1714907982260923 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 364.7756677054381,
            "unit": "iter/sec",
            "range": "stddev: 0.00011875967790021236",
            "extra": "mean: 2.741410923295233 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 288.40934262292603,
            "unit": "iter/sec",
            "range": "stddev: 0.00017191221596938872",
            "extra": "mean: 3.467294058179753 msec\nrounds: 275"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 184.59071080105193,
            "unit": "iter/sec",
            "range": "stddev: 0.0002708350409347869",
            "extra": "mean: 5.417390699999955 msec\nrounds: 170"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 819.70865568444,
            "unit": "iter/sec",
            "range": "stddev: 0.000028068406753155334",
            "extra": "mean: 1.2199456393015886 msec\nrounds: 743"
          }
        ]
      }
    ]
  }
}