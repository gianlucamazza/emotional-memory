window.BENCHMARK_DATA = {
  "lastUpdate": 1786355171995,
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
          "id": "b5101cd65d814efce42dd75a43797cac2d761beb",
          "message": "docs: record v0.18.0 DOI in roadmap and arXiv checklist\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-10T11:39:45+02:00",
          "tree_id": "2744cacac16184c13f78c2a9cbbcd05eff211828",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/b5101cd65d814efce42dd75a43797cac2d761beb"
        },
        "date": 1786355169747,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 542.9585913182486,
            "unit": "iter/sec",
            "range": "stddev: 0.0008545123596385661",
            "extra": "mean: 1.841761077160785 msec\nrounds: 1620"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 433.28493638013515,
            "unit": "iter/sec",
            "range": "stddev: 0.0010677486395893822",
            "extra": "mean: 2.3079500717344743 msec\nrounds: 1868"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 370.3861046741541,
            "unit": "iter/sec",
            "range": "stddev: 0.0014237252334944105",
            "extra": "mean: 2.6998853017980964 msec\nrounds: 2336"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 252.07795619407727,
            "unit": "iter/sec",
            "range": "stddev: 0.0023469024753409854",
            "extra": "mean: 3.967026768616333 msec\nrounds: 3505"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 444.2660896857599,
            "unit": "iter/sec",
            "range": "stddev: 0.0010009272015698927",
            "extra": "mean: 2.2509032834518705 msec\nrounds: 1831"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 378.6210174899279,
            "unit": "iter/sec",
            "range": "stddev: 0.0003018156583396433",
            "extra": "mean: 2.641163469026391 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 140100.43516355823,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012226184204085653",
            "extra": "mean: 7.137736573284476 usec\nrounds: 46381"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.035478861113383,
            "unit": "iter/sec",
            "range": "stddev: 0.0002509672852749772",
            "extra": "mean: 35.66908933333934 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8159286615947042,
            "unit": "iter/sec",
            "range": "stddev: 0.0030745077491947347",
            "extra": "mean: 1.2255973433333434 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.035405052883650805,
            "unit": "iter/sec",
            "range": "stddev: 0.980463677673077",
            "extra": "mean: 28.244556032333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3248.7019362326937,
            "unit": "iter/sec",
            "range": "stddev: 0.00001954540249981155",
            "extra": "mean: 307.8152504072547 usec\nrounds: 2456"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 881.9696071093441,
            "unit": "iter/sec",
            "range": "stddev: 0.00001674239291251576",
            "extra": "mean: 1.1338259186475832 msec\nrounds: 799"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 361.1861733695094,
            "unit": "iter/sec",
            "range": "stddev: 0.00002564872538449749",
            "extra": "mean: 2.7686552634919273 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 502.3333263564947,
            "unit": "iter/sec",
            "range": "stddev: 0.00043577801172689284",
            "extra": "mean: 1.9907100475557984 msec\nrounds: 757"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10597.790179909885,
            "unit": "iter/sec",
            "range": "stddev: 0.000006185439666534435",
            "extra": "mean: 94.35929406261401 usec\nrounds: 7444"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10170.741612577962,
            "unit": "iter/sec",
            "range": "stddev: 0.000004799488127104209",
            "extra": "mean: 98.32124717073917 usec\nrounds: 7157"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1850.2997917677176,
            "unit": "iter/sec",
            "range": "stddev: 0.000009294779218247946",
            "extra": "mean: 540.452960352242 usec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1837.7683348399794,
            "unit": "iter/sec",
            "range": "stddev: 0.000016247658638742987",
            "extra": "mean: 544.1382251735627 usec\nrounds: 1732"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1023.9660645113306,
            "unit": "iter/sec",
            "range": "stddev: 0.000022082734437073698",
            "extra": "mean: 976.5948644765215 usec\nrounds: 974"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 376.9752853444182,
            "unit": "iter/sec",
            "range": "stddev: 0.00003788535910289384",
            "extra": "mean: 2.652693794200233 msec\nrounds: 1069"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.4665439340273,
            "unit": "iter/sec",
            "range": "stddev: 0.001315413294913433",
            "extra": "mean: 40.872139632652875 msec\nrounds: 637"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 466.1987130823125,
            "unit": "iter/sec",
            "range": "stddev: 0.000026990325544004345",
            "extra": "mean: 2.1450080661707855 msec\nrounds: 2690"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.87126899120693,
            "unit": "iter/sec",
            "range": "stddev: 0.00008245479855119101",
            "extra": "mean: 2.667582401529567 msec\nrounds: 1046"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 305.6923220306927,
            "unit": "iter/sec",
            "range": "stddev: 0.00004409548556335299",
            "extra": "mean: 3.2712630574332717 msec\nrounds: 592"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.13817694434553,
            "unit": "iter/sec",
            "range": "stddev: 0.00010266233925558123",
            "extra": "mean: 5.12457385663291 msec\nrounds: 279"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 854.6541439103335,
            "unit": "iter/sec",
            "range": "stddev: 0.000020810875773639595",
            "extra": "mean: 1.1700639458958917 msec\nrounds: 1109"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5231.133060361891,
            "unit": "iter/sec",
            "range": "stddev: 0.000008968141775724462",
            "extra": "mean: 191.16317410798567 usec\nrounds: 4681"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 78.08937338726847,
            "unit": "iter/sec",
            "range": "stddev: 0.00029129915803910374",
            "extra": "mean: 12.805839727266116 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1877.2112389375513,
            "unit": "iter/sec",
            "range": "stddev: 0.000020329681184784165",
            "extra": "mean: 532.7050995954893 usec\nrounds: 1486"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 27.67877411476042,
            "unit": "iter/sec",
            "range": "stddev: 0.004424139853760124",
            "extra": "mean: 36.1287676923063 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17896.76329711082,
            "unit": "iter/sec",
            "range": "stddev: 0.000006004100696860732",
            "extra": "mean: 55.8760253683098 usec\nrounds: 14112"
          }
        ]
      }
    ]
  }
}