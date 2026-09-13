window.BENCHMARK_DATA = {
  "lastUpdate": 1789299599519,
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
          "id": "34037bb07d5a90fb058ad1744080cc4d9a0c683e",
          "message": "feat: host-owned Phase 6 measure.jsonl on FlyAffectHost (#137)\n\n* feat: host-owned Phase 6 measure.jsonl on FlyAffectHost\n\nFlyAffectHost writes fly measure.jsonl when measure_path is set. This\npackage owns mood_dt; the fly only supplies affect. Hypothesis taus and\nPolicy/LaunchGate thresholds stay at library defaults.\n\n* test: host-owned measure.jsonl coverage and example smoke tests\n\n* docs: Phase 6 host-owned measure.jsonl pointers\n\n* docs: Phase 6 host-owned measure.jsonl pointers\n\n* docs: Phase 6 host-owned measure.jsonl pointers\n\n* docs: Phase 6 host-owned measure.jsonl pointers",
          "timestamp": "2026-09-13T13:33:23+02:00",
          "tree_id": "0f84cb941cf7bb48c877ce9d61621b93b6f51e1f",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/34037bb07d5a90fb058ad1744080cc4d9a0c683e"
        },
        "date": 1789299597911,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 495.42155214489696,
            "unit": "iter/sec",
            "range": "stddev: 0.0007916681599502648",
            "extra": "mean: 2.0184830386779944 msec\nrounds: 1422"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 494.3786557241161,
            "unit": "iter/sec",
            "range": "stddev: 0.0010163688064435234",
            "extra": "mean: 2.02274104761926 msec\nrounds: 1197"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 480.8162503228,
            "unit": "iter/sec",
            "range": "stddev: 0.0007793446807241902",
            "extra": "mean: 2.079796594496633 msec\nrounds: 1381"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 404.61604987914916,
            "unit": "iter/sec",
            "range": "stddev: 0.0010737934129611952",
            "extra": "mean: 2.4714788261579845 msec\nrounds: 1835"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 490.118627407197,
            "unit": "iter/sec",
            "range": "stddev: 0.0008826121285731759",
            "extra": "mean: 2.0403223711168743 msec\nrounds: 1191"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 311.54963636471297,
            "unit": "iter/sec",
            "range": "stddev: 0.0006254269008873888",
            "extra": "mean: 3.209761409669432 msec\nrounds: 393"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136870.73251292136,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014182138453446094",
            "extra": "mean: 7.306163864547115 usec\nrounds: 48833"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.837260991523786,
            "unit": "iter/sec",
            "range": "stddev: 0.001080120989931644",
            "extra": "mean: 92.27423800000167 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5417088732514367,
            "unit": "iter/sec",
            "range": "stddev: 0.01537847499383377",
            "extra": "mean: 1.846010005333336 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02836171894041999,
            "unit": "iter/sec",
            "range": "stddev: 1.1932867147171615",
            "extra": "mean: 35.25879380233332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3210.1371022546377,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127054209500741",
            "extra": "mean: 311.5131747169461 usec\nrounds: 2381"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 871.9311104966953,
            "unit": "iter/sec",
            "range": "stddev: 0.00001844951720738771",
            "extra": "mean: 1.1468795962909848 msec\nrounds: 701"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 356.7164281327622,
            "unit": "iter/sec",
            "range": "stddev: 0.00003611177974879169",
            "extra": "mean: 2.803347200000056 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 448.4104306729656,
            "unit": "iter/sec",
            "range": "stddev: 0.0004355373731457804",
            "extra": "mean: 2.2300997737702475 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9907.332673117146,
            "unit": "iter/sec",
            "range": "stddev: 0.000006757284457770641",
            "extra": "mean: 100.93534082220033 usec\nrounds: 6276"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10348.801509897114,
            "unit": "iter/sec",
            "range": "stddev: 0.000005593610515474737",
            "extra": "mean: 96.62954681695713 usec\nrounds: 8074"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1767.0239482988118,
            "unit": "iter/sec",
            "range": "stddev: 0.000010624402867724182",
            "extra": "mean: 565.9232864176754 usec\nrounds: 1016"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1789.902320996765,
            "unit": "iter/sec",
            "range": "stddev: 0.00001094909067799401",
            "extra": "mean: 558.6897051695635 usec\nrounds: 1567"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1019.2665769123938,
            "unit": "iter/sec",
            "range": "stddev: 0.000041850848028368875",
            "extra": "mean: 981.0976074867903 usec\nrounds: 935"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.566744602715,
            "unit": "iter/sec",
            "range": "stddev: 0.00009661201504255317",
            "extra": "mean: 2.6985691904763365 msec\nrounds: 1008"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.46939418412902,
            "unit": "iter/sec",
            "range": "stddev: 0.0013892815443855332",
            "extra": "mean: 44.50498272473842 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 445.45587878953216,
            "unit": "iter/sec",
            "range": "stddev: 0.0001251956135501531",
            "extra": "mean: 2.244891239772991 msec\nrounds: 2298"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 355.52125483073274,
            "unit": "iter/sec",
            "range": "stddev: 0.00012292034353772227",
            "extra": "mean: 2.812771350270211 msec\nrounds: 925"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 284.1976633433239,
            "unit": "iter/sec",
            "range": "stddev: 0.00022418833534083114",
            "extra": "mean: 3.5186777689722026 msec\nrounds: 593"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 167.00695530561012,
            "unit": "iter/sec",
            "range": "stddev: 0.0004950538473178738",
            "extra": "mean: 5.9877745700475495 msec\nrounds: 207"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 840.4750612654256,
            "unit": "iter/sec",
            "range": "stddev: 0.000022484099060473854",
            "extra": "mean: 1.189803298261334 msec\nrounds: 1036"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5077.686646205588,
            "unit": "iter/sec",
            "range": "stddev: 0.000008206206887034379",
            "extra": "mean: 196.94007718008194 usec\nrounds: 4496"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 70.20872720851992,
            "unit": "iter/sec",
            "range": "stddev: 0.00028930108490398917",
            "extra": "mean: 14.243243536234463 msec\nrounds: 69"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1840.6429282108127,
            "unit": "iter/sec",
            "range": "stddev: 0.000017575197329807056",
            "extra": "mean: 543.288426382647 usec\nrounds: 1501"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 24.784803047723802,
            "unit": "iter/sec",
            "range": "stddev: 0.006385509154202231",
            "extra": "mean: 40.3473046799877 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 16378.866849115959,
            "unit": "iter/sec",
            "range": "stddev: 0.000005928378298261196",
            "extra": "mean: 61.054284720189564 usec\nrounds: 12500"
          }
        ]
      }
    ]
  }
}