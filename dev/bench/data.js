window.BENCHMARK_DATA = {
  "lastUpdate": 1789210270571,
  "repoUrl": "https://github.com/gianlucamazza/emotional-memory",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "info@gianlucamazza.it",
            "name": "Gianluca",
            "username": "gianlucamazza"
          },
          "committer": {
            "email": "info@gianlucamazza.it",
            "name": "Gianluca",
            "username": "gianlucamazza"
          },
          "distinct": true,
          "id": "ec26966439739e539420628a362fea8aef306537",
          "message": "chore(release): v0.18.1\n\nPrereserved Zenodo DOI: 10.5281/zenodo.22724258\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-09-12T12:43:15+02:00",
          "tree_id": "db4832e22af634702b9b549b756829881878967a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/ec26966439739e539420628a362fea8aef306537"
        },
        "date": 1789210268652,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 482.70711944736485,
            "unit": "iter/sec",
            "range": "stddev: 0.0008866993860741669",
            "extra": "mean: 2.071649577376995 msec\nrounds: 1441"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 464.6191226909174,
            "unit": "iter/sec",
            "range": "stddev: 0.001160241109117793",
            "extra": "mean: 2.15230056440281 msec\nrounds: 1281"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 437.9434796283467,
            "unit": "iter/sec",
            "range": "stddev: 0.0009436406467709187",
            "extra": "mean: 2.283399677165266 msec\nrounds: 1524"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 315.7248362094617,
            "unit": "iter/sec",
            "range": "stddev: 0.0023172461250187287",
            "extra": "mean: 3.1673149695976677 msec\nrounds: 2138"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 439.46908131306634,
            "unit": "iter/sec",
            "range": "stddev: 0.0010495001897162025",
            "extra": "mean: 2.275472934323737 msec\nrounds: 1279"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 313.2904170761451,
            "unit": "iter/sec",
            "range": "stddev: 0.00038067393443676987",
            "extra": "mean: 3.19192654959807 msec\nrounds: 373"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135190.1670654714,
            "unit": "iter/sec",
            "range": "stddev: 8.390675178774706e-7",
            "extra": "mean: 7.396987678221514 usec\nrounds: 37738"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 12.303145368011018,
            "unit": "iter/sec",
            "range": "stddev: 0.00020007418285797427",
            "extra": "mean: 81.28002799999952 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5553758765466212,
            "unit": "iter/sec",
            "range": "stddev: 0.009637110090518914",
            "extra": "mean: 1.8005823483333359 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.024136582253413513,
            "unit": "iter/sec",
            "range": "stddev: 2.292883061680689",
            "extra": "mean: 41.43088650666667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3456.8506482516586,
            "unit": "iter/sec",
            "range": "stddev: 0.00001404052174824788",
            "extra": "mean: 289.2806492828267 usec\nrounds: 2301"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 887.1647817440379,
            "unit": "iter/sec",
            "range": "stddev: 0.000016969332997371847",
            "extra": "mean: 1.127186313724204 msec\nrounds: 663"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.70373718800323,
            "unit": "iter/sec",
            "range": "stddev: 0.00003061263452977429",
            "extra": "mean: 2.811328348432452 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 430.0658014434624,
            "unit": "iter/sec",
            "range": "stddev: 0.0004535830347337607",
            "extra": "mean: 2.3252255739554837 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12091.721018209597,
            "unit": "iter/sec",
            "range": "stddev: 0.000003337890107093138",
            "extra": "mean: 82.70121337517168 usec\nrounds: 6594"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12084.134249572444,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034413503610131595",
            "extra": "mean: 82.75313558647213 usec\nrounds: 7250"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1911.633978288937,
            "unit": "iter/sec",
            "range": "stddev: 0.000012142404160535236",
            "extra": "mean: 523.1126938301645 usec\nrounds: 859"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1929.0088550544913,
            "unit": "iter/sec",
            "range": "stddev: 0.000012631705640287047",
            "extra": "mean: 518.4009380671048 usec\nrounds: 1324"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1140.3785218976677,
            "unit": "iter/sec",
            "range": "stddev: 0.00003237964941041353",
            "extra": "mean: 876.9018188240967 usec\nrounds: 988"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 356.5868115499568,
            "unit": "iter/sec",
            "range": "stddev: 0.00023285267846143162",
            "extra": "mean: 2.8043661952985683 msec\nrounds: 1106"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.6916103345515,
            "unit": "iter/sec",
            "range": "stddev: 0.003140540901537334",
            "extra": "mean: 48.328766288922836 msec\nrounds: 668"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 431.5414370308621,
            "unit": "iter/sec",
            "range": "stddev: 0.00018248994735288788",
            "extra": "mean: 2.3172745747901007 msec\nrounds: 2507"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 360.9798024803554,
            "unit": "iter/sec",
            "range": "stddev: 0.00010721994014121566",
            "extra": "mean: 2.7702380940119777 msec\nrounds: 1085"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 295.98749466076356,
            "unit": "iter/sec",
            "range": "stddev: 0.00010822823336768598",
            "extra": "mean: 3.3785211133535133 msec\nrounds: 644"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.76219802567843,
            "unit": "iter/sec",
            "range": "stddev: 0.00013519510845426253",
            "extra": "mean: 5.082277033058429 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 930.6914612057966,
            "unit": "iter/sec",
            "range": "stddev: 0.000019059051368016992",
            "extra": "mean: 1.074469941632867 msec\nrounds: 1165"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5301.208755298305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067303321407113026",
            "extra": "mean: 188.636223578358 usec\nrounds: 4750"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 75.86553874884883,
            "unit": "iter/sec",
            "range": "stddev: 0.0002154835402334286",
            "extra": "mean: 13.181215298694148 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2242.901764852636,
            "unit": "iter/sec",
            "range": "stddev: 0.000011560781132286618",
            "extra": "mean: 445.851002335674 usec\nrounds: 1714"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.232761501328667,
            "unit": "iter/sec",
            "range": "stddev: 0.006138459832030188",
            "extra": "mean: 35.419843714294075 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 21289.46215450636,
            "unit": "iter/sec",
            "range": "stddev: 0.000009176593731217689",
            "extra": "mean: 46.971595277635004 usec\nrounds: 15460"
          }
        ]
      }
    ]
  }
}