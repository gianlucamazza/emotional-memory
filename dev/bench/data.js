window.BENCHMARK_DATA = {
  "lastUpdate": 1784238484825,
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
          "id": "7daa6eac55823f45f462259254d727bea8bf44d5",
          "message": "perf: close H13 dual-path encode (sim + live Ollama PASS)\n\nComplete H13 with live measurement on local Ollama llama3.2:1b (DIRECT_VAD,\nfallback_count=0): dual hot path ~0x sync, combined ~0.93x. Add unit test in\nmake check, dotenv load, --h13-only, and make bench-perf-h13-ollama. Document\nmeasured tables and mark hypothesis CLOSED PASS.",
          "timestamp": "2026-07-16T23:41:50+02:00",
          "tree_id": "36305c9ac897e3dc79b8e30d76e0d011917a0e0c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7daa6eac55823f45f462259254d727bea8bf44d5"
        },
        "date": 1784238482178,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 672.3312030570715,
            "unit": "iter/sec",
            "range": "stddev: 0.0007213196028444449",
            "extra": "mean: 1.4873621742573115 msec\nrounds: 1515"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 438.0724658508414,
            "unit": "iter/sec",
            "range": "stddev: 0.0011352796017363954",
            "extra": "mean: 2.282727352100893 msec\nrounds: 2238"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 346.6176019786268,
            "unit": "iter/sec",
            "range": "stddev: 0.0015922579683796763",
            "extra": "mean: 2.8850237099662994 msec\nrounds: 3010"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 216.14134928444238,
            "unit": "iter/sec",
            "range": "stddev: 0.002611201567800487",
            "extra": "mean: 4.62660200517208 msec\nrounds: 5027"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 405.8559452924154,
            "unit": "iter/sec",
            "range": "stddev: 0.0011955040671223906",
            "extra": "mean: 2.46392842484914 msec\nrounds: 2495"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 419.9494077330786,
            "unit": "iter/sec",
            "range": "stddev: 0.0009876253762622507",
            "extra": "mean: 2.381239219738592 msec\nrounds: 537"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 177959.1114856693,
            "unit": "iter/sec",
            "range": "stddev: 8.971941839257739e-7",
            "extra": "mean: 5.619268334459672 usec\nrounds: 46647"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 42.82215959977407,
            "unit": "iter/sec",
            "range": "stddev: 0.0007702562601811752",
            "extra": "mean: 23.352395333309534 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.9755599793789809,
            "unit": "iter/sec",
            "range": "stddev: 0.004815748627021131",
            "extra": "mean: 1.0250522993333295 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.045263341127412184,
            "unit": "iter/sec",
            "range": "stddev: 0.13854399349325053",
            "extra": "mean: 22.092933819999967 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4285.343982489282,
            "unit": "iter/sec",
            "range": "stddev: 0.000013891867941011832",
            "extra": "mean: 233.35349602883392 usec\nrounds: 3022"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1130.3561229962552,
            "unit": "iter/sec",
            "range": "stddev: 0.00002470100400978468",
            "extra": "mean: 884.6769435364159 usec\nrounds: 921"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 442.0685782447219,
            "unit": "iter/sec",
            "range": "stddev: 0.00005963805810117386",
            "extra": "mean: 2.2620924653152263 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 550.313535860777,
            "unit": "iter/sec",
            "range": "stddev: 0.0004362892105215838",
            "extra": "mean: 1.817145926523218 msec\nrounds: 871"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13557.341286953093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038155178069182105",
            "extra": "mean: 73.76077498044178 usec\nrounds: 8737"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13748.796806995886,
            "unit": "iter/sec",
            "range": "stddev: 0.000003837498861097181",
            "extra": "mean: 72.73363728025741 usec\nrounds: 8825"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2365.863058206394,
            "unit": "iter/sec",
            "range": "stddev: 0.000023778430312853745",
            "extra": "mean: 422.6787330447263 usec\nrounds: 1150"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2381.7100979907677,
            "unit": "iter/sec",
            "range": "stddev: 0.00004669168490646224",
            "extra": "mean: 419.86638123741807 usec\nrounds: 1663"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1456.3498346946742,
            "unit": "iter/sec",
            "range": "stddev: 0.000023695114375632164",
            "extra": "mean: 686.6482051063311 usec\nrounds: 1214"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 444.40671258451727,
            "unit": "iter/sec",
            "range": "stddev: 0.00011827430959526514",
            "extra": "mean: 2.2501910337590143 msec\nrounds: 1333"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 27.956959547561297,
            "unit": "iter/sec",
            "range": "stddev: 0.0014894748976956715",
            "extra": "mean: 35.769268768256694 msec\nrounds: 712"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 545.5445598170178,
            "unit": "iter/sec",
            "range": "stddev: 0.00006272111099293191",
            "extra": "mean: 1.8330308349796616 msec\nrounds: 3139"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 454.1022943252249,
            "unit": "iter/sec",
            "range": "stddev: 0.0000618290428880131",
            "extra": "mean: 2.2021469886778573 msec\nrounds: 1325"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 376.8288084176368,
            "unit": "iter/sec",
            "range": "stddev: 0.00006293975193439232",
            "extra": "mean: 2.6537249214017278 msec\nrounds: 827"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 249.74458907763577,
            "unit": "iter/sec",
            "range": "stddev: 0.00007673212549723492",
            "extra": "mean: 4.00409075405089 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1186.7613276945294,
            "unit": "iter/sec",
            "range": "stddev: 0.00002615493664073382",
            "extra": "mean: 842.629412219437 usec\nrounds: 1424"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 6846.0682286389865,
            "unit": "iter/sec",
            "range": "stddev: 0.000005915125151045623",
            "extra": "mean: 146.06924246193236 usec\nrounds: 6368"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 99.79641758073707,
            "unit": "iter/sec",
            "range": "stddev: 0.0002859338506103686",
            "extra": "mean: 10.020399772276217 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2704.371731158675,
            "unit": "iter/sec",
            "range": "stddev: 0.000014249709511176911",
            "extra": "mean: 369.7716510191277 usec\nrounds: 2258"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 35.437377090896184,
            "unit": "iter/sec",
            "range": "stddev: 0.005953968078312933",
            "extra": "mean: 28.218792757574 msec\nrounds: 33"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 27876.258435234802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030861372166539695",
            "extra": "mean: 35.872819959798775 usec\nrounds: 20912"
          }
        ]
      }
    ]
  }
}