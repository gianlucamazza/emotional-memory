window.BENCHMARK_DATA = {
  "lastUpdate": 1789212889442,
  "repoUrl": "https://github.com/gianlucamazza/emotional-memory",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "65b425ad8bd004e5c6a23ceb436e93d33a97fc07",
          "message": "build(deps-dev): update mypy requirement from <2,>=1.10 to >=1.10,<3 in the dev-dependencies group (#131)\n\n* build(deps-dev): update mypy requirement in the dev-dependencies group\n\nUpdates the requirements on [mypy](https://github.com/python/mypy) to permit the latest version.\n\nUpdates `mypy` to 2.3.1\n- [Changelog](https://github.com/python/mypy/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/python/mypy/compare/v1.10.0...v2.3.1)\n\n---\nupdated-dependencies:\n- dependency-name: mypy\n  dependency-version: 2.3.1\n  dependency-type: direct:development\n  dependency-group: dev-dependencies\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\n\n* chore: update uv.lock after mypy constraint change\n\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>\n\n---------\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>",
          "timestamp": "2026-09-12T13:27:51+02:00",
          "tree_id": "e0b7047144318ceafdd8d6c07a038a8f99ea3658",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/65b425ad8bd004e5c6a23ceb436e93d33a97fc07"
        },
        "date": 1789212886481,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 500.53879955331945,
            "unit": "iter/sec",
            "range": "stddev: 0.0007840015014282398",
            "extra": "mean: 1.9978471217264264 msec\nrounds: 1413"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 488.6493391077013,
            "unit": "iter/sec",
            "range": "stddev: 0.0012738554978404331",
            "extra": "mean: 2.0464572853532377 msec\nrounds: 1188"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 486.9729083542086,
            "unit": "iter/sec",
            "range": "stddev: 0.0007599983351546762",
            "extra": "mean: 2.0535023259910625 msec\nrounds: 1362"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 402.7365123132805,
            "unit": "iter/sec",
            "range": "stddev: 0.0010872960753937006",
            "extra": "mean: 2.4830130108047426 msec\nrounds: 1851"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 488.7634495739963,
            "unit": "iter/sec",
            "range": "stddev: 0.0008752037396878059",
            "extra": "mean: 2.045979503728429 msec\nrounds: 1207"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 336.4593954016828,
            "unit": "iter/sec",
            "range": "stddev: 0.0002855647784727693",
            "extra": "mean: 2.9721268410595214 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136699.72526636402,
            "unit": "iter/sec",
            "range": "stddev: 0.00000116866693894671",
            "extra": "mean: 7.3153036558885995 usec\nrounds: 47758"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.779217913629545,
            "unit": "iter/sec",
            "range": "stddev: 0.00013427315669884256",
            "extra": "mean: 92.77110899999268 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5374769858311716,
            "unit": "iter/sec",
            "range": "stddev: 0.01420049046822496",
            "extra": "mean: 1.8605447793333287 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.029417074508307686,
            "unit": "iter/sec",
            "range": "stddev: 1.5067321773680449",
            "extra": "mean: 33.99386297633334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3195.6677473249206,
            "unit": "iter/sec",
            "range": "stddev: 0.000012338843233878359",
            "extra": "mean: 312.9236450932972 usec\nrounds: 2364"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 867.5001183466383,
            "unit": "iter/sec",
            "range": "stddev: 0.000020713061944010742",
            "extra": "mean: 1.1527375949018799 msec\nrounds: 706"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 358.12604062803655,
            "unit": "iter/sec",
            "range": "stddev: 0.000028406937592609236",
            "extra": "mean: 2.792313003115678 msec\nrounds: 321"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 448.0854499759947,
            "unit": "iter/sec",
            "range": "stddev: 0.0003991688158308819",
            "extra": "mean: 2.23171718709807 msec\nrounds: 620"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10365.590403335733,
            "unit": "iter/sec",
            "range": "stddev: 0.000005309892173165631",
            "extra": "mean: 96.47303830162839 usec\nrounds: 7232"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10212.989663039505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055232779434665985",
            "extra": "mean: 97.91452189743902 usec\nrounds: 7147"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1823.57515260178,
            "unit": "iter/sec",
            "range": "stddev: 0.000012112661588781731",
            "extra": "mean: 548.3733415501156 usec\nrounds: 1136"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1831.8907831993547,
            "unit": "iter/sec",
            "range": "stddev: 0.000016241540236391144",
            "extra": "mean: 545.8840718951176 usec\nrounds: 1683"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1015.8630865635145,
            "unit": "iter/sec",
            "range": "stddev: 0.00003422056005481753",
            "extra": "mean: 984.3846215367698 usec\nrounds: 938"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.73631598255906,
            "unit": "iter/sec",
            "range": "stddev: 0.00007732455018005255",
            "extra": "mean: 2.690078846229588 msec\nrounds: 1008"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.245196582693637,
            "unit": "iter/sec",
            "range": "stddev: 0.001380214883045667",
            "extra": "mean: 41.24528322916573 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 462.30892262032467,
            "unit": "iter/sec",
            "range": "stddev: 0.00006076412450122436",
            "extra": "mean: 2.163055807645008 msec\nrounds: 2407"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 371.48966830593656,
            "unit": "iter/sec",
            "range": "stddev: 0.00008093763456361535",
            "extra": "mean: 2.691864903161883 msec\nrounds: 1012"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.7329387479536,
            "unit": "iter/sec",
            "range": "stddev: 0.00004123837056538573",
            "extra": "mean: 3.303241477903962 msec\nrounds: 611"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.6250439945293,
            "unit": "iter/sec",
            "range": "stddev: 0.00010307099282447734",
            "extra": "mean: 5.111819936653741 msec\nrounds: 221"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 842.5460487771961,
            "unit": "iter/sec",
            "range": "stddev: 0.00012536168829988662",
            "extra": "mean: 1.186878748587475 msec\nrounds: 1062"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5117.954686997536,
            "unit": "iter/sec",
            "range": "stddev: 0.000009443311805018033",
            "extra": "mean: 195.39055367968746 usec\nrounds: 4443"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 76.84656924096312,
            "unit": "iter/sec",
            "range": "stddev: 0.00020010159825195644",
            "extra": "mean: 13.01294267105615 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1881.0973991418366,
            "unit": "iter/sec",
            "range": "stddev: 0.0000210426408521038",
            "extra": "mean: 531.6045838222963 usec\nrounds: 1360"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 27.98198001383366,
            "unit": "iter/sec",
            "range": "stddev: 0.004720046213636714",
            "extra": "mean: 35.73728519231386 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17138.17957339991,
            "unit": "iter/sec",
            "range": "stddev: 0.00000623409792546175",
            "extra": "mean: 58.34925440693219 usec\nrounds: 13160"
          }
        ]
      }
    ]
  }
}