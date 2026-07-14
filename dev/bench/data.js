window.BENCHMARK_DATA = {
  "lastUpdate": 1784043423387,
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
          "id": "2e4c70c9376f537cd86bc5497c56e84ba70f04fc",
          "message": "fix(security): pin chromadb <1.0 to resolve CVE-2026-45829\n\nPyPI 1.5.9 remains unpatched (fix merged in chroma-core/chroma PR #7237 but\nnot released). The optional [chroma] extra now requires chromadb>=0.6.3,<1.0,\noutside the vulnerable >=1.0.0,<=1.5.9 range. ChromaStore tests pass on 0.6.3;\nadd pydantic deprecation filter for chromadb 0.6.x and document remote-server\ntrust model in ChromaStore.",
          "timestamp": "2026-07-14T17:29:59+02:00",
          "tree_id": "ad871def5822827ddec1375e8152a92d4454d82b",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/2e4c70c9376f537cd86bc5497c56e84ba70f04fc"
        },
        "date": 1784043422370,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 595.9005773541759,
            "unit": "iter/sec",
            "range": "stddev: 0.0008117191990221796",
            "extra": "mean: 1.6781322891815995 msec\nrounds: 1442"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 416.41570549908806,
            "unit": "iter/sec",
            "range": "stddev: 0.001278639966001283",
            "extra": "mean: 2.4014464075063326 msec\nrounds: 1865"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 297.1404469268702,
            "unit": "iter/sec",
            "range": "stddev: 0.002070540552239477",
            "extra": "mean: 3.3654119132630633 msec\nrounds: 2571"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 180.36894216684934,
            "unit": "iter/sec",
            "range": "stddev: 0.0036608815928498143",
            "extra": "mean: 5.544191743803405 msec\nrounds: 4196"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 424.72647826730645,
            "unit": "iter/sec",
            "range": "stddev: 0.001091565402547499",
            "extra": "mean: 2.354456458847472 msec\nrounds: 1944"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.68829435131033,
            "unit": "iter/sec",
            "range": "stddev: 0.00025081792375055247",
            "extra": "mean: 2.626820984091438 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135264.28818408665,
            "unit": "iter/sec",
            "range": "stddev: 8.531265070167593e-7",
            "extra": "mean: 7.3929343319284655 usec\nrounds: 45060"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.838150803172766,
            "unit": "iter/sec",
            "range": "stddev: 0.0005722657647462161",
            "extra": "mean: 30.452384666659782 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8397133585716531,
            "unit": "iter/sec",
            "range": "stddev: 0.003674477504016418",
            "extra": "mean: 1.190882566999998 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03222966468724813,
            "unit": "iter/sec",
            "range": "stddev: 2.3555852539694833",
            "extra": "mean: 31.027316284666664 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3470.4075292277535,
            "unit": "iter/sec",
            "range": "stddev: 0.000010733283038845783",
            "extra": "mean: 288.1505966022738 usec\nrounds: 2531"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 894.1532492644106,
            "unit": "iter/sec",
            "range": "stddev: 0.000026708636493658275",
            "extra": "mean: 1.11837652082869 msec\nrounds: 48"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.4843643836976,
            "unit": "iter/sec",
            "range": "stddev: 0.000043189497995211796",
            "extra": "mean: 2.7740454199994247 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 496.4775449212924,
            "unit": "iter/sec",
            "range": "stddev: 0.0004540253958829563",
            "extra": "mean: 2.014189786082938 msec\nrounds: 776"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12444.76545429282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035973910316898277",
            "extra": "mean: 80.35507006321684 usec\nrounds: 7479"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12291.816432628491,
            "unit": "iter/sec",
            "range": "stddev: 0.000004090525378797022",
            "extra": "mean: 81.35494094636095 usec\nrounds: 8196"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2033.817749938366,
            "unit": "iter/sec",
            "range": "stddev: 0.000015611009787615542",
            "extra": "mean: 491.6861405257696 usec\nrounds: 1103"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2033.4970476844737,
            "unit": "iter/sec",
            "range": "stddev: 0.000023565778403498517",
            "extra": "mean: 491.7636842102582 usec\nrounds: 1748"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1150.810701364839,
            "unit": "iter/sec",
            "range": "stddev: 0.00008394747241661668",
            "extra": "mean: 868.9526425275849 usec\nrounds: 870"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 383.79056762662395,
            "unit": "iter/sec",
            "range": "stddev: 0.00008132483660119027",
            "extra": "mean: 2.6055877459001655 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.02437214694483,
            "unit": "iter/sec",
            "range": "stddev: 0.0008801298138630458",
            "extra": "mean: 47.5638460454723 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 465.2376665805967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000622421966095206",
            "extra": "mean: 2.1494390326342208 msec\nrounds: 429"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 384.5520562291681,
            "unit": "iter/sec",
            "range": "stddev: 0.0000657149247732525",
            "extra": "mean: 2.6004281703907073 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 317.7287636968658,
            "unit": "iter/sec",
            "range": "stddev: 0.00007729925228420831",
            "extra": "mean: 3.14733859271887 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.33443018647029,
            "unit": "iter/sec",
            "range": "stddev: 0.00009689863734319944",
            "extra": "mean: 4.799974728636776 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 952.3019943823498,
            "unit": "iter/sec",
            "range": "stddev: 0.000019705995603019994",
            "extra": "mean: 1.0500870584111153 msec\nrounds: 856"
          }
        ]
      }
    ]
  }
}