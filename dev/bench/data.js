window.BENCHMARK_DATA = {
  "lastUpdate": 1783415075267,
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
          "id": "699715f326c4af9e36cc962c2c43da615cef870f",
          "message": "chore(release): v0.15.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.21235738\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T10:56:02+02:00",
          "tree_id": "de306be6dc9a75cb2bfd97cd8a7bf3d688cb424d",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/699715f326c4af9e36cc962c2c43da615cef870f"
        },
        "date": 1783415073537,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 533.0583513413169,
            "unit": "iter/sec",
            "range": "stddev: 0.0009690221247026949",
            "extra": "mean: 1.8759672322621594 msec\nrounds: 1494"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 385.9412969062519,
            "unit": "iter/sec",
            "range": "stddev: 0.0015646922959896528",
            "extra": "mean: 2.5910676261289227 msec\nrounds: 1661"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 315.96651197269745,
            "unit": "iter/sec",
            "range": "stddev: 0.002059683322842547",
            "extra": "mean: 3.164892360765148 msec\nrounds: 2248"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 180.05482459709842,
            "unit": "iter/sec",
            "range": "stddev: 0.003829237226338982",
            "extra": "mean: 5.553863953591139 msec\nrounds: 3620"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 454.6878216078898,
            "unit": "iter/sec",
            "range": "stddev: 0.0009842576197093102",
            "extra": "mean: 2.199311159167954 msec\nrounds: 1778"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 379.50267722254733,
            "unit": "iter/sec",
            "range": "stddev: 0.0003137674006258976",
            "extra": "mean: 2.635027524228984 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134954.0110233998,
            "unit": "iter/sec",
            "range": "stddev: 0.000001231038405654962",
            "extra": "mean: 7.409931667956198 usec\nrounds: 43435"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.791589683186572,
            "unit": "iter/sec",
            "range": "stddev: 0.00015483664906050995",
            "extra": "mean: 35.98210866667273 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8143399369242771,
            "unit": "iter/sec",
            "range": "stddev: 0.004245287648245689",
            "extra": "mean: 1.2279884046666705 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02599216081512151,
            "unit": "iter/sec",
            "range": "stddev: 3.2104390687792534",
            "extra": "mean: 38.47313838633332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3145.114737251648,
            "unit": "iter/sec",
            "range": "stddev: 0.00001425779979116398",
            "extra": "mean: 317.9534241328976 usec\nrounds: 2221"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 851.6059164449355,
            "unit": "iter/sec",
            "range": "stddev: 0.00002377948088384718",
            "extra": "mean: 1.1742520580112241 msec\nrounds: 724"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 345.54572058985855,
            "unit": "iter/sec",
            "range": "stddev: 0.0003548990921556363",
            "extra": "mean: 2.8939730415210043 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 493.1936036276653,
            "unit": "iter/sec",
            "range": "stddev: 0.000971822172553436",
            "extra": "mean: 2.0276013164902 msec\nrounds: 752"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10374.344231824252,
            "unit": "iter/sec",
            "range": "stddev: 0.000005367551615428665",
            "extra": "mean: 96.39163475339564 usec\nrounds: 6549"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10582.430312416438,
            "unit": "iter/sec",
            "range": "stddev: 0.000005601612040356475",
            "extra": "mean: 94.49625185121164 usec\nrounds: 7834"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1832.0453982166487,
            "unit": "iter/sec",
            "range": "stddev: 0.00001999271413640715",
            "extra": "mean: 545.8380021441723 usec\nrounds: 932"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1809.725795425431,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152069658861316",
            "extra": "mean: 552.5698990022516 usec\nrounds: 1604"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1005.5160801721031,
            "unit": "iter/sec",
            "range": "stddev: 0.000023191673319134052",
            "extra": "mean: 994.5141800505478 usec\nrounds: 772"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 374.88103462734597,
            "unit": "iter/sec",
            "range": "stddev: 0.0000660887746747505",
            "extra": "mean: 2.6675129111134135 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.199936466354657,
            "unit": "iter/sec",
            "range": "stddev: 0.0005091099330108722",
            "extra": "mean: 49.505106199993065 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 418.8995374084966,
            "unit": "iter/sec",
            "range": "stddev: 0.00025565706941092775",
            "extra": "mean: 2.387207219627063 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 368.8553652641819,
            "unit": "iter/sec",
            "range": "stddev: 0.0001564896192940535",
            "extra": "mean: 2.7110897499993776 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 251.32795689513577,
            "unit": "iter/sec",
            "range": "stddev: 0.0005717551920844336",
            "extra": "mean: 3.978864955390699 msec\nrounds: 269"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 154.13536962417254,
            "unit": "iter/sec",
            "range": "stddev: 0.001126969118978777",
            "extra": "mean: 6.487803561494644 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 820.8577568878773,
            "unit": "iter/sec",
            "range": "stddev: 0.00006916967842580563",
            "extra": "mean: 1.2182378635140219 msec\nrounds: 740"
          }
        ]
      }
    ]
  }
}