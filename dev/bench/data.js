window.BENCHMARK_DATA = {
  "lastUpdate": 1780178914736,
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
          "id": "d9ee4974db201cca623b64d023c1a85f3889e282",
          "message": "ci(release): create GitHub release automatically on tag (#43) (#44)\n\nAfter Publish/Verify PyPI, add an on-tag step that extracts the matching\nCHANGELOG.md section as release notes and runs 'gh release create' (idempotent —\nuploads assets via --clobber if the release already exists), attaching wheel +\nsdist. Raises the job 'contents' permission to write. Previously the GitHub\nrelease was created by hand after every tag; now the on-tag workflow does\nPyPI + GitHub release end-to-end.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-31T00:02:37+02:00",
          "tree_id": "18c2b7c177ced04961de45d319997cb4d72839d6",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/d9ee4974db201cca623b64d023c1a85f3889e282"
        },
        "date": 1780178914025,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 526.2731646888911,
            "unit": "iter/sec",
            "range": "stddev: 0.0009066834207004722",
            "extra": "mean: 1.9001538879360393 msec\nrounds: 1749"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 392.0527588238349,
            "unit": "iter/sec",
            "range": "stddev: 0.0013375711956575627",
            "extra": "mean: 2.550677115498479 msec\nrounds: 2026"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 341.29295425533985,
            "unit": "iter/sec",
            "range": "stddev: 0.0017232463078798671",
            "extra": "mean: 2.9300341174105977 msec\nrounds: 2487"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 214.68041996755812,
            "unit": "iter/sec",
            "range": "stddev: 0.0030000740108426837",
            "extra": "mean: 4.658086658071179 msec\nrounds: 3878"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 421.6418106945969,
            "unit": "iter/sec",
            "range": "stddev: 0.0011257672495249762",
            "extra": "mean: 2.371681305401468 msec\nrounds: 1981"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 383.43965848321614,
            "unit": "iter/sec",
            "range": "stddev: 0.00024296093745191895",
            "extra": "mean: 2.6079722790170696 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 149010.86003630995,
            "unit": "iter/sec",
            "range": "stddev: 5.840453165207996e-7",
            "extra": "mean: 6.710920262834042 usec\nrounds: 43982"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.55969121324958,
            "unit": "iter/sec",
            "range": "stddev: 0.00020501024472339486",
            "extra": "mean: 29.79765200000391 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8294506866500787,
            "unit": "iter/sec",
            "range": "stddev: 0.003330037792509613",
            "extra": "mean: 1.2056171826666666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.032767193968640446,
            "unit": "iter/sec",
            "range": "stddev: 1.108078668280436",
            "extra": "mean: 30.51832881866666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3656.5897385528997,
            "unit": "iter/sec",
            "range": "stddev: 0.000008825894547331938",
            "extra": "mean: 273.4788618631718 usec\nrounds: 2512"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 905.9268126318096,
            "unit": "iter/sec",
            "range": "stddev: 0.00004051277367982996",
            "extra": "mean: 1.1038419285713579 msec\nrounds: 700"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 366.76693325949924,
            "unit": "iter/sec",
            "range": "stddev: 0.000054971319115891135",
            "extra": "mean: 2.7265271465802186 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 495.44104084776365,
            "unit": "iter/sec",
            "range": "stddev: 0.0004319590851601223",
            "extra": "mean: 2.0184036394903235 msec\nrounds: 785"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13378.825760036285,
            "unit": "iter/sec",
            "range": "stddev: 0.000002176769021884594",
            "extra": "mean: 74.74497522698046 usec\nrounds: 8154"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13300.610393187093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025279089767315365",
            "extra": "mean: 75.18451938959322 usec\nrounds: 8123"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2165.3427530148642,
            "unit": "iter/sec",
            "range": "stddev: 0.000010431888815464483",
            "extra": "mean: 461.8206510759895 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2174.3595984126646,
            "unit": "iter/sec",
            "range": "stddev: 0.000007943087576966548",
            "extra": "mean: 459.9055283817931 usec\nrounds: 1656"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1183.6961013347884,
            "unit": "iter/sec",
            "range": "stddev: 0.000013758611663805634",
            "extra": "mean: 844.8114333335689 usec\nrounds: 900"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 388.3300555711031,
            "unit": "iter/sec",
            "range": "stddev: 0.000041456015944130094",
            "extra": "mean: 2.5751290317442357 msec\nrounds: 378"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.967658422435147,
            "unit": "iter/sec",
            "range": "stddev: 0.0006659769039575453",
            "extra": "mean: 45.52146527272653 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 469.42734346791923,
            "unit": "iter/sec",
            "range": "stddev: 0.00004177429190232173",
            "extra": "mean: 2.1302551159726812 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 391.6789972566166,
            "unit": "iter/sec",
            "range": "stddev: 0.00005421302759674037",
            "extra": "mean: 2.5531111114054177 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 322.86324982625797,
            "unit": "iter/sec",
            "range": "stddev: 0.00005814069496883171",
            "extra": "mean: 3.0972865463570995 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 212.5463367765874,
            "unit": "iter/sec",
            "range": "stddev: 0.00012124250951744099",
            "extra": "mean: 4.70485643349913 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 964.1861098919127,
            "unit": "iter/sec",
            "range": "stddev: 0.000013152243126034742",
            "extra": "mean: 1.0371441672314716 msec\nrounds: 885"
          }
        ]
      }
    ]
  }
}