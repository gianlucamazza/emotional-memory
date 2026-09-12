window.BENCHMARK_DATA = {
  "lastUpdate": 1789212140548,
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
          "id": "07aa596eba6b08ad917a90e9045b905430a43594",
          "message": "build(deps): bump zizmorcore/zizmor-action in the github-actions group (#132)\n\nBumps the github-actions group with 1 update: [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action).\n\n\nUpdates `zizmorcore/zizmor-action` from 0.6.2 to 0.6.3\n- [Release notes](https://github.com/zizmorcore/zizmor-action/releases)\n- [Commits](https://github.com/zizmorcore/zizmor-action/compare/3dc1ecc9bcb9e94e9b2c709687979e1298497054...70fb788f84895a7701f5643d103d587e460b5c99)\n\n---\nupdated-dependencies:\n- dependency-name: zizmorcore/zizmor-action\n  dependency-version: 0.6.3\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-09-12T13:17:48+02:00",
          "tree_id": "4df4dacce32c6edbbd0b6ad0308dcf8c2fb4b8a9",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/07aa596eba6b08ad917a90e9045b905430a43594"
        },
        "date": 1789212139119,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 548.4787277238757,
            "unit": "iter/sec",
            "range": "stddev: 0.0007665518390655979",
            "extra": "mean: 1.8232247659811458 msec\nrounds: 1846"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 521.2678479682079,
            "unit": "iter/sec",
            "range": "stddev: 0.0008539592730631967",
            "extra": "mean: 1.9183995404623344 msec\nrounds: 1730"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 473.2495421943804,
            "unit": "iter/sec",
            "range": "stddev: 0.0009227461006352082",
            "extra": "mean: 2.1130501159349553 msec\nrounds: 2096"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 367.3672599420049,
            "unit": "iter/sec",
            "range": "stddev: 0.0013565120966433386",
            "extra": "mean: 2.7220716406733323 msec\nrounds: 2911"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 512.5359734060548,
            "unit": "iter/sec",
            "range": "stddev: 0.001014498582597634",
            "extra": "mean: 1.9510825617848946 msec\nrounds: 1748"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 419.86700048790124,
            "unit": "iter/sec",
            "range": "stddev: 0.00025443858401868514",
            "extra": "mean: 2.3817065852709605 msec\nrounds: 516"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 206872.7051927232,
            "unit": "iter/sec",
            "range": "stddev: 5.723787865337131e-7",
            "extra": "mean: 4.833890479018956 usec\nrounds: 55341"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 16.70490651781559,
            "unit": "iter/sec",
            "range": "stddev: 0.0004469176525443133",
            "extra": "mean: 59.86265166665324 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7159437939170087,
            "unit": "iter/sec",
            "range": "stddev: 0.024955547881502452",
            "extra": "mean: 1.3967576903333263 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.045788326544903654,
            "unit": "iter/sec",
            "range": "stddev: 0.2962454944468555",
            "extra": "mean: 21.839627596333333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4972.343127612568,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060839014065279824",
            "extra": "mean: 201.11242815218628 usec\nrounds: 3410"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1265.2147612853726,
            "unit": "iter/sec",
            "range": "stddev: 0.000013067762010926376",
            "extra": "mean: 790.3796498422669 usec\nrounds: 951"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 485.2406108024079,
            "unit": "iter/sec",
            "range": "stddev: 0.000023671170591083154",
            "extra": "mean: 2.0608332809291685 msec\nrounds: 388"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 526.3223906053471,
            "unit": "iter/sec",
            "range": "stddev: 0.0007799287856458012",
            "extra": "mean: 1.899976170213574 msec\nrounds: 799"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 16044.456873539373,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025765184158696523",
            "extra": "mean: 62.32682152358843 usec\nrounds: 10119"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 15883.525188732085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025266742572233893",
            "extra": "mean: 62.958316124270006 usec\nrounds: 11951"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2768.154852099971,
            "unit": "iter/sec",
            "range": "stddev: 0.000011774905470783662",
            "extra": "mean: 361.25146656495116 usec\nrounds: 1301"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2825.5896201815162,
            "unit": "iter/sec",
            "range": "stddev: 0.000011180716635683262",
            "extra": "mean: 353.908434847577 usec\nrounds: 2026"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1674.0679529546067,
            "unit": "iter/sec",
            "range": "stddev: 0.000026515880760285606",
            "extra": "mean: 597.3473168965893 usec\nrounds: 1379"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 500.5156824319768,
            "unit": "iter/sec",
            "range": "stddev: 0.000027380549309499077",
            "extra": "mean: 1.997939395507165 msec\nrounds: 1469"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 39.707121276794155,
            "unit": "iter/sec",
            "range": "stddev: 0.0015571329537932502",
            "extra": "mean: 25.18439936829229 msec\nrounds: 410"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 595.2711002121154,
            "unit": "iter/sec",
            "range": "stddev: 0.00002698713771896919",
            "extra": "mean: 1.6799068519262335 msec\nrounds: 3012"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 496.7487352309058,
            "unit": "iter/sec",
            "range": "stddev: 0.0000397741910010623",
            "extra": "mean: 2.0130901783477433 msec\nrounds: 1441"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 412.06011843293004,
            "unit": "iter/sec",
            "range": "stddev: 0.00003504884593815693",
            "extra": "mean: 2.426830346511118 msec\nrounds: 860"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 269.73299413740665,
            "unit": "iter/sec",
            "range": "stddev: 0.00004969538123321282",
            "extra": "mean: 3.7073699611645683 msec\nrounds: 412"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1321.6065112003714,
            "unit": "iter/sec",
            "range": "stddev: 0.00001752415048721604",
            "extra": "mean: 756.6548677879417 usec\nrounds: 1664"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 7799.144631239895,
            "unit": "iter/sec",
            "range": "stddev: 0.000004826248306061488",
            "extra": "mean: 128.21918906266285 usec\nrounds: 5760"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 115.23895612893666,
            "unit": "iter/sec",
            "range": "stddev: 0.00013243008836387926",
            "extra": "mean: 8.677621123894394 msec\nrounds: 113"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 3155.761325033905,
            "unit": "iter/sec",
            "range": "stddev: 0.000007392739620961358",
            "extra": "mean: 316.8807450890654 usec\nrounds: 2240"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 42.348025397760175,
            "unit": "iter/sec",
            "range": "stddev: 0.00506724446325248",
            "extra": "mean: 23.61385189999652 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 33336.27681063491,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020116778974415912",
            "extra": "mean: 29.997351104337508 usec\nrounds: 22771"
          }
        ]
      }
    ]
  }
}