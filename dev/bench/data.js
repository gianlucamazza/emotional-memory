window.BENCHMARK_DATA = {
  "lastUpdate": 1782243819900,
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
          "id": "05f79ba4b17910b2eb0e37ae8c95b2832950cbc9",
          "message": "build(deps): bump the github-actions group with 2 updates (#58)\n\nBumps the github-actions group with 2 updates: [actions/checkout](https://github.com/actions/checkout) and [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action).\n\n\nUpdates `actions/checkout` from 6.0.3 to 7.0.0\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/actions/checkout/compare/df4cb1c069e1874edd31b4311f1884172cec0e10...9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0)\n\nUpdates `zizmorcore/zizmor-action` from 0.5.6 to 0.5.7\n- [Release notes](https://github.com/zizmorcore/zizmor-action/releases)\n- [Commits](https://github.com/zizmorcore/zizmor-action/compare/5f14fd08f7cf1cb1609c1e344975f152c7ee938d...192e21d79ab29983730a13d1382995c2307fbcaa)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: 7.0.0\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: github-actions\n- dependency-name: zizmorcore/zizmor-action\n  dependency-version: 0.5.7\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-23T21:36:11+02:00",
          "tree_id": "f9fd3cc6cf718656823f1e938953d7178b1c3cdf",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/05f79ba4b17910b2eb0e37ae8c95b2832950cbc9"
        },
        "date": 1782243819060,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 633.6617911332227,
            "unit": "iter/sec",
            "range": "stddev: 0.0007548283353508088",
            "extra": "mean: 1.578128923020005 msec\nrounds: 1351"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 378.71260968651416,
            "unit": "iter/sec",
            "range": "stddev: 0.001512013257604589",
            "extra": "mean: 2.640524699792191 msec\nrounds: 1922"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 267.5079723919948,
            "unit": "iter/sec",
            "range": "stddev: 0.0030804516866366157",
            "extra": "mean: 3.7382063459949615 msec\nrounds: 2422"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 139.69027884702854,
            "unit": "iter/sec",
            "range": "stddev: 0.005504081678060674",
            "extra": "mean: 7.158694278898792 msec\nrounds: 3668"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 646.6655349858042,
            "unit": "iter/sec",
            "range": "stddev: 0.0006155690533298847",
            "extra": "mean: 1.5463944588015692 msec\nrounds: 1068"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 379.7445156684693,
            "unit": "iter/sec",
            "range": "stddev: 0.00026690754293733825",
            "extra": "mean: 2.633349419779471 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135673.23676466913,
            "unit": "iter/sec",
            "range": "stddev: 9.202224865639728e-7",
            "extra": "mean: 7.370650423373783 usec\nrounds: 45824"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.864576746158484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003102441577242606",
            "extra": "mean: 30.42789833332904 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.809060077181867,
            "unit": "iter/sec",
            "range": "stddev: 0.05508632203679371",
            "extra": "mean: 1.2360021563333323 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.023597152240119965,
            "unit": "iter/sec",
            "range": "stddev: 4.2594933066894365",
            "extra": "mean: 42.377995015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3455.9218787381633,
            "unit": "iter/sec",
            "range": "stddev: 0.000010380488260941051",
            "extra": "mean: 289.35839266283506 usec\nrounds: 2208"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 896.3637880262669,
            "unit": "iter/sec",
            "range": "stddev: 0.00001701130188230469",
            "extra": "mean: 1.1156184724975706 msec\nrounds: 709"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 359.0376919718454,
            "unit": "iter/sec",
            "range": "stddev: 0.00003824673304322929",
            "extra": "mean: 2.7852228954235168 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 503.0167622843277,
            "unit": "iter/sec",
            "range": "stddev: 0.00046316156976407605",
            "extra": "mean: 1.9880053210528104 msec\nrounds: 760"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 11402.549713573328,
            "unit": "iter/sec",
            "range": "stddev: 0.000005987692851529826",
            "extra": "mean: 87.69968341463344 usec\nrounds: 6548"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12199.728473743005,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037274111430095384",
            "extra": "mean: 81.96903743819058 usec\nrounds: 8494"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1942.9401744523493,
            "unit": "iter/sec",
            "range": "stddev: 0.00002073580906119164",
            "extra": "mean: 514.6838863846474 usec\nrounds: 1065"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1806.2926003141936,
            "unit": "iter/sec",
            "range": "stddev: 0.00009635161919424106",
            "extra": "mean: 553.6201608898006 usec\nrounds: 1125"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1111.3026308996793,
            "unit": "iter/sec",
            "range": "stddev: 0.00014105074584279826",
            "extra": "mean: 899.8448957062473 usec\nrounds: 489"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 388.2073005570628,
            "unit": "iter/sec",
            "range": "stddev: 0.000056111113097012765",
            "extra": "mean: 2.5759433131861194 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.005732175753224,
            "unit": "iter/sec",
            "range": "stddev: 0.0018540415394261965",
            "extra": "mean: 49.98567366666997 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 455.8394578551582,
            "unit": "iter/sec",
            "range": "stddev: 0.00013896303274587478",
            "extra": "mean: 2.1937548028537437 msec\nrounds: 421"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 382.54347355231454,
            "unit": "iter/sec",
            "range": "stddev: 0.00007224019613647418",
            "extra": "mean: 2.6140819779617686 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.2134182258375,
            "unit": "iter/sec",
            "range": "stddev: 0.0002537194946340755",
            "extra": "mean: 3.26569621211858 msec\nrounds: 297"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 204.52774647776388,
            "unit": "iter/sec",
            "range": "stddev: 0.0002071032837338788",
            "extra": "mean: 4.889312170213146 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 955.5087372842247,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184448047978575",
            "extra": "mean: 1.0465629051621543 msec\nrounds: 833"
          }
        ]
      }
    ]
  }
}