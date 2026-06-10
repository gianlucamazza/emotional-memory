window.BENCHMARK_DATA = {
  "lastUpdate": 1781129600452,
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
          "id": "6d2ad4f1729c7a8fbf443a604d7bd9a8fa9ceb0c",
          "message": "build(deps): bump the github-actions group across 1 directory with 3 updates (#51)\n\nBumps the github-actions group with 3 updates in the / directory: [actions/checkout](https://github.com/actions/checkout), [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) and [codecov/codecov-action](https://github.com/codecov/codecov-action).\n\n\nUpdates `actions/checkout` from 6.0.2 to 6.0.3\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/actions/checkout/compare/de0fac2e4500dabe0009e67214ff5f5447ce83dd...df4cb1c069e1874edd31b4311f1884172cec0e10)\n\nUpdates `astral-sh/setup-uv` from 7.6.0 to 8.2.0\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/37802adc94f370d6bfd71619e3f0bf239e1f3b78...fac544c07dec837d0ccb6301d7b5580bf5edae39)\n\nUpdates `codecov/codecov-action` from 6.0.1 to 7.0.0\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/e79a6962e0d4c0c17b229090214935d2e33f8354...fb8b3582c8e4def4969c97caa2f19720cb33a72f)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: 6.0.3\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 8.2.0\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: github-actions\n- dependency-name: codecov/codecov-action\n  dependency-version: 7.0.0\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-11T00:07:40+02:00",
          "tree_id": "be71b168f9b2dda2f7dfab2db7e8e69eb7186efc",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/6d2ad4f1729c7a8fbf443a604d7bd9a8fa9ceb0c"
        },
        "date": 1781129599968,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 603.0883908528119,
            "unit": "iter/sec",
            "range": "stddev: 0.0007707280805333244",
            "extra": "mean: 1.658131735193784 msec\nrounds: 1469"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 423.91238899055844,
            "unit": "iter/sec",
            "range": "stddev: 0.0012283804394110796",
            "extra": "mean: 2.358978001046986 msec\nrounds: 1910"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 350.9231253632492,
            "unit": "iter/sec",
            "range": "stddev: 0.001649165269052104",
            "extra": "mean: 2.84962696307026 msec\nrounds: 2410"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 208.54317779398028,
            "unit": "iter/sec",
            "range": "stddev: 0.0030103286595508237",
            "extra": "mean: 4.7951700486117055 msec\nrounds: 4032"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 436.98237658937376,
            "unit": "iter/sec",
            "range": "stddev: 0.0010621434755660184",
            "extra": "mean: 2.288421807316239 msec\nrounds: 1941"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 385.0973685972423,
            "unit": "iter/sec",
            "range": "stddev: 0.00029183947778370464",
            "extra": "mean: 2.59674586622756 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 132204.6222298476,
            "unit": "iter/sec",
            "range": "stddev: 8.760514200833884e-7",
            "extra": "mean: 7.5640320522335855 usec\nrounds: 26613"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.2715580315366,
            "unit": "iter/sec",
            "range": "stddev: 0.0005613619693015172",
            "extra": "mean: 30.05570099999962 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8476818039949106,
            "unit": "iter/sec",
            "range": "stddev: 0.003102340383555672",
            "extra": "mean: 1.1796879386666699 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.036333837913407295,
            "unit": "iter/sec",
            "range": "stddev: 1.1855061662713313",
            "extra": "mean: 27.522553559666676 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3409.5740079749307,
            "unit": "iter/sec",
            "range": "stddev: 0.000031235715607510296",
            "extra": "mean: 293.2917712479678 usec\nrounds: 2518"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 896.8628574066125,
            "unit": "iter/sec",
            "range": "stddev: 0.00002087344827321913",
            "extra": "mean: 1.1149976741055159 msec\nrounds: 672"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.0229699023318,
            "unit": "iter/sec",
            "range": "stddev: 0.0000439545136940791",
            "extra": "mean: 2.777600552184999 msec\nrounds: 297"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 498.21900675638034,
            "unit": "iter/sec",
            "range": "stddev: 0.0004634008452834988",
            "extra": "mean: 2.0071494391802296 msec\nrounds: 781"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12087.8892420853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036618129686459226",
            "extra": "mean: 82.72742907987536 usec\nrounds: 7304"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12341.577801620046,
            "unit": "iter/sec",
            "range": "stddev: 0.000003641358084264145",
            "extra": "mean: 81.02691698534143 usec\nrounds: 8914"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2035.2620374874427,
            "unit": "iter/sec",
            "range": "stddev: 0.000010683672409783375",
            "extra": "mean: 491.33722419080397 usec\nrounds: 1298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2090.143605283969,
            "unit": "iter/sec",
            "range": "stddev: 0.000011074939434776133",
            "extra": "mean: 478.4360258653802 usec\nrounds: 1817"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1191.7007692582138,
            "unit": "iter/sec",
            "range": "stddev: 0.00001961703075873643",
            "extra": "mean: 839.136825112952 usec\nrounds: 892"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 391.31117437825867,
            "unit": "iter/sec",
            "range": "stddev: 0.00006149181177217316",
            "extra": "mean: 2.555510973048155 msec\nrounds: 371"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.932442743598696,
            "unit": "iter/sec",
            "range": "stddev: 0.000576052773575105",
            "extra": "mean: 43.60634456523989 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 472.03386946158946,
            "unit": "iter/sec",
            "range": "stddev: 0.00004636947381488549",
            "extra": "mean: 2.1184920504552323 msec\nrounds: 436"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 390.25951019371536,
            "unit": "iter/sec",
            "range": "stddev: 0.00005880153999401206",
            "extra": "mean: 2.5623975172408335 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 321.7083181111015,
            "unit": "iter/sec",
            "range": "stddev: 0.00007297324571424369",
            "extra": "mean: 3.1084057940169627 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 211.96131820379424,
            "unit": "iter/sec",
            "range": "stddev: 0.00007050326577705753",
            "extra": "mean: 4.717841955665378 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 975.58876606482,
            "unit": "iter/sec",
            "range": "stddev: 0.00001945912574651375",
            "extra": "mean: 1.0250220531276166 msec\nrounds: 847"
          }
        ]
      }
    ]
  }
}