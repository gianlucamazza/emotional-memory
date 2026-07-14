window.BENCHMARK_DATA = {
  "lastUpdate": 1784023316378,
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
          "id": "468f2b302887b35ae02d70ef1101c65398512726",
          "message": "build(deps): bump astral-sh/setup-uv in the github-actions group (#119)\n\nBumps the github-actions group with 1 update: [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv).\n\n\nUpdates `astral-sh/setup-uv` from 8.3.0 to 8.3.2\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/d31148d669074a8d0a63714ba94f3201e7020bc3...11f9893b081a58869d3b5fccaea48c9e9e46f990)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 8.3.2\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-14T11:55:07+02:00",
          "tree_id": "0a23950ab3cc8a26dce2f4581a0b0d01efd80aee",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/468f2b302887b35ae02d70ef1101c65398512726"
        },
        "date": 1784023315767,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 578.2964199363221,
            "unit": "iter/sec",
            "range": "stddev: 0.0007994062251221816",
            "extra": "mean: 1.7292169993203708 msec\nrounds: 1471"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 446.7261949613642,
            "unit": "iter/sec",
            "range": "stddev: 0.0010421442629259014",
            "extra": "mean: 2.2385076390841294 msec\nrounds: 1704"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 303.2785827177825,
            "unit": "iter/sec",
            "range": "stddev: 0.0021501705264628805",
            "extra": "mean: 3.297298447647242 msec\nrounds: 2359"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 218.77644500811363,
            "unit": "iter/sec",
            "range": "stddev: 0.0033791125966614005",
            "extra": "mean: 4.5708759915305945 msec\nrounds: 3306"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 341.29535241915187,
            "unit": "iter/sec",
            "range": "stddev: 0.0018142968747850595",
            "extra": "mean: 2.9300135290792926 msec\nrounds: 1771"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 335.2578400621348,
            "unit": "iter/sec",
            "range": "stddev: 0.000535486985378024",
            "extra": "mean: 2.9827788660055368 msec\nrounds: 403"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136776.20483407518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012988309527284833",
            "extra": "mean: 7.311213242194516 usec\nrounds: 34360"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.100640608428563,
            "unit": "iter/sec",
            "range": "stddev: 0.00026090216849061185",
            "extra": "mean: 35.58637733333588 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7876858952989689,
            "unit": "iter/sec",
            "range": "stddev: 0.02597292036986385",
            "extra": "mean: 1.2695415850000036 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.027156202257374425,
            "unit": "iter/sec",
            "range": "stddev: 10.20071344524407",
            "extra": "mean: 36.824000297333335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3141.4499050681106,
            "unit": "iter/sec",
            "range": "stddev: 0.00001418322307510456",
            "extra": "mean: 318.3243502901947 usec\nrounds: 2418"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 867.8452808871323,
            "unit": "iter/sec",
            "range": "stddev: 0.000016127318648781725",
            "extra": "mean: 1.1522791239675532 msec\nrounds: 726"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.93704926627504,
            "unit": "iter/sec",
            "range": "stddev: 0.000022496813957387954",
            "extra": "mean: 2.8094855594869648 msec\nrounds: 311"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 490.58449002937806,
            "unit": "iter/sec",
            "range": "stddev: 0.001085959240572015",
            "extra": "mean: 2.038384866060719 msec\nrounds: 769"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9771.003390303944,
            "unit": "iter/sec",
            "range": "stddev: 0.000005148264410271314",
            "extra": "mean: 102.34363453320768 usec\nrounds: 7251"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9727.026155144395,
            "unit": "iter/sec",
            "range": "stddev: 0.000005113585014494201",
            "extra": "mean: 102.80634430813406 usec\nrounds: 7836"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1800.4293411058252,
            "unit": "iter/sec",
            "range": "stddev: 0.000009424533556978113",
            "extra": "mean: 555.4230744683372 usec\nrounds: 470"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1787.8498471387736,
            "unit": "iter/sec",
            "range": "stddev: 0.000012422609197082893",
            "extra": "mean: 559.3310878988931 usec\nrounds: 1661"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1012.6571088023406,
            "unit": "iter/sec",
            "range": "stddev: 0.00003705893631392947",
            "extra": "mean: 987.5010912456734 usec\nrounds: 811"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 373.41231870214995,
            "unit": "iter/sec",
            "range": "stddev: 0.000040294311290212",
            "extra": "mean: 2.678004848569669 msec\nrounds: 350"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.26888476678559,
            "unit": "iter/sec",
            "range": "stddev: 0.0009681312064090597",
            "extra": "mean: 49.33670557142293 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 387.7447209600457,
            "unit": "iter/sec",
            "range": "stddev: 0.0005312085943138195",
            "extra": "mean: 2.579016414521457 msec\nrounds: 427"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 301.36632757054906,
            "unit": "iter/sec",
            "range": "stddev: 0.000557682117896453",
            "extra": "mean: 3.318220745036297 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 259.10656509995744,
            "unit": "iter/sec",
            "range": "stddev: 0.0004718672199458379",
            "extra": "mean: 3.85941591103345 msec\nrounds: 281"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 141.63479550032918,
            "unit": "iter/sec",
            "range": "stddev: 0.0006924129783718752",
            "extra": "mean: 7.060411931033402 msec\nrounds: 174"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 820.162484671584,
            "unit": "iter/sec",
            "range": "stddev: 0.00004405849252016627",
            "extra": "mean: 1.219270594168213 msec\nrounds: 754"
          }
        ]
      }
    ]
  }
}