window.BENCHMARK_DATA = {
  "lastUpdate": 1782776845301,
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
          "id": "986bb314760cd3d67c51e650e88d8f525ad2ad41",
          "message": "build(deps): bump the github-actions group with 2 updates (#89)\n\nBumps the github-actions group with 2 updates: [actions/setup-python](https://github.com/actions/setup-python) and [actions/attest](https://github.com/actions/attest).\n\n\nUpdates `actions/setup-python` from 6.2.0 to 6.3.0\n- [Release notes](https://github.com/actions/setup-python/releases)\n- [Commits](https://github.com/actions/setup-python/compare/a309ff8b426b58ec0e2a45f0f869d46889d02405...ece7cb06caefa5fff74198d8649806c4678c61a1)\n\nUpdates `actions/attest` from 4.1.0 to 4.1.1\n- [Release notes](https://github.com/actions/attest/releases)\n- [Changelog](https://github.com/actions/attest/blob/main/RELEASE.md)\n- [Commits](https://github.com/actions/attest/compare/59d89421af93a897026c735860bf21b6eb4f7b26...a1948c3f048ba23858d222213b7c278aabede763)\n\n---\nupdated-dependencies:\n- dependency-name: actions/setup-python\n  dependency-version: 6.3.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: github-actions\n- dependency-name: actions/attest\n  dependency-version: 4.1.1\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-06-30T01:41:36+02:00",
          "tree_id": "5ceb7b3a6e0eb99dc07066860b71600d2671680c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/986bb314760cd3d67c51e650e88d8f525ad2ad41"
        },
        "date": 1782776843969,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 549.5743132546398,
            "unit": "iter/sec",
            "range": "stddev: 0.0008455490527765837",
            "extra": "mean: 1.8195901370970007 msec\nrounds: 1612"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 451.93467356316927,
            "unit": "iter/sec",
            "range": "stddev: 0.0010456408621525026",
            "extra": "mean: 2.21270917789012 msec\nrounds: 1782"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 380.9058217491546,
            "unit": "iter/sec",
            "range": "stddev: 0.0014767721823153905",
            "extra": "mean: 2.625320861224719 msec\nrounds: 2205"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 261.23311140863245,
            "unit": "iter/sec",
            "range": "stddev: 0.0024288103516886908",
            "extra": "mean: 3.8279986583927164 msec\nrounds: 3223"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 462.63164442064715,
            "unit": "iter/sec",
            "range": "stddev: 0.0009424616523642413",
            "extra": "mean: 2.1615469068318887 msec\nrounds: 1771"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.7906378177073,
            "unit": "iter/sec",
            "range": "stddev: 0.0003030313069998842",
            "extra": "mean: 2.6192365682823993 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134955.535652493,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011718331920307798",
            "extra": "mean: 7.409847955959169 usec\nrounds: 47552"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.20142612920274,
            "unit": "iter/sec",
            "range": "stddev: 0.0007169812932252883",
            "extra": "mean: 35.459199666661334 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8229949889308269,
            "unit": "iter/sec",
            "range": "stddev: 0.004214739614055603",
            "extra": "mean: 1.2150742270000023 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03387429720710618,
            "unit": "iter/sec",
            "range": "stddev: 2.197114224539956",
            "extra": "mean: 29.520907662999992 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3181.9256728635123,
            "unit": "iter/sec",
            "range": "stddev: 0.000014153183573936408",
            "extra": "mean: 314.2750971615466 usec\nrounds: 2501"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 867.6039464201887,
            "unit": "iter/sec",
            "range": "stddev: 0.00002356432726745031",
            "extra": "mean: 1.1525996442571396 msec\nrounds: 714"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.023410736204,
            "unit": "iter/sec",
            "range": "stddev: 0.000029987643180577346",
            "extra": "mean: 2.8167156580641333 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 495.51238237329113,
            "unit": "iter/sec",
            "range": "stddev: 0.0011116084898217516",
            "extra": "mean: 2.0181130392956685 msec\nrounds: 738"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10285.819914670576,
            "unit": "iter/sec",
            "range": "stddev: 0.000004768043162576923",
            "extra": "mean: 97.22122381062776 usec\nrounds: 7274"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10217.521239245158,
            "unit": "iter/sec",
            "range": "stddev: 0.000005091416815438682",
            "extra": "mean: 97.8710957956254 usec\nrounds: 7944"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1810.9906837715196,
            "unit": "iter/sec",
            "range": "stddev: 0.00001185264769374088",
            "extra": "mean: 552.1839559756472 usec\nrounds: 477"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1814.2476812918082,
            "unit": "iter/sec",
            "range": "stddev: 0.00001080155472142181",
            "extra": "mean: 551.192657051084 usec\nrounds: 1560"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 996.4743836265847,
            "unit": "iter/sec",
            "range": "stddev: 0.000046770776676679426",
            "extra": "mean: 1.0035380903225872 msec\nrounds: 775"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.8542446747564,
            "unit": "iter/sec",
            "range": "stddev: 0.00011039391975376478",
            "extra": "mean: 2.6964771587743646 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.2519002604947,
            "unit": "iter/sec",
            "range": "stddev: 0.001728523921306452",
            "extra": "mean: 43.007237636358425 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 473.5715633050881,
            "unit": "iter/sec",
            "range": "stddev: 0.00002743004699963468",
            "extra": "mean: 2.111613275554242 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 377.7884830873504,
            "unit": "iter/sec",
            "range": "stddev: 0.00003662080691719137",
            "extra": "mean: 2.6469838144027933 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 304.2061328830745,
            "unit": "iter/sec",
            "range": "stddev: 0.000035793543613128444",
            "extra": "mean: 3.287244706484477 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.71410946618678,
            "unit": "iter/sec",
            "range": "stddev: 0.000050164190388831006",
            "extra": "mean: 5.162246584699872 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 848.5179653226347,
            "unit": "iter/sec",
            "range": "stddev: 0.000023705224537070263",
            "extra": "mean: 1.17852543006531 msec\nrounds: 765"
          }
        ]
      }
    ]
  }
}