window.BENCHMARK_DATA = {
  "lastUpdate": 1789122675783,
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
          "id": "8a777372a5e251ae6d40bf257e902c661561fa7a",
          "message": "build(deps): bump the github-actions group across 1 directory with 4 updates (#129)\n\nBumps the github-actions group with 4 updates in the / directory: [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv), [anchore/sbom-action](https://github.com/anchore/sbom-action), [actions/attest](https://github.com/actions/attest) and [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action).\n\n\nUpdates `astral-sh/setup-uv` from 9.0.0 to 10.0.1\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/c771a70e6277c0a99b617c7a806ffedaca235ff9...20cfd1bf945f4377ade1205e4dbc17946fc9a30d)\n\nUpdates `anchore/sbom-action` from 0.24.0 to 0.24.2\n- [Release notes](https://github.com/anchore/sbom-action/releases)\n- [Changelog](https://github.com/anchore/sbom-action/blob/main/RELEASE.md)\n- [Commits](https://github.com/anchore/sbom-action/compare/e22c389904149dbc22b58101806040fa8d37a610...3ad7283483fc7af8ff2b4ea19663c2d5ca935e26)\n\nUpdates `actions/attest` from 4.2.1 to 4.2.2\n- [Release notes](https://github.com/actions/attest/releases)\n- [Changelog](https://github.com/actions/attest/blob/main/RELEASE.md)\n- [Commits](https://github.com/actions/attest/compare/508db95dd578ae2727ebd6217d5ba78e4fbda05d...1e69f48acb82d1966a394da916b4c1698aa569d6)\n\nUpdates `zizmorcore/zizmor-action` from 0.6.1 to 0.6.2\n- [Release notes](https://github.com/zizmorcore/zizmor-action/releases)\n- [Commits](https://github.com/zizmorcore/zizmor-action/compare/6fc4b006235f201fdab3722e17240ab420d580e5...3dc1ecc9bcb9e94e9b2c709687979e1298497054)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 10.0.1\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: github-actions\n- dependency-name: anchore/sbom-action\n  dependency-version: 0.24.2\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n- dependency-name: actions/attest\n  dependency-version: 4.2.2\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n- dependency-name: zizmorcore/zizmor-action\n  dependency-version: 0.6.2\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-09-11T12:24:48+02:00",
          "tree_id": "c2f47a65ffc4d5bc0b547261a414bd5ce55d5fdb",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8a777372a5e251ae6d40bf257e902c661561fa7a"
        },
        "date": 1789122673875,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 521.3505313592557,
            "unit": "iter/sec",
            "range": "stddev: 0.0009377925177851085",
            "extra": "mean: 1.9180952926101715 msec\nrounds: 1678"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 433.7490329305985,
            "unit": "iter/sec",
            "range": "stddev: 0.0010849467216497877",
            "extra": "mean: 2.305480644518241 msec\nrounds: 1806"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 404.84971641663174,
            "unit": "iter/sec",
            "range": "stddev: 0.0012174191445168542",
            "extra": "mean: 2.470052366223959 msec\nrounds: 2108"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 253.29507178011903,
            "unit": "iter/sec",
            "range": "stddev: 0.002366897259872363",
            "extra": "mean: 3.9479646918203057 msec\nrounds: 3472"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 457.7396750823546,
            "unit": "iter/sec",
            "range": "stddev: 0.0009683819247445649",
            "extra": "mean: 2.184647856491977 msec\nrounds: 1756"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 377.7645004913983,
            "unit": "iter/sec",
            "range": "stddev: 0.0002548693249095427",
            "extra": "mean: 2.6471518596882295 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138626.99780424827,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012790413858790818",
            "extra": "mean: 7.213602082128873 usec\nrounds: 43416"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 26.821471138382908,
            "unit": "iter/sec",
            "range": "stddev: 0.002902577000690572",
            "extra": "mean: 37.28356266666329 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8010932703577957,
            "unit": "iter/sec",
            "range": "stddev: 0.020073774820085044",
            "extra": "mean: 1.2482940963333344 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03465322145721922,
            "unit": "iter/sec",
            "range": "stddev: 2.326237080868917",
            "extra": "mean: 28.85734595366667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3144.1475748985386,
            "unit": "iter/sec",
            "range": "stddev: 0.000012994654685584078",
            "extra": "mean: 318.0512288874576 usec\nrounds: 2534"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.1059409129772,
            "unit": "iter/sec",
            "range": "stddev: 0.000017096585796702802",
            "extra": "mean: 1.159950248041432 msec\nrounds: 766"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 353.20210211030155,
            "unit": "iter/sec",
            "range": "stddev: 0.00003306556555730949",
            "extra": "mean: 2.8312402276918207 msec\nrounds: 325"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 486.8924932425837,
            "unit": "iter/sec",
            "range": "stddev: 0.0008443810934971581",
            "extra": "mean: 2.053841482213552 msec\nrounds: 759"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10521.558060104653,
            "unit": "iter/sec",
            "range": "stddev: 0.000005065784952735551",
            "extra": "mean: 95.04295792386222 usec\nrounds: 6108"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10541.47891621979,
            "unit": "iter/sec",
            "range": "stddev: 0.000004938231520953024",
            "extra": "mean: 94.86334962557639 usec\nrounds: 8409"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1785.2249441816207,
            "unit": "iter/sec",
            "range": "stddev: 0.000015190166315190946",
            "extra": "mean: 560.1534995683237 usec\nrounds: 1157"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1793.8736909247045,
            "unit": "iter/sec",
            "range": "stddev: 0.000012824157022072045",
            "extra": "mean: 557.4528491381803 usec\nrounds: 1624"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1015.4491277369741,
            "unit": "iter/sec",
            "range": "stddev: 0.000020960430277720005",
            "extra": "mean: 984.7859165811644 usec\nrounds: 971"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.9784850520065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000452080108058411",
            "extra": "mean: 2.7028598699717197 msec\nrounds: 1069"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 26.430715067076193,
            "unit": "iter/sec",
            "range": "stddev: 0.0005475336871903335",
            "extra": "mean: 37.83476903527535 msec\nrounds: 652"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 464.18309338035584,
            "unit": "iter/sec",
            "range": "stddev: 0.000025649454333538515",
            "extra": "mean: 2.1543223229386146 msec\nrounds: 2790"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.7822441124829,
            "unit": "iter/sec",
            "range": "stddev: 0.00005049706680840497",
            "extra": "mean: 2.6682160526790364 msec\nrounds: 1101"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 305.38474463954617,
            "unit": "iter/sec",
            "range": "stddev: 0.00003092811951379274",
            "extra": "mean: 3.2745578079884994 msec\nrounds: 651"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.33646362831544,
            "unit": "iter/sec",
            "range": "stddev: 0.00011904539718515735",
            "extra": "mean: 5.093297401409347 msec\nrounds: 284"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 856.8387249313779,
            "unit": "iter/sec",
            "range": "stddev: 0.000024655937276715825",
            "extra": "mean: 1.1670807713319533 msec\nrounds: 1172"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5244.263917244958,
            "unit": "iter/sec",
            "range": "stddev: 0.000007446790922703553",
            "extra": "mean: 190.68452995122027 usec\nrounds: 4691"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 77.6821678382138,
            "unit": "iter/sec",
            "range": "stddev: 0.0005574585086171072",
            "extra": "mean: 12.872967217942069 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1899.5720854263536,
            "unit": "iter/sec",
            "range": "stddev: 0.000018960328773515907",
            "extra": "mean: 526.434352069115 usec\nrounds: 1619"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.925881960577854,
            "unit": "iter/sec",
            "range": "stddev: 0.004627381947013851",
            "extra": "mean: 34.571115285710825 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 18657.601487052198,
            "unit": "iter/sec",
            "range": "stddev: 0.000005460402564038736",
            "extra": "mean: 53.59745735238097 usec\nrounds: 11853"
          }
        ]
      }
    ]
  }
}