window.BENCHMARK_DATA = {
  "lastUpdate": 1789982553424,
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
          "id": "d1c793cdece93c89f802ff08c6630332a0083457",
          "message": "build(deps): bump the github-actions group with 3 updates (#138)\n\nBumps the github-actions group with 3 updates: [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv), [codecov/codecov-action](https://github.com/codecov/codecov-action) and [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action).\n\n\nUpdates `astral-sh/setup-uv` from 10.0.1 to 10.1.0\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/20cfd1bf945f4377ade1205e4dbc17946fc9a30d...bec219d24cd3e171d82865faccec33120bb574f4)\n\nUpdates `codecov/codecov-action` from 7.0.0 to 7.1.0\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/fb8b3582c8e4def4969c97caa2f19720cb33a72f...0b35c9ecc4f0529d0eb674914510c22f85b196b4)\n\nUpdates `zizmorcore/zizmor-action` from 0.6.3 to 0.6.4\n- [Release notes](https://github.com/zizmorcore/zizmor-action/releases)\n- [Commits](https://github.com/zizmorcore/zizmor-action/compare/70fb788f84895a7701f5643d103d587e460b5c99...cc914d7f3750a2d13d75c7f184a1060aa0e9d482)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 10.1.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: github-actions\n- dependency-name: codecov/codecov-action\n  dependency-version: 7.1.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: github-actions\n- dependency-name: zizmorcore/zizmor-action\n  dependency-version: 0.6.4\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-09-21T11:15:50+02:00",
          "tree_id": "83662c34d4cceafadcc6f00b4b8b9a05ab409b48",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/d1c793cdece93c89f802ff08c6630332a0083457"
        },
        "date": 1789982551297,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 493.29799026518117,
            "unit": "iter/sec",
            "range": "stddev: 0.000833656092478957",
            "extra": "mean: 2.027172256393001 msec\nrounds: 1486"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 481.3072795825883,
            "unit": "iter/sec",
            "range": "stddev: 0.0009694348767349375",
            "extra": "mean: 2.077674787855371 msec\nrounds: 1301"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 455.2055991060566,
            "unit": "iter/sec",
            "range": "stddev: 0.0008870738861204706",
            "extra": "mean: 2.196809533898141 msec\nrounds: 1534"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 356.3723806987109,
            "unit": "iter/sec",
            "range": "stddev: 0.0015727365427377015",
            "extra": "mean: 2.806053594948575 msec\nrounds: 2138"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 476.1042972505711,
            "unit": "iter/sec",
            "range": "stddev: 0.0007675428885031198",
            "extra": "mean: 2.1003801179171155 msec\nrounds: 1306"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 320.7604514984043,
            "unit": "iter/sec",
            "range": "stddev: 0.0005262462327180392",
            "extra": "mean: 3.117591321899529 msec\nrounds: 379"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 130983.12846170446,
            "unit": "iter/sec",
            "range": "stddev: 8.838941677480343e-7",
            "extra": "mean: 7.634571045479114 usec\nrounds: 44380"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 12.278727105290626,
            "unit": "iter/sec",
            "range": "stddev: 0.0003668267244287632",
            "extra": "mean: 81.44166666666308 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5647080872922102,
            "unit": "iter/sec",
            "range": "stddev: 0.007611791353832125",
            "extra": "mean: 1.7708264189999927 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02996108352075107,
            "unit": "iter/sec",
            "range": "stddev: 1.2837871226170643",
            "extra": "mean: 33.37663003100002 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3496.62067210979,
            "unit": "iter/sec",
            "range": "stddev: 0.000010589163363443153",
            "extra": "mean: 285.99041582529463 usec\nrounds: 2376"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 899.8648414897606,
            "unit": "iter/sec",
            "range": "stddev: 0.000015205721519736286",
            "extra": "mean: 1.1112779985319372 msec\nrounds: 681"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.27834698872,
            "unit": "iter/sec",
            "range": "stddev: 0.00004749896056109104",
            "extra": "mean: 2.775631697986305 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 446.87469996365627,
            "unit": "iter/sec",
            "range": "stddev: 0.0004354834661640069",
            "extra": "mean: 2.2377637402191906 msec\nrounds: 639"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10954.444091775116,
            "unit": "iter/sec",
            "range": "stddev: 0.000004438170795250251",
            "extra": "mean: 91.28715173696729 usec\nrounds: 6966"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11009.168487480698,
            "unit": "iter/sec",
            "range": "stddev: 0.000003821572784143773",
            "extra": "mean: 90.83338138908223 usec\nrounds: 8393"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1802.5548697751772,
            "unit": "iter/sec",
            "range": "stddev: 0.000014084820015283777",
            "extra": "mean: 554.7681331468842 usec\nrounds: 1074"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1794.8978674907971,
            "unit": "iter/sec",
            "range": "stddev: 0.00003421711971625345",
            "extra": "mean: 557.1347641066419 usec\nrounds: 1471"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1181.8602226278047,
            "unit": "iter/sec",
            "range": "stddev: 0.00003430096391241132",
            "extra": "mean: 846.1237469999219 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 381.0793238870326,
            "unit": "iter/sec",
            "range": "stddev: 0.0000880552690131097",
            "extra": "mean: 2.6241255752212904 msec\nrounds: 1130"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.667764886785363,
            "unit": "iter/sec",
            "range": "stddev: 0.0025905875313834374",
            "extra": "mean: 46.151506868614554 msec\nrounds: 685"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 459.62832335679144,
            "unit": "iter/sec",
            "range": "stddev: 0.0000965029795923348",
            "extra": "mean: 2.175670969745133 msec\nrounds: 2545"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.8065089257746,
            "unit": "iter/sec",
            "range": "stddev: 0.00011173795859874084",
            "extra": "mean: 2.6538819694247517 msec\nrounds: 1112"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 308.4345967936121,
            "unit": "iter/sec",
            "range": "stddev: 0.0001480171627517331",
            "extra": "mean: 3.2421784404074043 msec\nrounds: 688"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 201.9636634387653,
            "unit": "iter/sec",
            "range": "stddev: 0.0002829059160093777",
            "extra": "mean: 4.95138572440877 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 958.2332303886008,
            "unit": "iter/sec",
            "range": "stddev: 0.000025600961832518588",
            "extra": "mean: 1.0435872690351817 msec\nrounds: 1182"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5467.962110143957,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060091877229164364",
            "extra": "mean: 182.88349111725512 usec\nrounds: 4897"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 79.03827991458176,
            "unit": "iter/sec",
            "range": "stddev: 0.00035349910914580975",
            "extra": "mean: 12.652097199998783 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2279.36606208719,
            "unit": "iter/sec",
            "range": "stddev: 0.000022145565882594505",
            "extra": "mean: 438.71847380420814 usec\nrounds: 1756"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.64844862102057,
            "unit": "iter/sec",
            "range": "stddev: 0.00809744450780616",
            "extra": "mean: 34.90590409374761 msec\nrounds: 32"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 21677.049436362184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036417913974319936",
            "extra": "mean: 46.13173960486288 usec\nrounds: 15488"
          }
        ]
      }
    ]
  }
}