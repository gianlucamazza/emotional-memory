window.BENCHMARK_DATA = {
  "lastUpdate": 1783414355881,
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
          "id": "1ccbb491e5304304e932e5cf2315c9f731f2637f",
          "message": "build(deps): bump astral-sh/setup-uv in the github-actions group (#101)\n\nBumps the github-actions group with 1 update: [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv).\n\n\nUpdates `astral-sh/setup-uv` from 8.2.0 to 8.3.0\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/fac544c07dec837d0ccb6301d7b5580bf5edae39...d31148d669074a8d0a63714ba94f3201e7020bc3)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 8.3.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: github-actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-07T10:47:37+02:00",
          "tree_id": "e984559eb685a8c5d5d65d2da44d6cbd5bc8da76",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1ccbb491e5304304e932e5cf2315c9f731f2637f"
        },
        "date": 1783414354826,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 661.6158926818655,
            "unit": "iter/sec",
            "range": "stddev: 0.0007353809031986407",
            "extra": "mean: 1.5114509960552667 msec\nrounds: 1521"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 393.42291572903014,
            "unit": "iter/sec",
            "range": "stddev: 0.0013518038567637638",
            "extra": "mean: 2.5417939830651592 msec\nrounds: 2362"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 310.6455489893646,
            "unit": "iter/sec",
            "range": "stddev: 0.0018657654142362954",
            "extra": "mean: 3.2191029398404045 msec\nrounds: 3258"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 206.27994873853746,
            "unit": "iter/sec",
            "range": "stddev: 0.0028707415734532055",
            "extra": "mean: 4.847780921583964 msec\nrounds: 5101"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 433.34435301060773,
            "unit": "iter/sec",
            "range": "stddev: 0.0010839694952693066",
            "extra": "mean: 2.307633624512747 msec\nrounds: 2309"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 423.465759173755,
            "unit": "iter/sec",
            "range": "stddev: 0.0009439844749056712",
            "extra": "mean: 2.361466017821959 msec\nrounds: 505"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 167738.35109754864,
            "unit": "iter/sec",
            "range": "stddev: 7.745680448002828e-7",
            "extra": "mean: 5.961665853138426 usec\nrounds: 47527"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 42.43751525009101,
            "unit": "iter/sec",
            "range": "stddev: 0.000566239801696704",
            "extra": "mean: 23.564056333337174 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.9738628279719115,
            "unit": "iter/sec",
            "range": "stddev: 0.0024297707565913746",
            "extra": "mean: 1.0268386586666622 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.046617957248118264,
            "unit": "iter/sec",
            "range": "stddev: 0.13382000634715935",
            "extra": "mean: 21.45096136833334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4188.314269428357,
            "unit": "iter/sec",
            "range": "stddev: 0.000009001487167739811",
            "extra": "mean: 238.7595427829453 usec\nrounds: 3097"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1089.0496941646022,
            "unit": "iter/sec",
            "range": "stddev: 0.00002629766844855047",
            "extra": "mean: 918.2317440225617 usec\nrounds: 711"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 429.46464395433475,
            "unit": "iter/sec",
            "range": "stddev: 0.00006251956494485513",
            "extra": "mean: 2.32848038616732 msec\nrounds: 347"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 538.2135318848688,
            "unit": "iter/sec",
            "range": "stddev: 0.0004473393314075949",
            "extra": "mean: 1.8579986209152277 msec\nrounds: 918"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13069.147856918937,
            "unit": "iter/sec",
            "range": "stddev: 0.000003249420741216916",
            "extra": "mean: 76.51608283478024 usec\nrounds: 8861"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12677.188147751878,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037136992249999383",
            "extra": "mean: 78.88184574884109 usec\nrounds: 9115"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2383.279406714822,
            "unit": "iter/sec",
            "range": "stddev: 0.00001552778681950668",
            "extra": "mean: 419.58991345392764 usec\nrounds: 1167"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2342.7546837838886,
            "unit": "iter/sec",
            "range": "stddev: 0.000014995001033217956",
            "extra": "mean: 426.84793543336554 usec\nrounds: 1905"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1418.074159099822,
            "unit": "iter/sec",
            "range": "stddev: 0.00002209443482120407",
            "extra": "mean: 705.1817379105117 usec\nrounds: 1034"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 442.74698703919665,
            "unit": "iter/sec",
            "range": "stddev: 0.00010415546446915631",
            "extra": "mean: 2.258626324455302 msec\nrounds: 413"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 34.19605789427567,
            "unit": "iter/sec",
            "range": "stddev: 0.0002733273600752851",
            "extra": "mean: 29.243136828569863 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 532.3291595958021,
            "unit": "iter/sec",
            "range": "stddev: 0.00003951458349094408",
            "extra": "mean: 1.878536957771204 msec\nrounds: 521"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 441.8540844437973,
            "unit": "iter/sec",
            "range": "stddev: 0.00004712730852656832",
            "extra": "mean: 2.263190576270881 msec\nrounds: 413"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 357.55437836484214,
            "unit": "iter/sec",
            "range": "stddev: 0.00023817406446208092",
            "extra": "mean: 2.796777386905937 msec\nrounds: 336"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 241.09655074183297,
            "unit": "iter/sec",
            "range": "stddev: 0.00008062173588435535",
            "extra": "mean: 4.147715912662738 msec\nrounds: 229"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1140.4180917382112,
            "unit": "iter/sec",
            "range": "stddev: 0.000024690819210349048",
            "extra": "mean: 876.8713923819047 usec\nrounds: 1050"
          }
        ]
      }
    ]
  }
}