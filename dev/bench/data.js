window.BENCHMARK_DATA = {
  "lastUpdate": 1782952160005,
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
          "id": "7607f3f80cccf77db45e14ed239f2c843e110a6c",
          "message": "docs: propagate Addenda W/X to residual surfaces + benchmarks index (#94)\n\n- research/index.md: ladder rows for Addendum W (arousal calibration,\n  measurement-only) and Addendum X (MADial-Bench FAIL, construct boundary)\n  with absolute GitHub closure links — X closure now reachable from mkdocs;\n  survey count 29->33\n- problem_register 2026-06: section 7 executed list + item 3 (independent\n  corpus = Addendum X, EXECUTED/FAIL with numbers) + item 4 (Addendum W);\n  C1/C3 marked RESOLVED (footnote and \"When NOT to use\" live in README);\n  dangling reference to nonexistent issue removed\n- README: validation section extended with U/V/W/T/T2A/X paragraph; new\n  \"When NOT to use\" bullet for counter-congruent emotional-support recall;\n  abstract mentions the third-party FAIL; study count 12+ -> 20+\n- validation_report 2026-06: post-report update banner (U/V/T/T2A/W/X);\n  added to mkdocs nav (pre-existing gap)\n- review_response 2026-06: section 3.4 construct validity updated with the\n  Addendum X boundary; executed list extended; 29 systems -> 33 papers\n- 10_scientific_quality_bar: 2026-07-02 update note (external validity vs\n  Gate 2 scope)\n- NEW benchmarks/README.md: chronological addenda index (prereg -> closure\n  -> verdict, A through X) + harness directory map\n\nmake check-all green (mkdocs strict, claim-matrix, reproduce-paper, bundle).\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T02:23:22+02:00",
          "tree_id": "561e8b3f6b914cb4b54f658a6367127e56f4f15e",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7607f3f80cccf77db45e14ed239f2c843e110a6c"
        },
        "date": 1782952158900,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 562.2037134909918,
            "unit": "iter/sec",
            "range": "stddev: 0.0008403304301348412",
            "extra": "mean: 1.778714682958107 msec\nrounds: 1555"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 454.2030323430689,
            "unit": "iter/sec",
            "range": "stddev: 0.0010525706963797257",
            "extra": "mean: 2.201658572910362 msec\nrounds: 1639"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 474.3868874777231,
            "unit": "iter/sec",
            "range": "stddev: 0.000981254313941026",
            "extra": "mean: 2.1079840661636315 msec\nrounds: 1723"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 261.56422214307395,
            "unit": "iter/sec",
            "range": "stddev: 0.0023296809771855696",
            "extra": "mean: 3.8231528448604353 msec\nrounds: 3152"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 410.13860740868614,
            "unit": "iter/sec",
            "range": "stddev: 0.0012089278221878844",
            "extra": "mean: 2.4382001156100417 msec\nrounds: 1877"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.6995453606821,
            "unit": "iter/sec",
            "range": "stddev: 0.000244949562816722",
            "extra": "mean: 2.5994311978259077 msec\nrounds: 460"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138993.48290465694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011150624593759344",
            "extra": "mean: 7.194581926448691 usec\nrounds: 44009"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.270203095683154,
            "unit": "iter/sec",
            "range": "stddev: 0.00029532322934881924",
            "extra": "mean: 35.37293300000025 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7959821071086628,
            "unit": "iter/sec",
            "range": "stddev: 0.03838260258716745",
            "extra": "mean: 1.2563096469999995 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.030504999105427328,
            "unit": "iter/sec",
            "range": "stddev: 2.4144856616565895",
            "extra": "mean: 32.781512189 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3171.662650094728,
            "unit": "iter/sec",
            "range": "stddev: 0.000011981625408724326",
            "extra": "mean: 315.2920440546011 usec\nrounds: 2565"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 856.6261249922505,
            "unit": "iter/sec",
            "range": "stddev: 0.000025581553066760583",
            "extra": "mean: 1.1673704207994433 msec\nrounds: 625"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.7056841394615,
            "unit": "iter/sec",
            "range": "stddev: 0.000036195367095535815",
            "extra": "mean: 2.8113129606552203 msec\nrounds: 305"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 497.5289300055769,
            "unit": "iter/sec",
            "range": "stddev: 0.0004310126372980369",
            "extra": "mean: 2.0099333720931383 msec\nrounds: 774"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10417.466375952885,
            "unit": "iter/sec",
            "range": "stddev: 0.000005676930041892192",
            "extra": "mean: 95.99263044499436 usec\nrounds: 6451"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10409.757183670228,
            "unit": "iter/sec",
            "range": "stddev: 0.000005239439322620548",
            "extra": "mean: 96.06372006147257 usec\nrounds: 7148"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1831.1385849458875,
            "unit": "iter/sec",
            "range": "stddev: 0.000024064243981164077",
            "extra": "mean: 546.1083110918944 usec\nrounds: 1154"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1849.15696654697,
            "unit": "iter/sec",
            "range": "stddev: 0.000011643976582964568",
            "extra": "mean: 540.7869737891174 usec\nrounds: 1755"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 996.1666530797869,
            "unit": "iter/sec",
            "range": "stddev: 0.00006699716037524681",
            "extra": "mean: 1.0038480980148872 msec\nrounds: 806"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 369.6700292984166,
            "unit": "iter/sec",
            "range": "stddev: 0.00005991537964182133",
            "extra": "mean: 2.7051151587751487 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.60485157164873,
            "unit": "iter/sec",
            "range": "stddev: 0.0008571463213088495",
            "extra": "mean: 48.532259333328625 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 474.10355807862277,
            "unit": "iter/sec",
            "range": "stddev: 0.00003696302732561624",
            "extra": "mean: 2.109243820174337 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.3084939135763,
            "unit": "iter/sec",
            "range": "stddev: 0.00005968137282449622",
            "extra": "mean: 2.629444295891131 msec\nrounds: 365"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 287.636993354423,
            "unit": "iter/sec",
            "range": "stddev: 0.0005059038075006732",
            "extra": "mean: 3.476604272412942 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 160.16637798602463,
            "unit": "iter/sec",
            "range": "stddev: 0.0006039484821187774",
            "extra": "mean: 6.243507611112086 msec\nrounds: 180"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 847.8539041739293,
            "unit": "iter/sec",
            "range": "stddev: 0.00003219612363282574",
            "extra": "mean: 1.179448481721987 msec\nrounds: 766"
          }
        ]
      }
    ]
  }
}