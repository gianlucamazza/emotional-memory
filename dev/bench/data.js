window.BENCHMARK_DATA = {
  "lastUpdate": 1780173337694,
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
          "id": "de7343e5d4a85dbb75af3ed56c5d56d98a64fa46",
          "message": "fix(ci): keep SBOM out of dist/ so PyPI publish only sees wheel+sdist (#41)\n\nThe release workflow wrote the CycloneDX SBOM to dist/sbom.cdx.json, then\n'Publish to PyPI' (packages-dir defaults to dist/) handed the whole dist/ to\ngh-action-pypi-publish, which rejected it: 'InvalidDistribution: Unknown\ndistribution format: sbom.cdx.json'. This blocked on-tag publishing\nindependently of the (now configured) Trusted Publisher. Write the SBOM to the\nrepo root instead and update the attest + upload paths; dist/ now contains only\nthe distributables.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T22:28:09+02:00",
          "tree_id": "b1c868f59ca3fb9198290683ac2d360a5eaac896",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/de7343e5d4a85dbb75af3ed56c5d56d98a64fa46"
        },
        "date": 1780173336651,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 622.0884710343822,
            "unit": "iter/sec",
            "range": "stddev: 0.00075398166924323",
            "extra": "mean: 1.607488398454391 msec\nrounds: 1423"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 410.80065350754273,
            "unit": "iter/sec",
            "range": "stddev: 0.0013069450921067134",
            "extra": "mean: 2.434270713694566 msec\nrounds: 1935"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 377.96022765277155,
            "unit": "iter/sec",
            "range": "stddev: 0.0015695123485803169",
            "extra": "mean: 2.645781028893576 msec\nrounds: 2215"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 148.7493880580337,
            "unit": "iter/sec",
            "range": "stddev: 0.005085104598524303",
            "extra": "mean: 6.722716732184847 msec\nrounds: 4294"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 443.66273791094966,
            "unit": "iter/sec",
            "range": "stddev: 0.0010699108873496173",
            "extra": "mean: 2.2539643620030954 msec\nrounds: 1837"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 370.48929285929967,
            "unit": "iter/sec",
            "range": "stddev: 0.0003935007336525626",
            "extra": "mean: 2.6991333333343293 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135074.09750470915,
            "unit": "iter/sec",
            "range": "stddev: 8.374874158283358e-7",
            "extra": "mean: 7.403343931023759 usec\nrounds: 45387"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.0407166698826,
            "unit": "iter/sec",
            "range": "stddev: 0.00047050122253777284",
            "extra": "mean: 30.265687333335716 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8437960957632809,
            "unit": "iter/sec",
            "range": "stddev: 0.008042013003542214",
            "extra": "mean: 1.1851204396666712 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02335465886615547,
            "unit": "iter/sec",
            "range": "stddev: 1.4546308654170983",
            "extra": "mean: 42.818009277333324 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3487.7749971143944,
            "unit": "iter/sec",
            "range": "stddev: 0.000018453367771445728",
            "extra": "mean: 286.7157430818641 usec\nrounds: 2168"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 908.9310297656654,
            "unit": "iter/sec",
            "range": "stddev: 0.000021215197948771526",
            "extra": "mean: 1.100193488011751 msec\nrounds: 709"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.605136763545,
            "unit": "iter/sec",
            "range": "stddev: 0.000046052045086984716",
            "extra": "mean: 2.7578208321193767 msec\nrounds: 274"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 505.11848661512835,
            "unit": "iter/sec",
            "range": "stddev: 0.00045825797346420806",
            "extra": "mean: 1.9797335209430638 msec\nrounds: 764"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12338.401416368319,
            "unit": "iter/sec",
            "range": "stddev: 0.000004138134075356453",
            "extra": "mean: 81.04777647073341 usec\nrounds: 7140"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12317.138518304653,
            "unit": "iter/sec",
            "range": "stddev: 0.000003665916496820308",
            "extra": "mean: 81.18768807494432 usec\nrounds: 8579"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2031.8104281362193,
            "unit": "iter/sec",
            "range": "stddev: 0.00001347620866453179",
            "extra": "mean: 492.17190056323335 usec\nrounds: 1066"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2046.304888332946,
            "unit": "iter/sec",
            "range": "stddev: 0.000013842809395505304",
            "extra": "mean: 488.685730900377 usec\nrounds: 1479"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1200.6064725062572,
            "unit": "iter/sec",
            "range": "stddev: 0.000026034315373465978",
            "extra": "mean: 832.9123846155079 usec\nrounds: 871"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 391.82069272583277,
            "unit": "iter/sec",
            "range": "stddev: 0.00005681896988107243",
            "extra": "mean: 2.5521878210238533 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.26429749949798,
            "unit": "iter/sec",
            "range": "stddev: 0.0007254826391207243",
            "extra": "mean: 51.90949735001027 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.6464569766884,
            "unit": "iter/sec",
            "range": "stddev: 0.0000565316101393989",
            "extra": "mean: 2.1202321891912908 msec\nrounds: 444"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.3098296994618,
            "unit": "iter/sec",
            "range": "stddev: 0.00010635186751717267",
            "extra": "mean: 2.6573847427760406 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 321.22838588567936,
            "unit": "iter/sec",
            "range": "stddev: 0.00011849119141605269",
            "extra": "mean: 3.113049916939426 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 209.45667897401776,
            "unit": "iter/sec",
            "range": "stddev: 0.00011388041284722298",
            "extra": "mean: 4.774256924621849 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 976.1153777470053,
            "unit": "iter/sec",
            "range": "stddev: 0.00003517509998146452",
            "extra": "mean: 1.0244690564225343 msec\nrounds: 833"
          }
        ]
      }
    ]
  }
}