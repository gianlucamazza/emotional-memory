window.BENCHMARK_DATA = {
  "lastUpdate": 1780176322190,
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
          "id": "7b668dee1ec820d664f068b79da5054876b02807",
          "message": "ci(release): create GitHub release automatically on tag (#43)\n\nAfter Publish/Verify PyPI, add an on-tag step that extracts the matching\nCHANGELOG.md section as release notes and runs 'gh release create' (idempotent —\nuploads assets via --clobber if the release already exists), attaching wheel +\nsdist. Raises the job 'contents' permission to write. Previously the GitHub\nrelease was created by hand after every tag; now the on-tag workflow does\nPyPI + GitHub release end-to-end.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-30T23:19:59+02:00",
          "tree_id": "18c2b7c177ced04961de45d319997cb4d72839d6",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7b668dee1ec820d664f068b79da5054876b02807"
        },
        "date": 1780176321527,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 546.7596962042842,
            "unit": "iter/sec",
            "range": "stddev: 0.0008516467226869836",
            "extra": "mean: 1.8289570481185815 msec\nrounds: 1621"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 419.8036949769107,
            "unit": "iter/sec",
            "range": "stddev: 0.0011834438714098533",
            "extra": "mean: 2.3820657415962003 msec\nrounds: 1904"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 383.3272303112425,
            "unit": "iter/sec",
            "range": "stddev: 0.001409180654687474",
            "extra": "mean: 2.6087371856887134 msec\nrounds: 2208"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 218.68439974636468,
            "unit": "iter/sec",
            "range": "stddev: 0.002791624748730128",
            "extra": "mean: 4.572799894093149 msec\nrounds: 3843"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 457.32730392397735,
            "unit": "iter/sec",
            "range": "stddev: 0.0009507412421739541",
            "extra": "mean: 2.18661774930069 msec\nrounds: 1787"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.153930860272,
            "unit": "iter/sec",
            "range": "stddev: 0.00030058910008721773",
            "extra": "mean: 2.630513375823954 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 140700.99220886058,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011569494158942818",
            "extra": "mean: 7.107270420066202 usec\nrounds: 47282"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.51324471581509,
            "unit": "iter/sec",
            "range": "stddev: 0.00013631813019343073",
            "extra": "mean: 35.07142066666802 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8241667384167957,
            "unit": "iter/sec",
            "range": "stddev: 0.004045013120776617",
            "extra": "mean: 1.213346709333327 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03718024257908746,
            "unit": "iter/sec",
            "range": "stddev: 1.215953046960071",
            "extra": "mean: 26.896005260666687 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3103.338474750003,
            "unit": "iter/sec",
            "range": "stddev: 0.00003841285644565796",
            "extra": "mean: 322.2336229632694 usec\nrounds: 2517"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 851.5287830784823,
            "unit": "iter/sec",
            "range": "stddev: 0.00002815895081053277",
            "extra": "mean: 1.174358424367945 msec\nrounds: 714"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 350.6257999526837,
            "unit": "iter/sec",
            "range": "stddev: 0.00004521356296466205",
            "extra": "mean: 2.8520434039222105 msec\nrounds: 255"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 499.29160694223316,
            "unit": "iter/sec",
            "range": "stddev: 0.0004249880032015686",
            "extra": "mean: 2.002837592492712 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9651.746855928694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053039391537576525",
            "extra": "mean: 103.60818771223147 usec\nrounds: 6771"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9757.066336209578,
            "unit": "iter/sec",
            "range": "stddev: 0.000011188482676793439",
            "extra": "mean: 102.48982281578702 usec\nrounds: 6592"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1735.993945936502,
            "unit": "iter/sec",
            "range": "stddev: 0.000022657539082645646",
            "extra": "mean: 576.0388752165483 usec\nrounds: 1154"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1700.9969361423982,
            "unit": "iter/sec",
            "range": "stddev: 0.000012185520782424686",
            "extra": "mean: 587.8905356924672 usec\nrounds: 1667"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1013.2015342861755,
            "unit": "iter/sec",
            "range": "stddev: 0.000021432530205556608",
            "extra": "mean: 986.9704754291787 usec\nrounds: 814"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 374.17515602301347,
            "unit": "iter/sec",
            "range": "stddev: 0.00003662323547598238",
            "extra": "mean: 2.6725451540628087 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.329797647659948,
            "unit": "iter/sec",
            "range": "stddev: 0.0009708865285994255",
            "extra": "mean: 41.10186260000319 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.91504622297356,
            "unit": "iter/sec",
            "range": "stddev: 0.00003315547470036695",
            "extra": "mean: 2.123525268560882 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 379.4660484819088,
            "unit": "iter/sec",
            "range": "stddev: 0.000036443455101422866",
            "extra": "mean: 2.6352818756792558 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.99902347750236,
            "unit": "iter/sec",
            "range": "stddev: 0.00003878228800387776",
            "extra": "mean: 3.2573393513523095 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 197.82938886279865,
            "unit": "iter/sec",
            "range": "stddev: 0.000050834124666568764",
            "extra": "mean: 5.054860684493818 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 860.5212466318164,
            "unit": "iter/sec",
            "range": "stddev: 0.000021610493768250847",
            "extra": "mean: 1.162086356280127 msec\nrounds: 828"
          }
        ]
      }
    ]
  }
}