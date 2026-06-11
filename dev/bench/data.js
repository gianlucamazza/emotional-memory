window.BENCHMARK_DATA = {
  "lastUpdate": 1781137363181,
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
          "id": "44a9c49f21f11f889f0f08cd36423c8ff2ec7ede",
          "message": "chore(deps): refresh uv.lock for Dependabot alerts (8/10 resolved) (#56)\n\nTargeted uv lock --upgrade-package for the open alerts: langchain-core\n1.4.4 (>=1.3.3), langsmith 0.8.14 (>=0.8.0), python-multipart 0.0.32\n(>=0.0.27), urllib3 2.7.0 (fixes both CVEs), idna 3.18 (>=3.15),\npymdown-extensions 10.21.3, starlette 1.2.1 (>=1.0.1).\n\nTwo alerts remain open pending an upstream release: chromadb\nCVE-2026-45829 (critical, vulnerable <=1.5.9, no patched version\npublished -- 1.5.9 is the latest on PyPI) and torch CVE-2025-3000 (low,\nvulnerable <=2.12.0, idem). Both are optional-extra/dev dependency\nchains, not runtime deps of the published wheel.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-06-11T02:14:59+02:00",
          "tree_id": "c104b15ada10a7dfeaecd17a3a073c248cb17579",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/44a9c49f21f11f889f0f08cd36423c8ff2ec7ede"
        },
        "date": 1781137362713,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 609.7687420315052,
            "unit": "iter/sec",
            "range": "stddev: 0.0007961457288310374",
            "extra": "mean: 1.6399659921372822 msec\nrounds: 1399"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 420.6722235701769,
            "unit": "iter/sec",
            "range": "stddev: 0.0012131563700596543",
            "extra": "mean: 2.3771476792861725 msec\nrounds: 1849"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 307.1284726402149,
            "unit": "iter/sec",
            "range": "stddev: 0.002027337442596967",
            "extra": "mean: 3.255966441025636 msec\nrounds: 2535"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 148.95432326303603,
            "unit": "iter/sec",
            "range": "stddev: 0.005144325983721528",
            "extra": "mean: 6.713467444876482 msec\nrounds: 4372"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 431.5610400634831,
            "unit": "iter/sec",
            "range": "stddev: 0.0011027534359672813",
            "extra": "mean: 2.317169315962578 msec\nrounds: 1823"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 366.9372923105725,
            "unit": "iter/sec",
            "range": "stddev: 0.0002715283198445676",
            "extra": "mean: 2.725261293838755 msec\nrounds: 422"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135277.34537293244,
            "unit": "iter/sec",
            "range": "stddev: 8.900012733786035e-7",
            "extra": "mean: 7.392220753912645 usec\nrounds: 33295"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.01776530767849,
            "unit": "iter/sec",
            "range": "stddev: 0.00042094141513672565",
            "extra": "mean: 30.286725666665387 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8363969713349738,
            "unit": "iter/sec",
            "range": "stddev: 0.005213060893208302",
            "extra": "mean: 1.195604520666663 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.022997122509001395,
            "unit": "iter/sec",
            "range": "stddev: 0.9833629443493979",
            "extra": "mean: 43.48370104166667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3529.8082335786494,
            "unit": "iter/sec",
            "range": "stddev: 0.000012502823589503184",
            "extra": "mean: 283.3015092681574 usec\nrounds: 2050"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 876.8312254797032,
            "unit": "iter/sec",
            "range": "stddev: 0.00007095211304232822",
            "extra": "mean: 1.1404703333334334 msec\nrounds: 36"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.3181773993658,
            "unit": "iter/sec",
            "range": "stddev: 0.00008162817515225365",
            "extra": "mean: 2.760005051851838 msec\nrounds: 270"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.2255205858962,
            "unit": "iter/sec",
            "range": "stddev: 0.0004413389483911932",
            "extra": "mean: 1.9951099034843092 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12528.732825483083,
            "unit": "iter/sec",
            "range": "stddev: 0.000003996365068562148",
            "extra": "mean: 79.81653164205312 usec\nrounds: 6700"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12421.463984415837,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036000554918176033",
            "extra": "mean: 80.50580843406345 usec\nrounds: 6687"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2080.784422447265,
            "unit": "iter/sec",
            "range": "stddev: 0.000012809714124734833",
            "extra": "mean: 480.58798845863805 usec\nrounds: 1213"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2028.9331504611962,
            "unit": "iter/sec",
            "range": "stddev: 0.00006386219408221684",
            "extra": "mean: 492.86986107585176 usec\nrounds: 1598"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1196.9402119581516,
            "unit": "iter/sec",
            "range": "stddev: 0.00001984488951967125",
            "extra": "mean: 835.4636179897705 usec\nrounds: 856"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 353.602735135455,
            "unit": "iter/sec",
            "range": "stddev: 0.0006333599141906613",
            "extra": "mean: 2.828032423496184 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 18.47281328154644,
            "unit": "iter/sec",
            "range": "stddev: 0.0006235394915882419",
            "extra": "mean: 54.13360622222917 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 449.97715983811037,
            "unit": "iter/sec",
            "range": "stddev: 0.00013211249813159767",
            "extra": "mean: 2.222335018870231 msec\nrounds: 424"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 366.4854131208335,
            "unit": "iter/sec",
            "range": "stddev: 0.0001553329838100077",
            "extra": "mean: 2.7286215609085946 msec\nrounds: 353"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 273.95447246881037,
            "unit": "iter/sec",
            "range": "stddev: 0.00045956998255909686",
            "extra": "mean: 3.650241556519394 msec\nrounds: 230"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 184.8297555892156,
            "unit": "iter/sec",
            "range": "stddev: 0.00043370895699424236",
            "extra": "mean: 5.410384257730131 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 964.7467409207243,
            "unit": "iter/sec",
            "range": "stddev: 0.000019607779384861276",
            "extra": "mean: 1.0365414648051894 msec\nrounds: 824"
          }
        ]
      }
    ]
  }
}