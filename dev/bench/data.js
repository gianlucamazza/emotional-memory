window.BENCHMARK_DATA = {
  "lastUpdate": 1782982457820,
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
          "id": "16cb4cf5cef4f7e91b220d3bcdac72130998bd18",
          "message": "chore(meta): pin codemeta relatedLink to concept DOI, teach sync to manage it (#97)\n\n- codemeta.json relatedLink carried the v0.10.0 version DOI (20070143) for\n  8 releases; now the concept DOI (19972258)\n- sync_release_metadata: relatedLink Zenodo entries are pinned to the\n  concept DOI on every sync (non-DOI links untouched)\n- regression test: zenodo links in relatedLink must equal the README\n  concept DOI; idempotence test still green\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T10:48:07+02:00",
          "tree_id": "e6c0cdc191f4db74871d9ccfd4a83c68d78117dd",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/16cb4cf5cef4f7e91b220d3bcdac72130998bd18"
        },
        "date": 1782982456489,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 411.6499381971396,
            "unit": "iter/sec",
            "range": "stddev: 0.0013563172342904144",
            "extra": "mean: 2.4292485124122596 msec\nrounds: 2135"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 466.0563448648391,
            "unit": "iter/sec",
            "range": "stddev: 0.0009309218887162417",
            "extra": "mean: 2.145663310924369 msec\nrounds: 1785"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 393.8852760199771,
            "unit": "iter/sec",
            "range": "stddev: 0.0014150527020347463",
            "extra": "mean: 2.5388103107192106 msec\nrounds: 2211"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 230.41275219753942,
            "unit": "iter/sec",
            "range": "stddev: 0.0025882708441624636",
            "extra": "mean: 4.340037565033169 msec\nrounds: 3775"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 464.05834165834517,
            "unit": "iter/sec",
            "range": "stddev: 0.000952242758459714",
            "extra": "mean: 2.15490146438577 msec\nrounds: 1783"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 386.89981140757806,
            "unit": "iter/sec",
            "range": "stddev: 0.00024081012671061647",
            "extra": "mean: 2.584648455531435 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 139921.5236815186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011354015552604798",
            "extra": "mean: 7.146863282279166 usec\nrounds: 52122"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.252364395638924,
            "unit": "iter/sec",
            "range": "stddev: 0.00011029162406064529",
            "extra": "mean: 35.395267666672225 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8320698620589358,
            "unit": "iter/sec",
            "range": "stddev: 0.0048671182930604",
            "extra": "mean: 1.2018221613333349 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03903090488144716,
            "unit": "iter/sec",
            "range": "stddev: 1.2257834773876692",
            "extra": "mean: 25.62072293833334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3216.2285279558837,
            "unit": "iter/sec",
            "range": "stddev: 0.000015949815069008366",
            "extra": "mean: 310.92317952778166 usec\nrounds: 2501"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 880.6557575006144,
            "unit": "iter/sec",
            "range": "stddev: 0.000011530615891699616",
            "extra": "mean: 1.1355174726139259 msec\nrounds: 639"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 363.3865693226587,
            "unit": "iter/sec",
            "range": "stddev: 0.0000645965428632331",
            "extra": "mean: 2.7518903680561695 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 499.2038223007665,
            "unit": "iter/sec",
            "range": "stddev: 0.0004235831807914396",
            "extra": "mean: 2.003189790076382 msec\nrounds: 786"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10487.724698935339,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048922281051021015",
            "extra": "mean: 95.34956615532775 usec\nrounds: 7694"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10461.142390251736,
            "unit": "iter/sec",
            "range": "stddev: 0.000004997660071392526",
            "extra": "mean: 95.59185437833774 usec\nrounds: 8165"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1861.724728658112,
            "unit": "iter/sec",
            "range": "stddev: 0.000008192682277912758",
            "extra": "mean: 537.1363363265722 usec\nrounds: 1225"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1858.0580248871474,
            "unit": "iter/sec",
            "range": "stddev: 0.000008164916314702189",
            "extra": "mean: 538.1963246603867 usec\nrounds: 1768"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1008.1710148629425,
            "unit": "iter/sec",
            "range": "stddev: 0.000015734644443098903",
            "extra": "mean: 991.8952095006886 usec\nrounds: 821"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 376.9585376347453,
            "unit": "iter/sec",
            "range": "stddev: 0.000028733019399478295",
            "extra": "mean: 2.652811649457723 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.454666625058163,
            "unit": "iter/sec",
            "range": "stddev: 0.0008742670703838826",
            "extra": "mean: 42.635438652179964 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 475.90642086648415,
            "unit": "iter/sec",
            "range": "stddev: 0.000025135766833813085",
            "extra": "mean: 2.101253431671078 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 381.9754291601999,
            "unit": "iter/sec",
            "range": "stddev: 0.000024714162662539035",
            "extra": "mean: 2.617969439025361 msec\nrounds: 369"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 307.8235628966113,
            "unit": "iter/sec",
            "range": "stddev: 0.00005442251692089748",
            "extra": "mean: 3.2486142080548595 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 197.08560020254524,
            "unit": "iter/sec",
            "range": "stddev: 0.000055247571951145816",
            "extra": "mean: 5.0739374108118405 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 856.1290349833168,
            "unit": "iter/sec",
            "range": "stddev: 0.00001704120381437272",
            "extra": "mean: 1.168048225370007 msec\nrounds: 812"
          }
        ]
      }
    ]
  }
}