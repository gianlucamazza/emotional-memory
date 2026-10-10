window.BENCHMARK_DATA = {
  "lastUpdate": 1791620418862,
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
          "id": "6181b956a63e663d19464f07160fe716d89a4c65",
          "message": "docs: link the live Hugging Face Gradio demo (#147)\n\nPoint README Quickstart and the docs index at the public Space so\npeople can try the Gradio chat without a local install.\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>",
          "timestamp": "2026-10-10T10:13:08+02:00",
          "tree_id": "406683d252d18b4cf2d37238401f7a998270d3fa",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/6181b956a63e663d19464f07160fe716d89a4c65"
        },
        "date": 1791620417418,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 494.9665366518774,
            "unit": "iter/sec",
            "range": "stddev: 0.0008034433318063198",
            "extra": "mean: 2.0203386005937722 msec\nrounds: 1347"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 573.348266055218,
            "unit": "iter/sec",
            "range": "stddev: 0.001383104023424247",
            "extra": "mean: 1.7441406195927904 msec\nrounds: 786"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 462.93767927064965,
            "unit": "iter/sec",
            "range": "stddev: 0.0008930366888870615",
            "extra": "mean: 2.160117970037528 msec\nrounds: 1335"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 363.90201157490657,
            "unit": "iter/sec",
            "range": "stddev: 0.0013452079201380946",
            "extra": "mean: 2.7479925040044946 msec\nrounds: 1873"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 487.2124352504263,
            "unit": "iter/sec",
            "range": "stddev: 0.0011451098503995362",
            "extra": "mean: 2.0524927683465255 msec\nrounds: 1131"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 296.0464141630148,
            "unit": "iter/sec",
            "range": "stddev: 0.0006256680227812951",
            "extra": "mean: 3.3778487161454374 msec\nrounds: 384"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 128284.82039239706,
            "unit": "iter/sec",
            "range": "stddev: 0.000001333841924296355",
            "extra": "mean: 7.795154539260407 usec\nrounds: 29410"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.053748209251845,
            "unit": "iter/sec",
            "range": "stddev: 0.0009145831163563582",
            "extra": "mean: 99.46539133332995 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5143195001687987,
            "unit": "iter/sec",
            "range": "stddev: 0.0377695784942737",
            "extra": "mean: 1.9443167129999968 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.023140809492341947,
            "unit": "iter/sec",
            "range": "stddev: 0.6593542383617117",
            "extra": "mean: 43.213700036333336 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3143.4171159951597,
            "unit": "iter/sec",
            "range": "stddev: 0.000027779343083086344",
            "extra": "mean: 318.12513678555024 usec\nrounds: 2047"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.0410225421347,
            "unit": "iter/sec",
            "range": "stddev: 0.000020762669827351624",
            "extra": "mean: 1.160037601286106 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 349.5367689400932,
            "unit": "iter/sec",
            "range": "stddev: 0.000036805515506122765",
            "extra": "mean: 2.860929346667357 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 428.1287959504489,
            "unit": "iter/sec",
            "range": "stddev: 0.0004686582039632198",
            "extra": "mean: 2.335745713576666 msec\nrounds: 604"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 9629.093298733826,
            "unit": "iter/sec",
            "range": "stddev: 0.000008362935917094495",
            "extra": "mean: 103.85193797338059 usec\nrounds: 4756"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9598.70312889194,
            "unit": "iter/sec",
            "range": "stddev: 0.000006727332694161808",
            "extra": "mean: 104.18074052004133 usec\nrounds: 5037"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1732.181409423865,
            "unit": "iter/sec",
            "range": "stddev: 0.000021348817510390735",
            "extra": "mean: 577.3067385203069 usec\nrounds: 784"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1733.5711853831942,
            "unit": "iter/sec",
            "range": "stddev: 0.0000241225975780543",
            "extra": "mean: 576.8439210524584 usec\nrounds: 950"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 978.1370421685345,
            "unit": "iter/sec",
            "range": "stddev: 0.00005459811123723345",
            "extra": "mean: 1.022351630588486 msec\nrounds: 850"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 360.0332226202916,
            "unit": "iter/sec",
            "range": "stddev: 0.00009425291667218856",
            "extra": "mean: 2.7775214540538338 msec\nrounds: 925"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.693488375906277,
            "unit": "iter/sec",
            "range": "stddev: 0.0021130070139876366",
            "extra": "mean: 40.49650599287995 msec\nrounds: 281"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 460.31952986271364,
            "unit": "iter/sec",
            "range": "stddev: 0.00005379397448738948",
            "extra": "mean: 2.1724040261733872 msec\nrounds: 2407"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 368.462190494067,
            "unit": "iter/sec",
            "range": "stddev: 0.00016348474455856764",
            "extra": "mean: 2.7139826712182074 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 294.2835297137202,
            "unit": "iter/sec",
            "range": "stddev: 0.00005689920002381521",
            "extra": "mean: 3.3980834774300916 msec\nrounds: 576"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 192.12368466267176,
            "unit": "iter/sec",
            "range": "stddev: 0.00006209579226317738",
            "extra": "mean: 5.204980332101099 msec\nrounds: 271"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 840.6730550669669,
            "unit": "iter/sec",
            "range": "stddev: 0.000043388058454452086",
            "extra": "mean: 1.1895230779346693 msec\nrounds: 1065"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5067.168757541801,
            "unit": "iter/sec",
            "range": "stddev: 0.00000766932314098596",
            "extra": "mean: 197.34886439525707 usec\nrounds: 4373"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 75.9400045927813,
            "unit": "iter/sec",
            "range": "stddev: 0.00007644741043515168",
            "extra": "mean: 13.168289959453832 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1724.410475262809,
            "unit": "iter/sec",
            "range": "stddev: 0.00002317895683253055",
            "extra": "mean: 579.9083306122892 usec\nrounds: 1470"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 27.230470478528567,
            "unit": "iter/sec",
            "range": "stddev: 0.006358215606601816",
            "extra": "mean: 36.72356674074022 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17216.364834371034,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058512824776167485",
            "extra": "mean: 58.084270960823474 usec\nrounds: 10603"
          }
        ]
      }
    ]
  }
}