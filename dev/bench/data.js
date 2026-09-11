window.BENCHMARK_DATA = {
  "lastUpdate": 1789119548259,
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
            "email": "info@gianlucamazza.it",
            "name": "Gianluca Mazza",
            "username": "gianlucamazza"
          },
          "distinct": true,
          "id": "facdac0c9a59ba85c98066f2d43dc92e12210261",
          "message": "docs: align claim matrix and API docs with retrieve_query_gated\n\nRecord that retrieve_query_gated() shipped in v0.17.0 (Addendum Y).\nSync evidence pages, paper Addenda Y/Z, missing API symbols, and\ndedupe state-store docs onto the canonical page.",
          "timestamp": "2026-09-11T11:32:15+02:00",
          "tree_id": "bc2bd4f59b7f3d395f8a03935ee6bb2be870b98a",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/facdac0c9a59ba85c98066f2d43dc92e12210261"
        },
        "date": 1789119546686,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 562.1622132586735,
            "unit": "iter/sec",
            "range": "stddev: 0.000826122503202742",
            "extra": "mean: 1.778845992161803 msec\nrounds: 1531"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 436.9720335958073,
            "unit": "iter/sec",
            "range": "stddev: 0.0010390939076353313",
            "extra": "mean: 2.288475973556205 msec\nrounds: 1853"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 383.2257057595824,
            "unit": "iter/sec",
            "range": "stddev: 0.0013910667424462601",
            "extra": "mean: 2.609428295051148 msec\nrounds: 2203"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 223.26778738789514,
            "unit": "iter/sec",
            "range": "stddev: 0.0033299994343766407",
            "extra": "mean: 4.4789264573247465 msec\nrounds: 3140"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 446.70180217664296,
            "unit": "iter/sec",
            "range": "stddev: 0.0011040275161927334",
            "extra": "mean: 2.2386298759649104 msec\nrounds: 1685"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 351.09252744091583,
            "unit": "iter/sec",
            "range": "stddev: 0.0008463198866791727",
            "extra": "mean: 2.84825201860295 msec\nrounds: 430"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134590.14469677358,
            "unit": "iter/sec",
            "range": "stddev: 0.000001452688914733663",
            "extra": "mean: 7.4299645211984995 usec\nrounds: 43322"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.34162144763976,
            "unit": "iter/sec",
            "range": "stddev: 0.0014606314392502178",
            "extra": "mean: 36.57427566668048 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.795957730575558,
            "unit": "iter/sec",
            "range": "stddev: 0.00684495681396767",
            "extra": "mean: 1.2563481220000199 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.032694213723995266,
            "unit": "iter/sec",
            "range": "stddev: 2.367696443794597",
            "extra": "mean: 30.586452038333316 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3210.7492554005976,
            "unit": "iter/sec",
            "range": "stddev: 0.000012617191032937876",
            "extra": "mean: 311.4537824210232 usec\nrounds: 2537"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 877.0755534351403,
            "unit": "iter/sec",
            "range": "stddev: 0.000019206265597796674",
            "extra": "mean: 1.140152631188289 msec\nrounds: 808"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 359.0133475899348,
            "unit": "iter/sec",
            "range": "stddev: 0.00002908090207041443",
            "extra": "mean: 2.785411758958334 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 480.758169455178,
            "unit": "iter/sec",
            "range": "stddev: 0.0010842001046748163",
            "extra": "mean: 2.080047856770184 msec\nrounds: 768"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10475.292923664902,
            "unit": "iter/sec",
            "range": "stddev: 0.000005992082422018211",
            "extra": "mean: 95.46272426815713 usec\nrounds: 6626"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10433.498199981266,
            "unit": "iter/sec",
            "range": "stddev: 0.000006253462638866391",
            "extra": "mean: 95.84513083078842 usec\nrounds: 8339"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1840.9264158421831,
            "unit": "iter/sec",
            "range": "stddev: 0.00001674266955066699",
            "extra": "mean: 543.2047644025588 usec\nrounds: 1163"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1863.885909260432,
            "unit": "iter/sec",
            "range": "stddev: 0.000013572183913855482",
            "extra": "mean: 536.5135253352433 usec\nrounds: 1717"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1011.8150552182375,
            "unit": "iter/sec",
            "range": "stddev: 0.000028137667691551877",
            "extra": "mean: 988.3229102420411 usec\nrounds: 947"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 374.08261939417906,
            "unit": "iter/sec",
            "range": "stddev: 0.00004083393116628675",
            "extra": "mean: 2.673206260209267 msec\nrounds: 1053"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 25.774704435475826,
            "unit": "iter/sec",
            "range": "stddev: 0.001370788827129853",
            "extra": "mean: 38.79772908757854 msec\nrounds: 628"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 464.86341208435556,
            "unit": "iter/sec",
            "range": "stddev: 0.000032142137842346774",
            "extra": "mean: 2.1511695134624556 msec\nrounds: 2526"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 373.6193847855253,
            "unit": "iter/sec",
            "range": "stddev: 0.00003957908816172679",
            "extra": "mean: 2.6765206536969326 msec\nrounds: 1028"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.8891337552177,
            "unit": "iter/sec",
            "range": "stddev: 0.0000629861262578015",
            "extra": "mean: 3.3015380499194733 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.46751640282594,
            "unit": "iter/sec",
            "range": "stddev: 0.00006253648420065471",
            "extra": "mean: 5.115939560715381 msec\nrounds: 280"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 851.2247727701748,
            "unit": "iter/sec",
            "range": "stddev: 0.000023693277600225527",
            "extra": "mean: 1.1747778401063917 msec\nrounds: 1132"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5202.129980911917,
            "unit": "iter/sec",
            "range": "stddev: 0.000009621489392477638",
            "extra": "mean: 192.22895307677473 usec\nrounds: 4582"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 77.77717496542326,
            "unit": "iter/sec",
            "range": "stddev: 0.00010168825907949449",
            "extra": "mean: 12.857242506488074 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1864.6249079751524,
            "unit": "iter/sec",
            "range": "stddev: 0.000019973747331029722",
            "extra": "mean: 536.3008912532052 usec\nrounds: 1600"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 28.234193360566476,
            "unit": "iter/sec",
            "range": "stddev: 0.0045827946215096045",
            "extra": "mean: 35.41804744443871 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17476.503304717484,
            "unit": "iter/sec",
            "range": "stddev: 0.000007480817725599538",
            "extra": "mean: 57.219684199073576 usec\nrounds: 13651"
          }
        ]
      }
    ]
  }
}