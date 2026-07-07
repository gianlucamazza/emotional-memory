window.BENCHMARK_DATA = {
  "lastUpdate": 1783425839761,
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
          "id": "bc37703baa84d22f2b6c10497aaf3bad3ecad51f",
          "message": "feat(bench): addendum y query-affect gate harness (4-corpus, harness-only) (#111)\n\nHarness for the pre-registered Hy study (#110):\n\n- benchmarks/common/gate.py: gated_scores (exact per-query selection between the\n  cosine and aft arms on abs(valence)<tau), routing_rate, recovery_fraction,\n  gate_analysis, gate_report (paired one-tailed bootstrap contrasts + tau\n  sensitivity, reuses common/statistics). Pure, unit-tested.\n- benchmarks/gate/runner.py: orchestrates the 4 corpora, Holm m=2 verdict\n  (Hg1 esmemeval gated>aft, Hg2 curated gated>cosine); --dry-run smoke over the\n  two no-LLM corpora (esmemeval + madial) writes results.dry.* (gitignored)\n- per-query gate export (_per_query_gate = aligned cosine/aft/valence) + a\n  gate_inputs() added to each of the 4 runners; additive, no existing key or\n  code path changed, scored artifacts untouched\n- tests/test_gate.py (9 tests: exact composition incl. tau=0 -> aft, tau>1 ->\n  cosine, boundary strict-<, routing, recovery, misalignment guard)\n- Makefile: bench-y-gate[-dry]; .gitignore for gate dry artifacts\n\nDry-run exercised end to end; construction sanity holds (100% routing ->\ngated==cosine exactly; partial routing -> gated between arms). Zero src/ change.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T13:57:10+02:00",
          "tree_id": "1b2daed3d71ba8a9b89ed2c78671a71798df48a3",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/bc37703baa84d22f2b6c10497aaf3bad3ecad51f"
        },
        "date": 1783425838324,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 522.1830439555081,
            "unit": "iter/sec",
            "range": "stddev: 0.0008922918833914721",
            "extra": "mean: 1.9150372873562778 msec\nrounds: 1653"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 451.0241893979801,
            "unit": "iter/sec",
            "range": "stddev: 0.0010382124509365847",
            "extra": "mean: 2.217175981924127 msec\nrounds: 1715"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 368.0241660089424,
            "unit": "iter/sec",
            "range": "stddev: 0.0015622582652554547",
            "extra": "mean: 2.7172128690475765 msec\nrounds: 2184"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 211.06490152457715,
            "unit": "iter/sec",
            "range": "stddev: 0.003344717710540651",
            "extra": "mean: 4.737879167861343 msec\nrounds: 3348"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 470.90727291595437,
            "unit": "iter/sec",
            "range": "stddev: 0.000963672917288596",
            "extra": "mean: 2.1235603217758667 msec\nrounds: 1644"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 376.7531066137455,
            "unit": "iter/sec",
            "range": "stddev: 0.00027181760322405403",
            "extra": "mean: 2.654258140000473 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136275.31786260044,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012928157122554162",
            "extra": "mean: 7.33808598420038 usec\nrounds: 39798"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.675153695351515,
            "unit": "iter/sec",
            "range": "stddev: 0.0009427467731016959",
            "extra": "mean: 36.13349399999777 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8111736053557388,
            "unit": "iter/sec",
            "range": "stddev: 0.0027139720971622867",
            "extra": "mean: 1.23278172933333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.024952777579377246,
            "unit": "iter/sec",
            "range": "stddev: 1.2869543690558791",
            "extra": "mean: 40.07569886033334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3110.236027222976,
            "unit": "iter/sec",
            "range": "stddev: 0.000014799940403747599",
            "extra": "mean: 321.51900731883234 usec\nrounds: 2186"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 857.6288898611408,
            "unit": "iter/sec",
            "range": "stddev: 0.00002397221985811607",
            "extra": "mean: 1.1660054970418623 msec\nrounds: 676"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 348.1928869428679,
            "unit": "iter/sec",
            "range": "stddev: 0.00017674367105842935",
            "extra": "mean: 2.8719713627121903 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 491.1973257323787,
            "unit": "iter/sec",
            "range": "stddev: 0.00047920148187063287",
            "extra": "mean: 2.035841702739307 msec\nrounds: 730"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10709.441446103376,
            "unit": "iter/sec",
            "range": "stddev: 0.000005198174680102375",
            "extra": "mean: 93.37555137983871 usec\nrounds: 5907"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10485.787392795122,
            "unit": "iter/sec",
            "range": "stddev: 0.00000584014478774024",
            "extra": "mean: 95.36718250525551 usec\nrounds: 8071"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1823.7211354934593,
            "unit": "iter/sec",
            "range": "stddev: 0.000017412339968491776",
            "extra": "mean: 548.3294460638149 usec\nrounds: 1029"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1812.3309895623677,
            "unit": "iter/sec",
            "range": "stddev: 0.000013357164401144358",
            "extra": "mean: 551.7755894255689 usec\nrounds: 1532"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1004.586991169565,
            "unit": "iter/sec",
            "range": "stddev: 0.000020207272348546227",
            "extra": "mean: 995.4339532465729 usec\nrounds: 770"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 365.3733245057956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000537297860380016",
            "extra": "mean: 2.736926679999426 msec\nrounds: 350"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.964834423428524,
            "unit": "iter/sec",
            "range": "stddev: 0.000820269478800511",
            "extra": "mean: 47.69892190908432 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 455.9368758169301,
            "unit": "iter/sec",
            "range": "stddev: 0.00007354263399003324",
            "extra": "mean: 2.193286073227217 msec\nrounds: 437"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 366.0050085311248,
            "unit": "iter/sec",
            "range": "stddev: 0.000051130208580801495",
            "extra": "mean: 2.7322030482950637 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 295.5440169446415,
            "unit": "iter/sec",
            "range": "stddev: 0.00010659115641873821",
            "extra": "mean: 3.383590743396137 msec\nrounds: 265"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 182.4659375739624,
            "unit": "iter/sec",
            "range": "stddev: 0.0002169890474860459",
            "extra": "mean: 5.480474949438992 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 840.2449980978556,
            "unit": "iter/sec",
            "range": "stddev: 0.000026961299973967895",
            "extra": "mean: 1.1901290721917979 msec\nrounds: 748"
          }
        ]
      }
    ]
  }
}