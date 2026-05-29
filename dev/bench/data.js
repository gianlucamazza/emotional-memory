window.BENCHMARK_DATA = {
  "lastUpdate": 1780045048474,
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
          "id": "1b52d88f733b8b1b5e8a0ea34bef98e3f21f1678",
          "message": "chore(mypy): drop redundant bare langchain_core override entry (#38)\n\nmypy --warn-unused-configs reported `module = ['langchain_core']` as an unused\nsection: only `langchain_core.*` is ever imported (via langchain_core.messages\nin integrations/langchain.py), so the bare module entry never matched. Keep the\n`.*` glob, which still silences the missing-import for the optional langchain extra.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-29T10:51:56+02:00",
          "tree_id": "752844ee36045379e46d96aeecb8f173b013a97f",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1b52d88f733b8b1b5e8a0ea34bef98e3f21f1678"
        },
        "date": 1780045047706,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 527.0188257522793,
            "unit": "iter/sec",
            "range": "stddev: 0.0008903538443387878",
            "extra": "mean: 1.8974654246412848 msec\nrounds: 1672"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 417.05300942581175,
            "unit": "iter/sec",
            "range": "stddev: 0.0011967302182305103",
            "extra": "mean: 2.397776727176181 msec\nrounds: 1884"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 356.42934654387335,
            "unit": "iter/sec",
            "range": "stddev: 0.0015602460379897834",
            "extra": "mean: 2.8056051211734574 msec\nrounds: 2352"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 251.47555242971208,
            "unit": "iter/sec",
            "range": "stddev: 0.0024429239137241307",
            "extra": "mean: 3.9765296878292054 msec\nrounds: 3418"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 445.3357009377437,
            "unit": "iter/sec",
            "range": "stddev: 0.0009911760563924919",
            "extra": "mean: 2.2454970439026094 msec\nrounds: 1845"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 379.7276534318526,
            "unit": "iter/sec",
            "range": "stddev: 0.0002333235220307337",
            "extra": "mean: 2.6334663566435883 msec\nrounds: 429"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136249.2708870591,
            "unit": "iter/sec",
            "range": "stddev: 0.000001627453776698645",
            "extra": "mean: 7.33948881700019 usec\nrounds: 47304"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.263244537940786,
            "unit": "iter/sec",
            "range": "stddev: 0.0003439891888679103",
            "extra": "mean: 35.381642000004376 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8134663466358462,
            "unit": "iter/sec",
            "range": "stddev: 0.004535223291443368",
            "extra": "mean: 1.2293071546666663 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03608153821466672,
            "unit": "iter/sec",
            "range": "stddev: 1.7905431959054692",
            "extra": "mean: 27.715004666666673 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3251.168844504656,
            "unit": "iter/sec",
            "range": "stddev: 0.000014782980496573993",
            "extra": "mean: 307.5816876414362 usec\nrounds: 2209"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 883.1133353498805,
            "unit": "iter/sec",
            "range": "stddev: 0.000019572133097806216",
            "extra": "mean: 1.1323574902238458 msec\nrounds: 716"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.6034492531633,
            "unit": "iter/sec",
            "range": "stddev: 0.00009720567867161877",
            "extra": "mean: 2.7578336666671297 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 489.6967658658254,
            "unit": "iter/sec",
            "range": "stddev: 0.0007757044943895222",
            "extra": "mean: 2.0420800579148506 msec\nrounds: 777"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10366.410403782807,
            "unit": "iter/sec",
            "range": "stddev: 0.000014216178940450049",
            "extra": "mean: 96.46540712251657 usec\nrounds: 7020"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10437.618828424103,
            "unit": "iter/sec",
            "range": "stddev: 0.000005059510706854318",
            "extra": "mean: 95.80729249057876 usec\nrounds: 8243"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1811.1564449610478,
            "unit": "iter/sec",
            "range": "stddev: 0.000011662605234253749",
            "extra": "mean: 552.1334188342337 usec\nrounds: 1115"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1798.351642243051,
            "unit": "iter/sec",
            "range": "stddev: 0.000013269084101596468",
            "extra": "mean: 556.0647742689068 usec\nrounds: 1710"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1014.1796573400832,
            "unit": "iter/sec",
            "range": "stddev: 0.000050391640803290834",
            "extra": "mean: 986.0185942032475 usec\nrounds: 828"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 376.29852262289654,
            "unit": "iter/sec",
            "range": "stddev: 0.00004345536968307222",
            "extra": "mean: 2.657464592286319 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.754283623573226,
            "unit": "iter/sec",
            "range": "stddev: 0.0012920406999476627",
            "extra": "mean: 42.09767029166992 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.6256600160457,
            "unit": "iter/sec",
            "range": "stddev: 0.000036267472091778224",
            "extra": "mean: 2.1248310174288108 msec\nrounds: 459"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 378.67328027446877,
            "unit": "iter/sec",
            "range": "stddev: 0.000051450437097517594",
            "extra": "mean: 2.640798947512703 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 305.2490639382542,
            "unit": "iter/sec",
            "range": "stddev: 0.00006672636715624989",
            "extra": "mean: 3.2760133220335774 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.3554842783943,
            "unit": "iter/sec",
            "range": "stddev: 0.00008341862489071939",
            "extra": "mean: 5.09280402161924 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 853.192384021394,
            "unit": "iter/sec",
            "range": "stddev: 0.000027372048364791692",
            "extra": "mean: 1.1720685964010253 msec\nrounds: 778"
          }
        ]
      }
    ]
  }
}