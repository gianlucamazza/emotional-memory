window.BENCHMARK_DATA = {
  "lastUpdate": 1783458362042,
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
          "id": "973dd586fa4d7270ab9e161093463408bb3a9e19",
          "message": "fix(core): correct PAD similarity normaliser + harden appraisal JSON parsing (#115)\n\n* fix(core): correct PAD similarity normaliser + harden appraisal JSON parsing\n\nTwo latent correctness bugs, both silent:\n\nB1 — resonance `_emotional_similarity` normalised `CoreAffect.distance()` by a\nstale 2-D max (2.24 ~= sqrt(5)) that predated the dominance axis, while\nretrieval correctly uses the full 3-D max sqrt(6). This over-penalised emotional\nsimilarity and clamped every distance in (sqrt(5), sqrt(6)] to zero. Introduce a\nsingle source of truth `affect.MAX_PAD_DISTANCE` and use it in both modules.\nAll 127 fidelity benchmarks still pass (change is within psychological tolerance).\n\nB2 — `_extract_json` used `\\{[^{}]*\\}`, which stops at the first inner brace and\nreturns a nested sub-object (or fails) on any nested LLM payload, silently\ntriggering the neutral fallback — the same silent-corruption class as the\nAddendum X2 adapter bug. Parse the whole payload first, then fall back to a\nstring-aware balanced-brace scan. Also promote the parse/validation-error\nfallback log from debug to warning so a swallowed appraisal is never silent.\n\nAdds regression tests for both.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: changelog entry for PAD normaliser + JSON parser fixes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-07T22:59:24+02:00",
          "tree_id": "010ed03f35ac45f1c1446665f5f7c126ccd614d9",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/973dd586fa4d7270ab9e161093463408bb3a9e19"
        },
        "date": 1783458360394,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 500.6878932389817,
            "unit": "iter/sec",
            "range": "stddev: 0.0010073156954152386",
            "extra": "mean: 1.997252207419949 msec\nrounds: 1779"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 404.9719315166296,
            "unit": "iter/sec",
            "range": "stddev: 0.0013346136115409007",
            "extra": "mean: 2.4693069375326235 msec\nrounds: 1921"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 305.8603424113935,
            "unit": "iter/sec",
            "range": "stddev: 0.002150903358906092",
            "extra": "mean: 3.269466031836723 msec\nrounds: 2450"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 157.98631705543838,
            "unit": "iter/sec",
            "range": "stddev: 0.004081623717494545",
            "extra": "mean: 6.329662078577942 msec\nrounds: 4416"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 403.8624363835988,
            "unit": "iter/sec",
            "range": "stddev: 0.0012668657592152044",
            "extra": "mean: 2.4760906435234165 msec\nrounds: 1930"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 378.0900109349567,
            "unit": "iter/sec",
            "range": "stddev: 0.00031974003156982537",
            "extra": "mean: 2.6448728373626125 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 146270.81227010285,
            "unit": "iter/sec",
            "range": "stddev: 6.170608179865896e-7",
            "extra": "mean: 6.836633942754114 usec\nrounds: 35825"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.36998395748768,
            "unit": "iter/sec",
            "range": "stddev: 0.00012557394469113622",
            "extra": "mean: 29.967050666670048 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8360705605339095,
            "unit": "iter/sec",
            "range": "stddev: 0.014130751938759503",
            "extra": "mean: 1.196071297333333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02902832474730089,
            "unit": "iter/sec",
            "range": "stddev: 0.8643694657146204",
            "extra": "mean: 34.44911164200001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3566.756751661551,
            "unit": "iter/sec",
            "range": "stddev: 0.000006730399968088039",
            "extra": "mean: 280.366750419455 usec\nrounds: 2384"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 885.9810197979551,
            "unit": "iter/sec",
            "range": "stddev: 0.00004648198493246203",
            "extra": "mean: 1.1286923508000732 msec\nrounds: 687"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 365.83724666692177,
            "unit": "iter/sec",
            "range": "stddev: 0.00005855732413590537",
            "extra": "mean: 2.733455953735773 msec\nrounds: 281"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 497.1142556549767,
            "unit": "iter/sec",
            "range": "stddev: 0.00044687815440211143",
            "extra": "mean: 2.0116099842730164 msec\nrounds: 763"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12059.790831985747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025147262221810273",
            "extra": "mean: 82.92017779841888 usec\nrounds: 7441"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12435.40971989908,
            "unit": "iter/sec",
            "range": "stddev: 0.000018370355953805045",
            "extra": "mean: 80.41552490223181 usec\nrounds: 8453"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2160.8267061380006,
            "unit": "iter/sec",
            "range": "stddev: 0.00001017954654362973",
            "extra": "mean: 462.78583893813425 usec\nrounds: 1130"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2190.284010921969,
            "unit": "iter/sec",
            "range": "stddev: 0.000009655659928977753",
            "extra": "mean: 456.5617951888641 usec\nrounds: 1455"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1163.9537890841027,
            "unit": "iter/sec",
            "range": "stddev: 0.00001494277331680275",
            "extra": "mean: 859.1406371784609 usec\nrounds: 893"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 381.2748089711553,
            "unit": "iter/sec",
            "range": "stddev: 0.00010370232061970574",
            "extra": "mean: 2.6227801482569313 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.045173443750457,
            "unit": "iter/sec",
            "range": "stddev: 0.00027370972573276866",
            "extra": "mean: 47.516833380955504 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 474.85722790983965,
            "unit": "iter/sec",
            "range": "stddev: 0.0000643073685320917",
            "extra": "mean: 2.1058961330369983 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 385.1828317820723,
            "unit": "iter/sec",
            "range": "stddev: 0.00006396324959091263",
            "extra": "mean: 2.5961697082225546 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 316.73031301195374,
            "unit": "iter/sec",
            "range": "stddev: 0.00007837858848435924",
            "extra": "mean: 3.157260163987711 msec\nrounds: 311"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 205.5304567006161,
            "unit": "iter/sec",
            "range": "stddev: 0.00017229552856787658",
            "extra": "mean: 4.865458949748941 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 957.9962119748648,
            "unit": "iter/sec",
            "range": "stddev: 0.000013141626988099145",
            "extra": "mean: 1.0438454635833543 msec\nrounds: 865"
          }
        ]
      }
    ]
  }
}