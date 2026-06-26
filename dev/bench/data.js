window.BENCHMARK_DATA = {
  "lastUpdate": 1782511899700,
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
          "id": "6cd499a7852c75acdb258d0a675eb955f9b283fa",
          "message": "docs(paper): fold Addenda R (downstream PASS) and S (human-gold) into the paper (#68)\n\narXiv (#31) is not yet submitted, so the v1 should carry the two June 2026 results.\n\n- Abstract: add downstream-conversion clause (Addendum R: judge-acc 0.595 vs 0.440,\n  Δ+0.155, p<0.001) and human-gold clause (Addendum S: EmoBank valence r=0.70).\n  Condensed the negative-results enumeration to keep the abstract within arXiv's\n  1920-char hard limit (now 1909).\n- Limitations: \"No human evaluation\" → \"signal yes, perceived utility not yet\" —\n  reports Addendum S (appraisal valence human-validated vs EmoBank; arousal/dominance\n  weak; +0.15 bias persists; keyword engine not validated) while keeping Gate 2\n  (human perceived utility) explicitly open.\n- New \"Downstream value (oracle-affect regime)\" paragraph: Addendum R, scoped to the\n  state-injection boundary; explicitly does not contradict the LoCoMo oracle-free FAIL.\n- Conclusion: addenda range A--P → A--S.\n- refs.bib: add buechel2017emobank (EmoBank, EACL 2017).\n- Regenerated paper/tables (no diff) and paper/arxiv-submission.tar.gz (bundle matches).\n\nRefs #61 #62 #60; supports #31\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T00:06:06+02:00",
          "tree_id": "f276f0550627ac5434d5caa7c5fa5d635ea0db4e",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/6cd499a7852c75acdb258d0a675eb955f9b283fa"
        },
        "date": 1782511898271,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 494.21647576094676,
            "unit": "iter/sec",
            "range": "stddev: 0.0009934315479480998",
            "extra": "mean: 2.0234048216630107 msec\nrounds: 1828"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 427.3720510075167,
            "unit": "iter/sec",
            "range": "stddev: 0.0011519152969626",
            "extra": "mean: 2.339881603494029 msec\nrounds: 1889"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 385.84675340826857,
            "unit": "iter/sec",
            "range": "stddev: 0.0014087010351201692",
            "extra": "mean: 2.5917025118567976 msec\nrounds: 2235"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 229.7588750799945,
            "unit": "iter/sec",
            "range": "stddev: 0.002625114137214813",
            "extra": "mean: 4.352388997603827 msec\nrounds: 3756"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 445.7467869494822,
            "unit": "iter/sec",
            "range": "stddev: 0.000992987040603491",
            "extra": "mean: 2.2434261542155163 msec\nrounds: 1874"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 385.7006338566112,
            "unit": "iter/sec",
            "range": "stddev: 0.00027538443304997846",
            "extra": "mean: 2.5926843572981055 msec\nrounds: 459"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 140447.2765746303,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010448126541951019",
            "extra": "mean: 7.120109584101648 usec\nrounds: 52535"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.359116559825335,
            "unit": "iter/sec",
            "range": "stddev: 0.00013671341351522318",
            "extra": "mean: 35.262029333334034 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8309869861787295,
            "unit": "iter/sec",
            "range": "stddev: 0.0023703085080383755",
            "extra": "mean: 1.203388279999994 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.038224278545522794,
            "unit": "iter/sec",
            "range": "stddev: 1.3052359392274275",
            "extra": "mean: 26.16138323733333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3228.9506424048377,
            "unit": "iter/sec",
            "range": "stddev: 0.000011413427061834664",
            "extra": "mean: 309.6981374900256 usec\nrounds: 2582"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 876.116923405452,
            "unit": "iter/sec",
            "range": "stddev: 0.000013755995366594897",
            "extra": "mean: 1.141400163933618 msec\nrounds: 732"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 360.13799623684844,
            "unit": "iter/sec",
            "range": "stddev: 0.00003787878962505752",
            "extra": "mean: 2.7767133999999816 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.2895809085471,
            "unit": "iter/sec",
            "range": "stddev: 0.00042538932965351415",
            "extra": "mean: 1.9948549462918823 msec\nrounds: 782"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10376.98742136017,
            "unit": "iter/sec",
            "range": "stddev: 0.000005034015414770328",
            "extra": "mean: 96.36708221709729 usec\nrounds: 6495"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 9860.15427259801,
            "unit": "iter/sec",
            "range": "stddev: 0.000010051563699355205",
            "extra": "mean: 101.41829147430917 usec\nrounds: 8011"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1814.9592778066528,
            "unit": "iter/sec",
            "range": "stddev: 0.000011559473521015825",
            "extra": "mean: 550.9765492967329 usec\nrounds: 1136"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1806.006904517157,
            "unit": "iter/sec",
            "range": "stddev: 0.000010062705626967197",
            "extra": "mean: 553.7077391558223 usec\nrounds: 1706"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 995.3868801226531,
            "unit": "iter/sec",
            "range": "stddev: 0.00001864131769937702",
            "extra": "mean: 1.0046344993785516 msec\nrounds: 805"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.5570566912988,
            "unit": "iter/sec",
            "range": "stddev: 0.00003244428553430356",
            "extra": "mean: 2.6913766862752158 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.94201990536048,
            "unit": "iter/sec",
            "range": "stddev: 0.00047582513495784105",
            "extra": "mean: 41.76757032000069 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 474.23593323566496,
            "unit": "iter/sec",
            "range": "stddev: 0.00002343313474786409",
            "extra": "mean: 2.1086550594702906 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.6503278406103,
            "unit": "iter/sec",
            "range": "stddev: 0.000029739733174004853",
            "extra": "mean: 2.627082986301091 msec\nrounds: 365"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 308.060931096702,
            "unit": "iter/sec",
            "range": "stddev: 0.000029078482957527433",
            "extra": "mean: 3.246111074325405 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 196.39506885118936,
            "unit": "iter/sec",
            "range": "stddev: 0.00009598418630296627",
            "extra": "mean: 5.091777537233944 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 853.6225224341672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000193983126100444",
            "extra": "mean: 1.1714779937488373 msec\nrounds: 800"
          }
        ]
      }
    ]
  }
}