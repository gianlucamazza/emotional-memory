window.BENCHMARK_DATA = {
  "lastUpdate": 1781131127821,
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
          "id": "b193b4e1d864f446222868c8c26d1ad4ac3d52cf",
          "message": "fix(bench): align Hl3 classifier log by query text, measure ground-truth accuracy (#52)\n\nThe _LoggingClassifier log was matched to predictions by list index, but\npredictions restored from the resume checkpoint never hit the classifier\nin the resumed run — the log misaligned and every Hl3 record fell into\nthe 'unknown' fallback (all 200 predictions logged as 'unknown' in the\nAddendum L run). Match by query text instead (classifiers are\ndeterministic per query), extract the logic into _classifier_predictions()\nand cover it with a regression test.\n\nResolve the Hl3 required follow-up offline: HeuristicQueryClassifier is\npure, so classifying the exact 200-QA stratified subset (seed=42)\nreproduces a clean live run. Accuracy 0.465 overall, 0.600 excluding\nadversarial; multi_hop essentially undetected (2/28). Documented in the\nAddendum L closure post-closure addendum.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-06-11T00:32:52+02:00",
          "tree_id": "58a95ed5a7b8e90574eb79d1a4c4825a0c68e204",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/b193b4e1d864f446222868c8c26d1ad4ac3d52cf"
        },
        "date": 1781131126574,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 530.3123472191364,
            "unit": "iter/sec",
            "range": "stddev: 0.0008821699800685402",
            "extra": "mean: 1.8856811561032327 msec\nrounds: 1704"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 423.46251640578544,
            "unit": "iter/sec",
            "range": "stddev: 0.0011944365916121713",
            "extra": "mean: 2.3614841013265604 msec\nrounds: 1885"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 369.68449489165437,
            "unit": "iter/sec",
            "range": "stddev: 0.0015754244794989806",
            "extra": "mean: 2.705009308797427 msec\nrounds: 2228"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 240.8014511347012,
            "unit": "iter/sec",
            "range": "stddev: 0.002704937052109761",
            "extra": "mean: 4.152798894225156 msec\nrounds: 3394"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 438.6057150715769,
            "unit": "iter/sec",
            "range": "stddev: 0.001034316795309224",
            "extra": "mean: 2.2799520517802834 msec\nrounds: 1854"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.09491485517316,
            "unit": "iter/sec",
            "range": "stddev: 0.0002464025493020578",
            "extra": "mean: 2.603523143171682 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138101.3814788537,
            "unit": "iter/sec",
            "range": "stddev: 0.00000112593507072269",
            "extra": "mean: 7.2410571805403805 usec\nrounds: 41745"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.566412309434256,
            "unit": "iter/sec",
            "range": "stddev: 0.0006194119557152377",
            "extra": "mean: 35.006145999991155 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8197936281470278,
            "unit": "iter/sec",
            "range": "stddev: 0.004548247849424885",
            "extra": "mean: 1.2198191906666693 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03234883335254127,
            "unit": "iter/sec",
            "range": "stddev: 4.070153564949041",
            "extra": "mean: 30.913015906999988 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3297.717870047343,
            "unit": "iter/sec",
            "range": "stddev: 0.000013816687314458673",
            "extra": "mean: 303.24001003325486 usec\nrounds: 2392"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 893.7542097239558,
            "unit": "iter/sec",
            "range": "stddev: 0.00003854006074412993",
            "extra": "mean: 1.1188758487737465 msec\nrounds: 734"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 368.49261898919383,
            "unit": "iter/sec",
            "range": "stddev: 0.00003564033576596991",
            "extra": "mean: 2.713758562500068 msec\nrounds: 320"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.44788021314974,
            "unit": "iter/sec",
            "range": "stddev: 0.0007737463835136001",
            "extra": "mean: 1.994225201580135 msec\nrounds: 759"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10467.649360437006,
            "unit": "iter/sec",
            "range": "stddev: 0.000004872752752881279",
            "extra": "mean: 95.53243193066334 usec\nrounds: 6846"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10067.956323291553,
            "unit": "iter/sec",
            "range": "stddev: 0.000013968832635196924",
            "extra": "mean: 99.32502365813465 usec\nrounds: 8327"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1837.1673280376592,
            "unit": "iter/sec",
            "range": "stddev: 0.00001150118871882898",
            "extra": "mean: 544.3162333330486 usec\nrounds: 1080"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1820.5197778808838,
            "unit": "iter/sec",
            "range": "stddev: 0.000011801154658580452",
            "extra": "mean: 549.2936754381308 usec\nrounds: 1710"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1027.2303520337682,
            "unit": "iter/sec",
            "range": "stddev: 0.00001964049319515834",
            "extra": "mean: 973.4914841837996 usec\nrounds: 822"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.30735981017125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000330956581605895",
            "extra": "mean: 2.6156964398920657 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.90911691555853,
            "unit": "iter/sec",
            "range": "stddev: 0.0009158593437967702",
            "extra": "mean: 45.643099347827224 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 467.1484758877975,
            "unit": "iter/sec",
            "range": "stddev: 0.000036977712076655726",
            "extra": "mean: 2.1406470353981977 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.0745034428145,
            "unit": "iter/sec",
            "range": "stddev: 0.000028646720928891186",
            "extra": "mean: 2.6310630966869333 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 305.9193572731512,
            "unit": "iter/sec",
            "range": "stddev: 0.00005266334147622087",
            "extra": "mean: 3.2688353195875535 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 195.09781845530762,
            "unit": "iter/sec",
            "range": "stddev: 0.00011105301792409211",
            "extra": "mean: 5.12563394054084 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 857.0580580229287,
            "unit": "iter/sec",
            "range": "stddev: 0.00002367509048385115",
            "extra": "mean: 1.1667820991110116 msec\nrounds: 787"
          }
        ]
      }
    ]
  }
}