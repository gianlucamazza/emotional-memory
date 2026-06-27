window.BENCHMARK_DATA = {
  "lastUpdate": 1782579725904,
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
          "id": "7235516e9f4c9c4d1adcaac4c58e22e6c4bfe9cb",
          "message": "docs(bench): record Addendum W library integration as evaluated-and-declined (#86)\n\nThe closure's suggested follow-up (ship the calibration as opt-in post-processing\non DIRECT_VAD_SCHEMA) was evaluated against the codebase and declined: that output\ndrives the engine, so calibrated arousal (band [0.45,0.61]) would silently disable\nthe decay floor (0.7 never reached), collapse the affect-proximity s3 signal\n(sqrt(6) normalizer assumes [0,1]), and mis-fire the inverted-U / 0.7 weight gate.\nCalibration optimizes absolute agreement with human gold; retrieval/decay need\ndiscriminative spread — opposed objectives. Calibration is measurement/reporting\nonly; the engine keeps raw direct-VAD arousal. Prevents re-litigating the footgun.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T18:55:58+02:00",
          "tree_id": "90975a66564f377cb59452ac2768d38c2d8d2578",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/7235516e9f4c9c4d1adcaac4c58e22e6c4bfe9cb"
        },
        "date": 1782579725103,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 543.7733186788046,
            "unit": "iter/sec",
            "range": "stddev: 0.0008500827382660049",
            "extra": "mean: 1.8390015943218407 msec\nrounds: 1585"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 427.8381536637258,
            "unit": "iter/sec",
            "range": "stddev: 0.0011295183983765673",
            "extra": "mean: 2.3373324502189785 msec\nrounds: 1828"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 354.33253071216524,
            "unit": "iter/sec",
            "range": "stddev: 0.0015942811522960607",
            "extra": "mean: 2.8222077097751135 msec\nrounds: 2312"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 236.44936047387029,
            "unit": "iter/sec",
            "range": "stddev: 0.002754590941902577",
            "extra": "mean: 4.229235376217093 msec\nrounds: 3389"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 474.4331243677676,
            "unit": "iter/sec",
            "range": "stddev: 0.0009061248904960523",
            "extra": "mean: 2.1077786280892714 msec\nrounds: 1659"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 375.06375227581896,
            "unit": "iter/sec",
            "range": "stddev: 0.00029759308364997396",
            "extra": "mean: 2.6662133942088007 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134192.03696946998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014268683620617058",
            "extra": "mean: 7.452007008638746 usec\nrounds: 38524"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 26.234838657186398,
            "unit": "iter/sec",
            "range": "stddev: 0.0026323677485612573",
            "extra": "mean: 38.11725366666489 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8022239573435042,
            "unit": "iter/sec",
            "range": "stddev: 0.004430864059787514",
            "extra": "mean: 1.2465347000000027 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.031448385380390034,
            "unit": "iter/sec",
            "range": "stddev: 1.6706901733685415",
            "extra": "mean: 31.79813487733333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3168.418543726864,
            "unit": "iter/sec",
            "range": "stddev: 0.00001625700952729214",
            "extra": "mean: 315.6148678588866 usec\nrounds: 2293"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 854.6137009249799,
            "unit": "iter/sec",
            "range": "stddev: 0.00012024079165790855",
            "extra": "mean: 1.1701193169705366 msec\nrounds: 713"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.8692770867309,
            "unit": "iter/sec",
            "range": "stddev: 0.00008795011795390185",
            "extra": "mean: 2.8179390681814294 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 487.8779213274677,
            "unit": "iter/sec",
            "range": "stddev: 0.0008965783180637294",
            "extra": "mean: 2.049693081578889 msec\nrounds: 760"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10772.574721772928,
            "unit": "iter/sec",
            "range": "stddev: 0.000006347056287966999",
            "extra": "mean: 92.82831874712883 usec\nrounds: 7313"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10616.690412972095,
            "unit": "iter/sec",
            "range": "stddev: 0.000005884104677435102",
            "extra": "mean: 94.19131208518066 usec\nrounds: 6934"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1811.9124966346594,
            "unit": "iter/sec",
            "range": "stddev: 0.000016064380521972704",
            "extra": "mean: 551.9030316625895 usec\nrounds: 1137"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1783.9170171382243,
            "unit": "iter/sec",
            "range": "stddev: 0.00001657053678103443",
            "extra": "mean: 560.5641912672647 usec\nrounds: 1626"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 999.6139591621386,
            "unit": "iter/sec",
            "range": "stddev: 0.00002766367023326613",
            "extra": "mean: 1.000386189922943 msec\nrounds: 774"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.60928263984874,
            "unit": "iter/sec",
            "range": "stddev: 0.000040165959396494316",
            "extra": "mean: 2.6909984403408096 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.07597269972574,
            "unit": "iter/sec",
            "range": "stddev: 0.000857848203571118",
            "extra": "mean: 45.29811726087266 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 464.64642881558797,
            "unit": "iter/sec",
            "range": "stddev: 0.00006630458710207787",
            "extra": "mean: 2.1521740790068287 msec\nrounds: 443"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 373.8968802981339,
            "unit": "iter/sec",
            "range": "stddev: 0.0000495044438645973",
            "extra": "mean: 2.6745342170350037 msec\nrounds: 364"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 300.86785237982804,
            "unit": "iter/sec",
            "range": "stddev: 0.00009829586496074456",
            "extra": "mean: 3.3237183437516573 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 191.02185886454643,
            "unit": "iter/sec",
            "range": "stddev: 0.00013345465767487593",
            "extra": "mean: 5.235002977900555 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 840.2898684943051,
            "unit": "iter/sec",
            "range": "stddev: 0.00003353630066533337",
            "extra": "mean: 1.1900655208325617 msec\nrounds: 768"
          }
        ]
      }
    ]
  }
}