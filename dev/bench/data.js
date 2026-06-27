window.BENCHMARK_DATA = {
  "lastUpdate": 1782573146973,
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
          "id": "98a312d648e16e6b93fd96b48843e789cd97a269",
          "message": "feat(bench): Addendum W closure — affine arousal calibration ADOPTED (#84)\n\nEXECUTED on the Addendum V LLM dump (gpt-5-mini, N=300 EmoBank, 0 fallback).\nVerdict adopt=YES: Hw1 PASS, Hw2 PASS, Gw OK.\n\n- Calibrated direct-VAD arousal MAE 0.04 vs raw 0.20 (Hw1, -80%, p<0.001) and vs\n  the SEC->projection 0.11 (Hw2, p<0.001) -> calibrated direct-VAD now DOMINATES\n  the production projection on both r (0.57 vs 0.21) AND MAE (0.04 vs 0.12).\n- r preserved (Gw); deployable affine: arousal_cal = 0.163*arousal_direct + 0.449.\n- Honest caveat documented: the large MAE cut partly reflects EmoBank's narrow\n  arousal variance (shrink-to-mean); the durable claim is dominance-on-both-axes\n  via r-preservation, not the absolute 0.04. Coefficients are EmoBank-fit.\n\npredictions.json committed so the calibration is reproducible offline with no LLM.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T17:04:53+02:00",
          "tree_id": "4d70d643c9947b31f3dc7fba4f0e2c044d3a2657",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/98a312d648e16e6b93fd96b48843e789cd97a269"
        },
        "date": 1782573146384,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 619.2708808805206,
            "unit": "iter/sec",
            "range": "stddev: 0.0007588684452838608",
            "extra": "mean: 1.6148022309366998 msec\nrounds: 1377"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 424.4212844927838,
            "unit": "iter/sec",
            "range": "stddev: 0.0011950045418339075",
            "extra": "mean: 2.3561495064864086 msec\nrounds: 1850"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 310.2549884121638,
            "unit": "iter/sec",
            "range": "stddev: 0.0018794708087340286",
            "extra": "mean: 3.223155266955876 msec\nrounds: 2536"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 172.13069753756255,
            "unit": "iter/sec",
            "range": "stddev: 0.004083279022932906",
            "extra": "mean: 5.80953899743408 msec\nrounds: 4287"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 424.7663258681315,
            "unit": "iter/sec",
            "range": "stddev: 0.0010797933418282433",
            "extra": "mean: 2.3542355857805206 msec\nrounds: 1941"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 372.80576308870417,
            "unit": "iter/sec",
            "range": "stddev: 0.0003119914560625588",
            "extra": "mean: 2.682361967033389 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135506.17570238514,
            "unit": "iter/sec",
            "range": "stddev: 8.761372947608328e-7",
            "extra": "mean: 7.379737453415552 usec\nrounds: 43737"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.84796271186657,
            "unit": "iter/sec",
            "range": "stddev: 0.0006369280196061442",
            "extra": "mean: 30.44328833333528 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8428069583261768,
            "unit": "iter/sec",
            "range": "stddev: 0.002072768467102963",
            "extra": "mean: 1.1865113239999943 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.027060740792585726,
            "unit": "iter/sec",
            "range": "stddev: 9.31568174194344",
            "extra": "mean: 36.953903356333335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3441.880042804536,
            "unit": "iter/sec",
            "range": "stddev: 0.000009375388642602134",
            "extra": "mean: 290.5388879227683 usec\nrounds: 2186"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 879.4693555161589,
            "unit": "iter/sec",
            "range": "stddev: 0.00008570086664791639",
            "extra": "mean: 1.1370492828747876 msec\nrounds: 654"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 356.1543614688562,
            "unit": "iter/sec",
            "range": "stddev: 0.00003686700421502919",
            "extra": "mean: 2.8077713154369013 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 486.4448174643103,
            "unit": "iter/sec",
            "range": "stddev: 0.0004901269155970597",
            "extra": "mean: 2.0557316351168002 msec\nrounds: 729"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12132.093140178573,
            "unit": "iter/sec",
            "range": "stddev: 0.000004325465169268657",
            "extra": "mean: 82.42600748655981 usec\nrounds: 6545"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12077.37595518863,
            "unit": "iter/sec",
            "range": "stddev: 0.000004269883494839399",
            "extra": "mean: 82.7994428351288 usec\nrounds: 6560"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2005.1763840046488,
            "unit": "iter/sec",
            "range": "stddev: 0.000013332655978901356",
            "extra": "mean: 498.709244721327 usec\nrounds: 1042"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2034.6919566198956,
            "unit": "iter/sec",
            "range": "stddev: 0.000014898787786602511",
            "extra": "mean: 491.4748872656068 usec\nrounds: 1437"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1157.368504732687,
            "unit": "iter/sec",
            "range": "stddev: 0.000016881269709179933",
            "extra": "mean: 864.0290416672142 usec\nrounds: 840"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 384.8409722730807,
            "unit": "iter/sec",
            "range": "stddev: 0.00005603039847819769",
            "extra": "mean: 2.5984759213486406 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 17.77211587128265,
            "unit": "iter/sec",
            "range": "stddev: 0.00041674997505425305",
            "extra": "mean: 56.267920333327645 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 456.67584385572076,
            "unit": "iter/sec",
            "range": "stddev: 0.00013544117490118902",
            "extra": "mean: 2.189737016867338 msec\nrounds: 415"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 367.4724285171511,
            "unit": "iter/sec",
            "range": "stddev: 0.00012822677892151843",
            "extra": "mean: 2.7212925988359604 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 256.1933811162218,
            "unit": "iter/sec",
            "range": "stddev: 0.0007289433453956874",
            "extra": "mean: 3.9033014656469645 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 181.5876041373958,
            "unit": "iter/sec",
            "range": "stddev: 0.0007086449404756268",
            "extra": "mean: 5.506983831580065 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 942.6064032091215,
            "unit": "iter/sec",
            "range": "stddev: 0.000024540575159633362",
            "extra": "mean: 1.060888188957216 msec\nrounds: 815"
          }
        ]
      }
    ]
  }
}