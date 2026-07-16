window.BENCHMARK_DATA = {
  "lastUpdate": 1784240600035,
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
          "id": "ebd499ed677f0a9dcd6ba39a6dec7dc68aae0293",
          "message": "chore(release): v0.17.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.21402228\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-17T00:15:24+02:00",
          "tree_id": "058bf5f8d6b3360b1d0725601dc1dce5714a827d",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/ebd499ed677f0a9dcd6ba39a6dec7dc68aae0293"
        },
        "date": 1784240598238,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 543.3851796968755,
            "unit": "iter/sec",
            "range": "stddev: 0.0008646360045744687",
            "extra": "mean: 1.840315189600579 msec\nrounds: 1577"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 452.5341569050085,
            "unit": "iter/sec",
            "range": "stddev: 0.0009810723584463932",
            "extra": "mean: 2.209777946573677 msec\nrounds: 1722"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 388.0410366340378,
            "unit": "iter/sec",
            "range": "stddev: 0.0013441953739248423",
            "extra": "mean: 2.577047027485141 msec\nrounds: 2183"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 257.9615720469219,
            "unit": "iter/sec",
            "range": "stddev: 0.0022780582414696875",
            "extra": "mean: 3.876546386599416 msec\nrounds: 3373"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 437.0075450456632,
            "unit": "iter/sec",
            "range": "stddev: 0.0010236930447288352",
            "extra": "mean: 2.2882900108635638 msec\nrounds: 1841"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 369.867271689735,
            "unit": "iter/sec",
            "range": "stddev: 0.00031521181836502747",
            "extra": "mean: 2.7036725780886472 msec\nrounds: 429"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 134223.51568817647,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012013000956266687",
            "extra": "mean: 7.450259329543759 usec\nrounds: 36363"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.51730684962296,
            "unit": "iter/sec",
            "range": "stddev: 0.000582300711770731",
            "extra": "mean: 36.34076566666996 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7977798060307033,
            "unit": "iter/sec",
            "range": "stddev: 0.008750338760017106",
            "extra": "mean: 1.2534787073333291 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03183288720105187,
            "unit": "iter/sec",
            "range": "stddev: 0.3033324765284042",
            "extra": "mean: 31.41405282166666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3150.1051222097485,
            "unit": "iter/sec",
            "range": "stddev: 0.000012136562987823666",
            "extra": "mean: 317.4497234868517 usec\nrounds: 2329"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 854.8010175803389,
            "unit": "iter/sec",
            "range": "stddev: 0.00001846382894665987",
            "extra": "mean: 1.1698629031007377 msec\nrounds: 774"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 345.3233954002786,
            "unit": "iter/sec",
            "range": "stddev: 0.00031781065557040284",
            "extra": "mean: 2.8958362315442274 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 487.28168968731165,
            "unit": "iter/sec",
            "range": "stddev: 0.000951591669027827",
            "extra": "mean: 2.0522010598052622 msec\nrounds: 719"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10409.356499472966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055478035642682234",
            "extra": "mean: 96.06741781306374 usec\nrounds: 5378"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10440.359880838705,
            "unit": "iter/sec",
            "range": "stddev: 0.00000536049139372731",
            "extra": "mean: 95.78213887390126 usec\nrounds: 7496"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1805.6242149943641,
            "unit": "iter/sec",
            "range": "stddev: 0.000010371648840780296",
            "extra": "mean: 553.8250936688514 usec\nrounds: 1153"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1782.6471467289598,
            "unit": "iter/sec",
            "range": "stddev: 0.000011378396933255662",
            "extra": "mean: 560.9635097079836 usec\nrounds: 1236"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 997.4479840951651,
            "unit": "iter/sec",
            "range": "stddev: 0.000021004316036175732",
            "extra": "mean: 1.0025585453532697 msec\nrounds: 893"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 362.9038934617633,
            "unit": "iter/sec",
            "range": "stddev: 0.00009118030160338335",
            "extra": "mean: 2.755550486000402 msec\nrounds: 1000"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.6078515317842,
            "unit": "iter/sec",
            "range": "stddev: 0.0018604520217694216",
            "extra": "mean: 40.637436336462436 msec\nrounds: 639"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 457.8079097592862,
            "unit": "iter/sec",
            "range": "stddev: 0.000054334878215569404",
            "extra": "mean: 2.184322242326037 msec\nrounds: 2476"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 358.0308122509181,
            "unit": "iter/sec",
            "range": "stddev: 0.00011536114990513507",
            "extra": "mean: 2.793055697393921 msec\nrounds: 998"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 290.2969378218482,
            "unit": "iter/sec",
            "range": "stddev: 0.00014130354855787297",
            "extra": "mean: 3.444748702839188 msec\nrounds: 599"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 181.3806858476701,
            "unit": "iter/sec",
            "range": "stddev: 0.00039076061292130917",
            "extra": "mean: 5.513266174546475 msec\nrounds: 275"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 832.5779729238626,
            "unit": "iter/sec",
            "range": "stddev: 0.00004288778817364542",
            "extra": "mean: 1.2010887058279738 msec\nrounds: 1064"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5121.3657815856,
            "unit": "iter/sec",
            "range": "stddev: 0.000009661126671895486",
            "extra": "mean: 195.26041346150345 usec\nrounds: 3120"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 73.94250455785536,
            "unit": "iter/sec",
            "range": "stddev: 0.00043782756383693754",
            "extra": "mean: 13.52402121052801 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1829.7022390723478,
            "unit": "iter/sec",
            "range": "stddev: 0.00002386283294117065",
            "extra": "mean: 546.5370149555024 usec\nrounds: 1471"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 23.660590462395408,
            "unit": "iter/sec",
            "range": "stddev: 0.005765584864205175",
            "extra": "mean: 42.264372125004 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17301.838839517623,
            "unit": "iter/sec",
            "range": "stddev: 0.000006016963363597629",
            "extra": "mean: 57.79732485520481 usec\nrounds: 12975"
          }
        ]
      }
    ]
  }
}