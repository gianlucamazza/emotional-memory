window.BENCHMARK_DATA = {
  "lastUpdate": 1779894509659,
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
          "id": "75efc3e94a9362909faaf1a6a730f4c4f399605e",
          "message": "fix(benchmarks): write benchmark JSON to /tmp to avoid git switch conflict\n\nbenchmark-results.json is tracked in the gh-pages branch; writing it to the\ncwd caused 'git switch gh-pages' to fail with 'local changes would be\noverwritten'.  Use /tmp/benchmark-results.json so the file never lands in\nthe git working tree.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-27T17:00:40+02:00",
          "tree_id": "e0a7e8796700f9acb412d25e62ed22a27c9601c4",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/75efc3e94a9362909faaf1a6a730f4c4f399605e"
        },
        "date": 1779894508561,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 429.3763844778013,
            "unit": "iter/sec",
            "range": "stddev: 0.001402271721097056",
            "extra": "mean: 2.3289590116051193 msec\nrounds: 1551"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 344.9984995074566,
            "unit": "iter/sec",
            "range": "stddev: 0.001668584655016087",
            "extra": "mean: 2.8985633312251164 msec\nrounds: 1739"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 296.3826324836854,
            "unit": "iter/sec",
            "range": "stddev: 0.002330184852460306",
            "extra": "mean: 3.3740168633364362 msec\nrounds: 2122"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 148.90222075014137,
            "unit": "iter/sec",
            "range": "stddev: 0.004645929517929827",
            "extra": "mean: 6.715816560439382 msec\nrounds: 3549"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 494.8934629801322,
            "unit": "iter/sec",
            "range": "stddev: 0.0008711512459425189",
            "extra": "mean: 2.0206369144143363 msec\nrounds: 1554"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 363.1030747744856,
            "unit": "iter/sec",
            "range": "stddev: 0.0003439530213188929",
            "extra": "mean: 2.7540389202737416 msec\nrounds: 439"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 132791.50350229046,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015580277089090418",
            "extra": "mean: 7.530602287237086 usec\nrounds: 46169"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.035093714447257,
            "unit": "iter/sec",
            "range": "stddev: 0.0007124223244259476",
            "extra": "mean: 36.98896000000218 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7995364149279895,
            "unit": "iter/sec",
            "range": "stddev: 0.013332352567260569",
            "extra": "mean: 1.250724771666673 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02169036694085258,
            "unit": "iter/sec",
            "range": "stddev: 5.102358664282205",
            "extra": "mean: 46.10341552666666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3137.07707692402,
            "unit": "iter/sec",
            "range": "stddev: 0.00001524633591902009",
            "extra": "mean: 318.76806832573084 usec\nrounds: 2049"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 869.0753098748253,
            "unit": "iter/sec",
            "range": "stddev: 0.000021542633176254024",
            "extra": "mean: 1.1506482679205696 msec\nrounds: 586"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.97181076258187,
            "unit": "iter/sec",
            "range": "stddev: 0.00006260920213023476",
            "extra": "mean: 2.8092112065215122 msec\nrounds: 276"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 427.3517612201445,
            "unit": "iter/sec",
            "range": "stddev: 0.0010049008219228637",
            "extra": "mean: 2.3399926962857736 msec\nrounds: 754"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10262.620766082804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055666431850076",
            "extra": "mean: 97.44099707015633 usec\nrounds: 4437"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10130.864990372544,
            "unit": "iter/sec",
            "range": "stddev: 0.00000536449519298919",
            "extra": "mean: 98.70825452222584 usec\nrounds: 6247"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1812.6547015406775,
            "unit": "iter/sec",
            "range": "stddev: 0.000024952356448602606",
            "extra": "mean: 551.6770508746334 usec\nrounds: 629"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1763.4379580137754,
            "unit": "iter/sec",
            "range": "stddev: 0.0001393386770374563",
            "extra": "mean: 567.0741039998575 usec\nrounds: 375"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1004.4322985556677,
            "unit": "iter/sec",
            "range": "stddev: 0.00003071198908033151",
            "extra": "mean: 995.5872600253485 usec\nrounds: 773"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 237.48778760353645,
            "unit": "iter/sec",
            "range": "stddev: 0.002054035892252138",
            "extra": "mean: 4.210742834782755 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.577981533004348,
            "unit": "iter/sec",
            "range": "stddev: 0.002085745878603932",
            "extra": "mean: 51.07778850001523 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 231.1695795707363,
            "unit": "iter/sec",
            "range": "stddev: 0.001113053777612273",
            "extra": "mean: 4.325828691893291 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 210.9826792938194,
            "unit": "iter/sec",
            "range": "stddev: 0.001986504430131497",
            "extra": "mean: 4.739725570587606 msec\nrounds: 340"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 156.95393363171752,
            "unit": "iter/sec",
            "range": "stddev: 0.0005552756505781193",
            "extra": "mean: 6.371296194120479 msec\nrounds: 170"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 112.69291608442266,
            "unit": "iter/sec",
            "range": "stddev: 0.0004561871074900356",
            "extra": "mean: 8.87367223021242 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 776.7631247689931,
            "unit": "iter/sec",
            "range": "stddev: 0.000052000967524778174",
            "extra": "mean: 1.2873937602243886 msec\nrounds: 709"
          }
        ]
      }
    ]
  }
}