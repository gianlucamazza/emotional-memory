window.BENCHMARK_DATA = {
  "lastUpdate": 1781138614666,
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
          "id": "35790d7f4d36ec2bc443998df26fa5fb4ad219df",
          "message": "fix(release): commit README BibTeX DOI and arXiv bundle in the release commit\n\nThe on-tag G3b gate failed for v0.11.4: phase 2 of scripts/release.py\nsyncs the README BibTeX version/doi (extension added in #36) and patches\npaper/main.tex, but phase 3's managed file list included neither\nREADME.md nor the arXiv bundle artifacts -- the tag carried a stale\nREADME DOI and a stale bundle. Add README.md and the bundle to the\nmanaged list, regenerate the bundle in phase 2 right after the DOI-\npatched PDF rebuild, and include the synced files this release left in\nthe worktree.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-06-11T02:36:11+02:00",
          "tree_id": "0b73c1c4c15e6dfa56c34d4ff15367e6e30ba28c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/35790d7f4d36ec2bc443998df26fa5fb4ad219df"
        },
        "date": 1781138614122,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 727.6787771010495,
            "unit": "iter/sec",
            "range": "stddev: 0.0006381180031767741",
            "extra": "mean: 1.3742327404185577 msec\nrounds: 1148"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 434.6598729952946,
            "unit": "iter/sec",
            "range": "stddev: 0.0011761878699067104",
            "extra": "mean: 2.3006494551909684 msec\nrounds: 1830"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 316.9768496729782,
            "unit": "iter/sec",
            "range": "stddev: 0.0019144572694960794",
            "extra": "mean: 3.1548045260456394 msec\nrounds: 2534"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 196.78823641064386,
            "unit": "iter/sec",
            "range": "stddev: 0.0034506904766957937",
            "extra": "mean: 5.08160456254748 msec\nrounds: 3941"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 439.2301462293385,
            "unit": "iter/sec",
            "range": "stddev: 0.0010598413495149777",
            "extra": "mean: 2.2767107599164254 msec\nrounds: 1916"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.5533185438058,
            "unit": "iter/sec",
            "range": "stddev: 0.00025154106106976457",
            "extra": "mean: 2.6004196343610193 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137623.33311115604,
            "unit": "iter/sec",
            "range": "stddev: 8.38961951132449e-7",
            "extra": "mean: 7.266209714542496 usec\nrounds: 48237"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.14064222432699,
            "unit": "iter/sec",
            "range": "stddev: 0.0005271024069010157",
            "extra": "mean: 30.17443033333696 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8507046121023939,
            "unit": "iter/sec",
            "range": "stddev: 0.004418038767275661",
            "extra": "mean: 1.1754961543333404 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02573227489048597,
            "unit": "iter/sec",
            "range": "stddev: 5.971751607086007",
            "extra": "mean: 38.861702055333296 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3512.727789384738,
            "unit": "iter/sec",
            "range": "stddev: 0.00001206381601577706",
            "extra": "mean: 284.679047155872 usec\nrounds: 2163"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 901.6655164245695,
            "unit": "iter/sec",
            "range": "stddev: 0.000020884759355727812",
            "extra": "mean: 1.1090587161027987 msec\nrounds: 708"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.62719413342614,
            "unit": "iter/sec",
            "range": "stddev: 0.000046139304429506906",
            "extra": "mean: 2.75765308332628 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 490.3126528303439,
            "unit": "iter/sec",
            "range": "stddev: 0.0009602643380756462",
            "extra": "mean: 2.0395149793248684 msec\nrounds: 774"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12281.415116850942,
            "unit": "iter/sec",
            "range": "stddev: 0.00000400158478079221",
            "extra": "mean: 81.42384167341852 usec\nrounds: 6594"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12156.61346847379,
            "unit": "iter/sec",
            "range": "stddev: 0.000007416629297243747",
            "extra": "mean: 82.25975125336824 usec\nrounds: 8776"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2070.5226829412127,
            "unit": "iter/sec",
            "range": "stddev: 0.000013659209906467495",
            "extra": "mean: 482.9698357032646 usec\nrounds: 1126"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2040.8937367675169,
            "unit": "iter/sec",
            "range": "stddev: 0.00004599000539699861",
            "extra": "mean: 489.9814145070858 usec\nrounds: 1544"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1185.373840816318,
            "unit": "iter/sec",
            "range": "stddev: 0.000026307704721874088",
            "extra": "mean: 843.615714778505 usec\nrounds: 866"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 389.7321669127061,
            "unit": "iter/sec",
            "range": "stddev: 0.0001182025343727684",
            "extra": "mean: 2.565864675532375 msec\nrounds: 376"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 18.959974183413717,
            "unit": "iter/sec",
            "range": "stddev: 0.0006977959711748921",
            "extra": "mean: 52.742687850008 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 465.2253351231116,
            "unit": "iter/sec",
            "range": "stddev: 0.00007394774436149401",
            "extra": "mean: 2.1494960065650166 msec\nrounds: 457"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.3310868174615,
            "unit": "iter/sec",
            "range": "stddev: 0.00017340454278453932",
            "extra": "mean: 2.6572346399994524 msec\nrounds: 350"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 315.0187145701813,
            "unit": "iter/sec",
            "range": "stddev: 0.0002824307686431494",
            "extra": "mean: 3.1744145783986917 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 198.43486139639546,
            "unit": "iter/sec",
            "range": "stddev: 0.0005426448182320558",
            "extra": "mean: 5.039437087631442 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 967.9508739027311,
            "unit": "iter/sec",
            "range": "stddev: 0.000027868870629321283",
            "extra": "mean: 1.0331102816902766 msec\nrounds: 852"
          }
        ]
      }
    ]
  }
}