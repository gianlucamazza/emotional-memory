window.BENCHMARK_DATA = {
  "lastUpdate": 1784241141954,
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
          "id": "8a07df8be796d2151d14f415577996a1960327cd",
          "message": "docs: clear post-0.17.0 roadmap and arXiv checklist drift\n\nMark v0.14–v0.17 as shipped in ROADMAP (v0.14 was still \"release pending\").\nRefresh ARXIV_CHECKLIST software snapshot to v0.17.0 / Zenodo 21402228.",
          "timestamp": "2026-07-17T00:25:21+02:00",
          "tree_id": "85ca09fb586d76cb152f15dd43a92990a5d220f7",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8a07df8be796d2151d14f415577996a1960327cd"
        },
        "date": 1784241140487,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 605.3789307813461,
            "unit": "iter/sec",
            "range": "stddev: 0.0007839450290320322",
            "extra": "mean: 1.651857950704243 msec\nrounds: 1420"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 400.5010163305905,
            "unit": "iter/sec",
            "range": "stddev: 0.0012349253819497765",
            "extra": "mean: 2.4968725651736112 msec\nrounds: 1987"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 364.97908518863005,
            "unit": "iter/sec",
            "range": "stddev: 0.0014561502820896208",
            "extra": "mean: 2.739883025031902 msec\nrounds: 2357"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 208.20688637807714,
            "unit": "iter/sec",
            "range": "stddev: 0.002822613324570834",
            "extra": "mean: 4.80291510715994 msec\nrounds: 4190"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 425.6410968397078,
            "unit": "iter/sec",
            "range": "stddev: 0.0010905129889251348",
            "extra": "mean: 2.34939719736835 msec\nrounds: 1900"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 375.79490241440266,
            "unit": "iter/sec",
            "range": "stddev: 0.0002532193003473476",
            "extra": "mean: 2.661025984054631 msec\nrounds: 439"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 132395.9393662213,
            "unit": "iter/sec",
            "range": "stddev: 8.538742510244408e-7",
            "extra": "mean: 7.553101740030661 usec\nrounds: 42471"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.49021795347823,
            "unit": "iter/sec",
            "range": "stddev: 0.0004291461280944894",
            "extra": "mean: 30.778494666667672 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8300973666137255,
            "unit": "iter/sec",
            "range": "stddev: 0.005423331419207322",
            "extra": "mean: 1.204677957333331 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03245091374421781,
            "unit": "iter/sec",
            "range": "stddev: 2.682169469736594",
            "extra": "mean: 30.81577325933334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3454.605311303426,
            "unit": "iter/sec",
            "range": "stddev: 0.000011825240804331693",
            "extra": "mean: 289.4686685995684 usec\nrounds: 2242"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 890.5499342177202,
            "unit": "iter/sec",
            "range": "stddev: 0.000020746300592052935",
            "extra": "mean: 1.1229016606221227 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 357.0805650567042,
            "unit": "iter/sec",
            "range": "stddev: 0.000034864452487340244",
            "extra": "mean: 2.800488455150732 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 488.59674709268086,
            "unit": "iter/sec",
            "range": "stddev: 0.00045247677156003824",
            "extra": "mean: 2.046677563758549 msec\nrounds: 745"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12189.84278632519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037925313107204832",
            "extra": "mean: 82.03551247779995 usec\nrounds: 6732"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11779.842387248851,
            "unit": "iter/sec",
            "range": "stddev: 0.000003846530411822025",
            "extra": "mean: 84.89077927582927 usec\nrounds: 8010"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2007.8671114497802,
            "unit": "iter/sec",
            "range": "stddev: 0.000012662052795731569",
            "extra": "mean: 498.0409282554313 usec\nrounds: 1129"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2004.716012143679,
            "unit": "iter/sec",
            "range": "stddev: 0.000012666550119925957",
            "extra": "mean: 498.8237705203352 usec\nrounds: 1730"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1147.3768808665993,
            "unit": "iter/sec",
            "range": "stddev: 0.00002074852814654591",
            "extra": "mean: 871.553206863217 usec\nrounds: 1020"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 375.8451671971553,
            "unit": "iter/sec",
            "range": "stddev: 0.00009208377916107232",
            "extra": "mean: 2.660670103748959 msec\nrounds: 1147"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.5656116036744,
            "unit": "iter/sec",
            "range": "stddev: 0.0015904107584298007",
            "extra": "mean: 46.370120095718384 msec\nrounds: 397"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 462.7455340479433,
            "unit": "iter/sec",
            "range": "stddev: 0.00006645493231851343",
            "extra": "mean: 2.161014912996208 msec\nrounds: 2747"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 381.85855948039244,
            "unit": "iter/sec",
            "range": "stddev: 0.00007756120585544146",
            "extra": "mean: 2.6187706813767195 msec\nrounds: 1133"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 314.24486154540284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000821040000824364",
            "extra": "mean: 3.1822318273787196 msec\nrounds: 672"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 205.03253297465187,
            "unit": "iter/sec",
            "range": "stddev: 0.0001265958013048797",
            "extra": "mean: 4.877274769480753 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 945.7559672286211,
            "unit": "iter/sec",
            "range": "stddev: 0.00001972267592595842",
            "extra": "mean: 1.0573552107002104 msec\nrounds: 1215"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5493.417518171572,
            "unit": "iter/sec",
            "range": "stddev: 0.000005527458986369007",
            "extra": "mean: 182.03604526546889 usec\nrounds: 5037"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 80.28723028220675,
            "unit": "iter/sec",
            "range": "stddev: 0.00019681087980782274",
            "extra": "mean: 12.455280827163122 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 2268.6207847334763,
            "unit": "iter/sec",
            "range": "stddev: 0.000013093064914391741",
            "extra": "mean: 440.7964551543517 usec\nrounds: 1795"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 30.24430521686404,
            "unit": "iter/sec",
            "range": "stddev: 0.006438924485570431",
            "extra": "mean: 33.064075793098596 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 21523.612036696537,
            "unit": "iter/sec",
            "range": "stddev: 0.000003791079791029893",
            "extra": "mean: 46.460603280483625 usec\nrounds: 16218"
          }
        ]
      }
    ]
  }
}