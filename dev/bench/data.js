window.BENCHMARK_DATA = {
  "lastUpdate": 1791617653241,
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
          "id": "123bcc76d55399287e3ea85b9554697d1e03917e",
          "message": "docs(readme): retarget bench badge and add plain intro (#145)\n\nThe /dev/bench Pages dashboard 404s because GitHub Pages is not\nenabled (has_pages: false), even though github-action-benchmark\nstill writes index.html + data.js on gh-pages. Point the badge\nat in-repo docs/benchmarks.md (HTTP 200). Add a two-line intro\nabove the existing evaluation stats; claims and DOI are unchanged.\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>",
          "timestamp": "2026-10-10T09:27:37+02:00",
          "tree_id": "11cb803ba8c661b61d7c3d3d51cfbd67602dce60",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/123bcc76d55399287e3ea85b9554697d1e03917e"
        },
        "date": 1791617652454,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 490.3404237056157,
            "unit": "iter/sec",
            "range": "stddev: 0.0007893765190995022",
            "extra": "mean: 2.0393994695415265 msec\nrounds: 1461"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 487.7691848879261,
            "unit": "iter/sec",
            "range": "stddev: 0.0009831540531352787",
            "extra": "mean: 2.050150011484978 msec\nrounds: 1219"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 477.2765817638414,
            "unit": "iter/sec",
            "range": "stddev: 0.0007654012810616793",
            "extra": "mean: 2.0952211740713573 msec\nrounds: 1373"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 399.9693476145485,
            "unit": "iter/sec",
            "range": "stddev: 0.0010957838306484106",
            "extra": "mean: 2.5001915920909585 msec\nrounds: 1846"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 494.80627276154496,
            "unit": "iter/sec",
            "range": "stddev: 0.0008458945885588831",
            "extra": "mean: 2.020992972499999 msec\nrounds: 1200"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 339.6520907440756,
            "unit": "iter/sec",
            "range": "stddev: 0.00027449853637976143",
            "extra": "mean: 2.944189148988604 msec\nrounds: 396"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 132771.98247301875,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012638768111764852",
            "extra": "mean: 7.531709487001256 usec\nrounds: 45041"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 10.83134008393989,
            "unit": "iter/sec",
            "range": "stddev: 0.00022990597001391343",
            "extra": "mean: 92.32467933333055 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.5430187250759724,
            "unit": "iter/sec",
            "range": "stddev: 0.012049190069034902",
            "extra": "mean: 1.8415571210000035 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.026692333294025105,
            "unit": "iter/sec",
            "range": "stddev: 1.9201327052649293",
            "extra": "mean: 37.46394101199999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3179.721250247968,
            "unit": "iter/sec",
            "range": "stddev: 0.00001166783595529631",
            "extra": "mean: 314.4929763645212 usec\nrounds: 2327"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 847.835759001482,
            "unit": "iter/sec",
            "range": "stddev: 0.000021937186817987736",
            "extra": "mean: 1.1794737239883886 msec\nrounds: 692"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 345.4043575733961,
            "unit": "iter/sec",
            "range": "stddev: 0.00008734462319517058",
            "extra": "mean: 2.8951574526314614 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 449.9705223008564,
            "unit": "iter/sec",
            "range": "stddev: 0.000391246199010264",
            "extra": "mean: 2.2223678006431418 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10171.59940384843,
            "unit": "iter/sec",
            "range": "stddev: 0.000005400248696430471",
            "extra": "mean: 98.3129555438105 usec\nrounds: 5781"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10200.81153587814,
            "unit": "iter/sec",
            "range": "stddev: 0.000005333968634688282",
            "extra": "mean: 98.03141607732043 usec\nrounds: 7775"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1802.8522356849883,
            "unit": "iter/sec",
            "range": "stddev: 0.000008514741970034446",
            "extra": "mean: 554.6766286256693 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1827.7188309137794,
            "unit": "iter/sec",
            "range": "stddev: 0.00001575536824149498",
            "extra": "mean: 547.1301072605593 usec\nrounds: 1529"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 993.5075776707642,
            "unit": "iter/sec",
            "range": "stddev: 0.00005936802636865041",
            "extra": "mean: 1.0065348493309503 msec\nrounds: 896"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.3298505966552,
            "unit": "iter/sec",
            "range": "stddev: 0.00006385583588109466",
            "extra": "mean: 2.7002954214704937 msec\nrounds: 1006"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.81607788769292,
            "unit": "iter/sec",
            "range": "stddev: 0.0015351582388966499",
            "extra": "mean: 40.2964563749992 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 462.0098668719262,
            "unit": "iter/sec",
            "range": "stddev: 0.0000360436587114237",
            "extra": "mean: 2.1644559385074995 msec\nrounds: 2358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 365.02376889672695,
            "unit": "iter/sec",
            "range": "stddev: 0.0000719533451720222",
            "extra": "mean: 2.739547627329774 msec\nrounds: 966"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 293.63883517758717,
            "unit": "iter/sec",
            "range": "stddev: 0.000059799032793372884",
            "extra": "mean: 3.4055440909075223 msec\nrounds: 583"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 189.88575241028366,
            "unit": "iter/sec",
            "range": "stddev: 0.00010154569438229523",
            "extra": "mean: 5.266324552035443 msec\nrounds: 221"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 830.7069296821595,
            "unit": "iter/sec",
            "range": "stddev: 0.000026332528543146185",
            "extra": "mean: 1.2037939786810428 msec\nrounds: 1032"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 4998.778378676191,
            "unit": "iter/sec",
            "range": "stddev: 0.000008502899006467444",
            "extra": "mean: 200.0488767947393 usec\nrounds: 3969"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 75.28163084455538,
            "unit": "iter/sec",
            "range": "stddev: 0.00009354709351198494",
            "extra": "mean: 13.283452932427053 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1822.1601424179514,
            "unit": "iter/sec",
            "range": "stddev: 0.000022109549410411054",
            "extra": "mean: 548.7991843971685 usec\nrounds: 1551"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 26.136107302307757,
            "unit": "iter/sec",
            "range": "stddev: 0.0043301157632824345",
            "extra": "mean: 38.26124481481993 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 16414.76122362658,
            "unit": "iter/sec",
            "range": "stddev: 0.000005994018268615249",
            "extra": "mean: 60.920776511853866 usec\nrounds: 12900"
          }
        ]
      }
    ]
  }
}