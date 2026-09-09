window.BENCHMARK_DATA = {
  "lastUpdate": 1788964410830,
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
          "id": "e03573a79606fb2782aa793fd88308b5c4ca4241",
          "message": "docs(research): fix ES-MemEval nDCG typo and resync evidence table with claim matrix (#130)\n\n- Fix ES-MemEval AFT nDCG@4 typo: 0.120 → 0.133 in 08_limitations.md\n  (source: benchmarks/esmemeval/results.md u_ndcg@4 AFT = 0.133)\n- Resync 09_current_evidence.md claim table from claim_validation_matrix.json:\n  * Add missing row for cross_domain_affect_replication\n  * Add missing row for query_affect_gate\n  * Fix Hi3_arc status: Not established → Falsified (per JSON SSOT)\n- Add DOI split documentation to ssot-policy.md: badge/codemeta use\n  concept DOI (10.5281/zenodo.19972258), CITATION.cff/bibtex use\n  version DOI (10.5281/zenodo.21870707) - intentional split\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Gianluca Mazza <gianlucamazza@users.noreply.github.com>",
          "timestamp": "2026-09-09T16:26:38+02:00",
          "tree_id": "e88488f1827f885d54848c937915ecb685efa8e5",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e03573a79606fb2782aa793fd88308b5c4ca4241"
        },
        "date": 1788964409353,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 570.669012388099,
            "unit": "iter/sec",
            "range": "stddev: 0.0008056067235561908",
            "extra": "mean: 1.7523292456607453 msec\nrounds: 1498"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 448.7896625098603,
            "unit": "iter/sec",
            "range": "stddev: 0.0009973055139304852",
            "extra": "mean: 2.2282153167421255 msec\nrounds: 1768"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 339.65814148363955,
            "unit": "iter/sec",
            "range": "stddev: 0.0016551421881059844",
            "extra": "mean: 2.9441367006012644 msec\nrounds: 2328"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 230.66046572288533,
            "unit": "iter/sec",
            "range": "stddev: 0.0028889314079810946",
            "extra": "mean: 4.335376662255579 msec\nrounds: 3476"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 429.82697556848416,
            "unit": "iter/sec",
            "range": "stddev: 0.0010467146128447411",
            "extra": "mean: 2.326517545059641 msec\nrounds: 1842"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 365.4765224415212,
            "unit": "iter/sec",
            "range": "stddev: 0.00031792418141190733",
            "extra": "mean: 2.736153866518217 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136340.81090926364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012305844635627702",
            "extra": "mean: 7.334561041048167 usec\nrounds: 32003"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.17802126177546,
            "unit": "iter/sec",
            "range": "stddev: 0.0004489394058198532",
            "extra": "mean: 35.48865233331829 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.7987970786984255,
            "unit": "iter/sec",
            "range": "stddev: 0.008037462517530097",
            "extra": "mean: 1.2518823950000144 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.0310720950544954,
            "unit": "iter/sec",
            "range": "stddev: 1.7214331150608007",
            "extra": "mean: 32.18321771499999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3176.980633650881,
            "unit": "iter/sec",
            "range": "stddev: 0.00002155493446899578",
            "extra": "mean: 314.76427316172624 usec\nrounds: 2094"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 879.7635693275589,
            "unit": "iter/sec",
            "range": "stddev: 0.000040412601747823145",
            "extra": "mean: 1.1366690266162567 msec\nrounds: 789"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 361.17157097301,
            "unit": "iter/sec",
            "range": "stddev: 0.00004282868378986303",
            "extra": "mean: 2.7687672019864737 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 486.2613714549592,
            "unit": "iter/sec",
            "range": "stddev: 0.0008529036079396147",
            "extra": "mean: 2.056507176393358 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10079.29407698872,
            "unit": "iter/sec",
            "range": "stddev: 0.000006081889808668451",
            "extra": "mean: 99.21329731642864 usec\nrounds: 6932"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10195.889129881092,
            "unit": "iter/sec",
            "range": "stddev: 0.000005217946321773413",
            "extra": "mean: 98.07874401745896 usec\nrounds: 7188"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1819.4826309736206,
            "unit": "iter/sec",
            "range": "stddev: 0.000013574429008320175",
            "extra": "mean: 549.6067854546606 usec\nrounds: 1100"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1796.4030522533055,
            "unit": "iter/sec",
            "range": "stddev: 0.000012672740402928973",
            "extra": "mean: 556.6679475108089 usec\nrounds: 1486"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1014.0059594016855,
            "unit": "iter/sec",
            "range": "stddev: 0.000047143039730036595",
            "extra": "mean: 986.1874979413832 usec\nrounds: 972"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 371.10701426446155,
            "unit": "iter/sec",
            "range": "stddev: 0.00007365802444230761",
            "extra": "mean: 2.6946405256769714 msec\nrounds: 1071"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.671978890602507,
            "unit": "iter/sec",
            "range": "stddev: 0.0020118476537845757",
            "extra": "mean: 44.107309945251316 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 404.6407053664277,
            "unit": "iter/sec",
            "range": "stddev: 0.0004522407910868544",
            "extra": "mean: 2.47132823449988 msec\nrounds: 2371"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 291.9689443973663,
            "unit": "iter/sec",
            "range": "stddev: 0.0007721859794098397",
            "extra": "mean: 3.425021801767423 msec\nrounds: 1019"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 259.860114933563,
            "unit": "iter/sec",
            "range": "stddev: 0.0003279412028122211",
            "extra": "mean: 3.848224265796482 msec\nrounds: 538"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 142.09825360068652,
            "unit": "iter/sec",
            "range": "stddev: 0.0010421665518260104",
            "extra": "mean: 7.0373841666634585 msec\nrounds: 276"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 842.9427643106478,
            "unit": "iter/sec",
            "range": "stddev: 0.000036953614615008556",
            "extra": "mean: 1.1863201658985618 msec\nrounds: 1085"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 5190.0048790630735,
            "unit": "iter/sec",
            "range": "stddev: 0.000010863875697005165",
            "extra": "mean: 192.67804622575332 usec\nrounds: 4608"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 67.14421687534295,
            "unit": "iter/sec",
            "range": "stddev: 0.0010610540481664939",
            "extra": "mean: 14.893315411758495 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 1848.8374230166473,
            "unit": "iter/sec",
            "range": "stddev: 0.000023682600123443416",
            "extra": "mean: 540.8804406221692 usec\nrounds: 1541"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 23.50574023433309,
            "unit": "iter/sec",
            "range": "stddev: 0.006173423060056836",
            "extra": "mean: 42.542799760008165 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 17221.11156217835,
            "unit": "iter/sec",
            "range": "stddev: 0.000006588401251460082",
            "extra": "mean: 58.06826094758235 usec\nrounds: 11738"
          }
        ]
      }
    ]
  }
}