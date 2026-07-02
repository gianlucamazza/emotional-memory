window.BENCHMARK_DATA = {
  "lastUpdate": 1782981051005,
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
          "id": "0c3939589fe14bc0ae9380a7acc53269f9299b67",
          "message": "docs: fix drift found by audit (versions, counts, roadmap, conventions) (#95)\n\n- SECURITY.md: supported-versions table shifted to 0.14.x latest\n- ARXIV_CHECKLIST: page-count row 18pp->19pp with X in addenda list;\n  stale 43/43 cite-key mention -> 44/44 (matches refs.bib)\n- docs/comparison.md: 29-system -> 33-paper; benchmark caveat now states\n  the oracle-affect regime bound and the Addendum X counter-congruence\n  result, linking README \"When NOT to use\"\n- ROADMAP: v0.11.x section date range corrected (contained Q 2026-06-11\n  and X 2026-07-02); missing addenda entries added (R/S/U/V/T/T2A/W);\n  new milestone section v0.12.0-v0.14.0 (v0.14.0 = bump on main, release\n  pending via make release)\n- CLAUDE.md: bench-x-madial command + madialbench/query_appraisal/\n  circularity_audit/appraisal_vad/arousal_calibration conventions +\n  pointer to benchmarks/README.md index\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T10:24:27+02:00",
          "tree_id": "84aa9c7d869c19b9cccc576dbcdaf813b1ccbfc6",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/0c3939589fe14bc0ae9380a7acc53269f9299b67"
        },
        "date": 1782981050378,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 564.4190416333838,
            "unit": "iter/sec",
            "range": "stddev: 0.0008163297239053197",
            "extra": "mean: 1.7717332801283237 msec\nrounds: 1560"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 429.658941437091,
            "unit": "iter/sec",
            "range": "stddev: 0.0011434110754505338",
            "extra": "mean: 2.327427416395141 msec\nrounds: 1842"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 359.3025097884734,
            "unit": "iter/sec",
            "range": "stddev: 0.0016335352231807137",
            "extra": "mean: 2.783170094160251 msec\nrounds: 2209"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 221.78432674383117,
            "unit": "iter/sec",
            "range": "stddev: 0.0029513162053589797",
            "extra": "mean: 4.508884891379343 msec\nrounds: 3480"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 467.85674248021735,
            "unit": "iter/sec",
            "range": "stddev: 0.0009379310824658081",
            "extra": "mean: 2.1374064092755565 msec\nrounds: 1725"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 381.23232731365954,
            "unit": "iter/sec",
            "range": "stddev: 0.00031863237811358743",
            "extra": "mean: 2.6230724111107406 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135308.25022189572,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011058767771884318",
            "extra": "mean: 7.390532346402178 usec\nrounds: 44920"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.112027704816523,
            "unit": "iter/sec",
            "range": "stddev: 0.0003210177664066057",
            "extra": "mean: 35.57196266666551 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8039170582113874,
            "unit": "iter/sec",
            "range": "stddev: 0.004505488371482104",
            "extra": "mean: 1.2439094180000012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.029420223913824768,
            "unit": "iter/sec",
            "range": "stddev: 1.128965966448094",
            "extra": "mean: 33.990223967333336 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3146.854991533161,
            "unit": "iter/sec",
            "range": "stddev: 0.000014093182199773866",
            "extra": "mean: 317.7775914970889 usec\nrounds: 2164"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 859.780987535324,
            "unit": "iter/sec",
            "range": "stddev: 0.000024253264263105274",
            "extra": "mean: 1.1630868959624618 msec\nrounds: 644"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 349.2488715314931,
            "unit": "iter/sec",
            "range": "stddev: 0.00005554195881752863",
            "extra": "mean: 2.8632877054545505 msec\nrounds: 275"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 481.28504173087384,
            "unit": "iter/sec",
            "range": "stddev: 0.0010138894884824782",
            "extra": "mean: 2.077770787148591 msec\nrounds: 747"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10356.756031134331,
            "unit": "iter/sec",
            "range": "stddev: 0.000017137354231214537",
            "extra": "mean: 96.55533035574211 usec\nrounds: 6045"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10489.837769369356,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060589950194214046",
            "extra": "mean: 95.33035896132067 usec\nrounds: 7856"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1821.895505880822,
            "unit": "iter/sec",
            "range": "stddev: 0.00001596068716466894",
            "extra": "mean: 548.878899350781 usec\nrounds: 924"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1812.438135542112,
            "unit": "iter/sec",
            "range": "stddev: 0.00001748738581357847",
            "extra": "mean: 551.7429700853726 usec\nrounds: 1404"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1005.0272314710955,
            "unit": "iter/sec",
            "range": "stddev: 0.000020264962072273163",
            "extra": "mean: 994.9979151671971 usec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 367.1567869657589,
            "unit": "iter/sec",
            "range": "stddev: 0.0003197760667404141",
            "extra": "mean: 2.7236320708222674 msec\nrounds: 353"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.04579552478892,
            "unit": "iter/sec",
            "range": "stddev: 0.0008389810819538201",
            "extra": "mean: 47.515428857138886 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 445.7942876785572,
            "unit": "iter/sec",
            "range": "stddev: 0.000109533046398909",
            "extra": "mean: 2.2431871103764713 msec\nrounds: 453"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.4334063963358,
            "unit": "iter/sec",
            "range": "stddev: 0.00007202644903524776",
            "extra": "mean: 2.6707018735969976 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 301.5991150231071,
            "unit": "iter/sec",
            "range": "stddev: 0.00008587924300865062",
            "extra": "mean: 3.315659596426152 msec\nrounds: 280"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 188.2010248323318,
            "unit": "iter/sec",
            "range": "stddev: 0.0002694345750275806",
            "extra": "mean: 5.313467346370189 msec\nrounds: 179"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 846.946883805131,
            "unit": "iter/sec",
            "range": "stddev: 0.00006130726059967038",
            "extra": "mean: 1.180711587847443 msec\nrounds: 757"
          }
        ]
      }
    ]
  }
}