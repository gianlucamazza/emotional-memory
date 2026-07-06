window.BENCHMARK_DATA = {
  "lastUpdate": 1783379454930,
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
          "id": "c031cab70feed8143235c2da1cf55bde689a883d",
          "message": "feat(bench): addendum x2 es-memeval third-party retrieval harness (#103)\n\nHarness for the pre-registered Hx2 study (#102):\n\n- benchmarks/datasets/esmemeval/evo_emo.json vendored byte-identical from\n  ES-MemEval v1.0.0 (CC-BY-4.0, Zenodo 10.5281/zenodo.18338564), sha256-pinned;\n  pre-commit byte-mutating/size hooks exclude it (byte-identity is pinned)\n- benchmarks/esmemeval/: loader (session documents, upstream transcript\n  rendering, composite seeker/session keys, in-family gold resolution,\n  50-candidate pools seeded ex-ante), metrics (upstream-verbatim recall/ndcg\n  incl. their rank-offset quirk + standard grid), adapters (naive_cosine,\n  aft_query_appraised via pool-scoped export/import engines), runner\n  (paired bootstrap, D1 AUC with bootstrap CI, per-seeker D2, per-capability\n  breakdown, zero-gold-inclusive rescale; dry-run writes results.dry.*)\n- tests/test_esmemeval.py: hand-computed metric examples, loader integrity,\n  pool determinism, D1 mapping\n- Makefile: bench-x2-esmem[-dry]; CLAUDE.md command list\n- prereg Amendment A1 (pre-run): exact D1 class counts (7/376/18)\n\nDry-run exercised end to end; naive_cosine recall@4=0.65 matches the\npublished upstream session-level baseline (65.0%).\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T01:04:29+02:00",
          "tree_id": "57f98e0379efb7046258dd411ded349c454d5822",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/c031cab70feed8143235c2da1cf55bde689a883d"
        },
        "date": 1783379454105,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 606.5963269974868,
            "unit": "iter/sec",
            "range": "stddev: 0.0007804093240950502",
            "extra": "mean: 1.6485427878368 msec\nrounds: 1447"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 409.5489572254017,
            "unit": "iter/sec",
            "range": "stddev: 0.0013108940349190812",
            "extra": "mean: 2.4417105265626016 msec\nrounds: 1920"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 321.01515499432116,
            "unit": "iter/sec",
            "range": "stddev: 0.0018625716768703232",
            "extra": "mean: 3.1151177271293946 msec\nrounds: 2536"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 180.41904316867738,
            "unit": "iter/sec",
            "range": "stddev: 0.0037280394574418564",
            "extra": "mean: 5.5426521637468165 msec\nrounds: 4281"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 431.1072398003813,
            "unit": "iter/sec",
            "range": "stddev: 0.0010773677450635968",
            "extra": "mean: 2.3196084585891836 msec\nrounds: 1956"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.1925942589084,
            "unit": "iter/sec",
            "range": "stddev: 0.0002512635759420026",
            "extra": "mean: 2.616481886413976 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135463.32653469674,
            "unit": "iter/sec",
            "range": "stddev: 8.316424894179849e-7",
            "extra": "mean: 7.382071779729004 usec\nrounds: 47799"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.77683081871588,
            "unit": "iter/sec",
            "range": "stddev: 0.0005237515108473012",
            "extra": "mean: 30.509356000000782 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8445527117229198,
            "unit": "iter/sec",
            "range": "stddev: 0.004549369567172239",
            "extra": "mean: 1.184058716666674 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03149064093918945,
            "unit": "iter/sec",
            "range": "stddev: 0.9966208904025412",
            "extra": "mean: 31.755466709333334 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3448.507886175885,
            "unit": "iter/sec",
            "range": "stddev: 0.000008099811132088981",
            "extra": "mean: 289.9804880855061 usec\nrounds: 2434"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 885.1123532387099,
            "unit": "iter/sec",
            "range": "stddev: 0.000021204882819713158",
            "extra": "mean: 1.1298000715286656 msec\nrounds: 713"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.3926280977183,
            "unit": "iter/sec",
            "range": "stddev: 0.00003484615255142344",
            "extra": "mean: 2.8137893724825416 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 501.2486503426842,
            "unit": "iter/sec",
            "range": "stddev: 0.00046479177104903",
            "extra": "mean: 1.995017840579399 msec\nrounds: 759"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12310.393500910031,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034437566415260817",
            "extra": "mean: 81.2321718169347 usec\nrounds: 7281"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12139.008799009043,
            "unit": "iter/sec",
            "range": "stddev: 0.000003471603597683792",
            "extra": "mean: 82.37904894521817 usec\nrounds: 8152"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2001.7639147907062,
            "unit": "iter/sec",
            "range": "stddev: 0.000010848904903682265",
            "extra": "mean: 499.5594098840345 usec\nrounds: 344"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2063.4970733380774,
            "unit": "iter/sec",
            "range": "stddev: 0.000014949139115373205",
            "extra": "mean: 484.6142080455294 usec\nrounds: 1591"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1160.7552416258823,
            "unit": "iter/sec",
            "range": "stddev: 0.000018284611983175002",
            "extra": "mean: 861.5080631462748 usec\nrounds: 871"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 387.08223585205013,
            "unit": "iter/sec",
            "range": "stddev: 0.00004725269461795075",
            "extra": "mean: 2.583430360214769 msec\nrounds: 372"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.092394595250862,
            "unit": "iter/sec",
            "range": "stddev: 0.0005635664266360097",
            "extra": "mean: 49.77007570000467 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.69376031833156,
            "unit": "iter/sec",
            "range": "stddev: 0.00004153979430251923",
            "extra": "mean: 2.120019563805828 msec\nrounds: 431"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 384.8941918284813,
            "unit": "iter/sec",
            "range": "stddev: 0.000058324782423912",
            "extra": "mean: 2.5981166284931247 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 314.87394593649003,
            "unit": "iter/sec",
            "range": "stddev: 0.00006198107711796261",
            "extra": "mean: 3.1758740693067686 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 205.78251944966127,
            "unit": "iter/sec",
            "range": "stddev: 0.00008301417550664435",
            "extra": "mean: 4.85949925520579 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 955.6752334471356,
            "unit": "iter/sec",
            "range": "stddev: 0.000019794563485258058",
            "extra": "mean: 1.0463805746989845 msec\nrounds: 830"
          }
        ]
      }
    ]
  }
}