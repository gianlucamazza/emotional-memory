window.BENCHMARK_DATA = {
  "lastUpdate": 1780234341754,
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
          "id": "e5bed45d9d68ffb9d73e5a3c20413ae5c52ea7ec",
          "message": "chore(release): mint Zenodo DOI for v0.11.3 (#49)\n\n* docs(paper): add Addendum O/P to Hg1 affect-free scope discussion\n\nRecord the recalibrated SEC->affect mapping (Addendum O, held-out\ncalibration PASS) and the leakage-free Hg1 re-run (Addendum P,\nrealistic_recall_v4_noAF, N=160): Hp1 FAIL (naive cosine 0.887 vs\ndual 0.800, delta -0.087, p=0.0018, d=-0.24), with dual>neutral and\ndual>sync PASS showing the inferred affect carries signal and the\ndeferred dual-path schedule is essential. Updates abstract, oracle-\naffect operationalization paragraph, and conclusion study list.\nRegenerates the arXiv bundle to match.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* chore(release): mint Zenodo DOI for v0.11.3\n\nReserve a Zenodo version DOI for v0.11.3 (record 20475352, draft) and\npropagate via sync-metadata to release.toml, CITATION.cff, README.md,\ndemo/README.md, paper/SUBMISSION.md, and paper/main.tex. Regenerate the\narXiv bundle to match. Previous version_doi pointed at v0.11.1\n(20440996); v0.11.2/0.11.3 had never been deposited.\n\nThe deposit draft is reserved but NOT yet published.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-31T15:26:21+02:00",
          "tree_id": "b2b35244a040b246a14143e79ed9c1419e106eb3",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/e5bed45d9d68ffb9d73e5a3c20413ae5c52ea7ec"
        },
        "date": 1780234340252,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 695.8114649020056,
            "unit": "iter/sec",
            "range": "stddev: 0.0006931810246326334",
            "extra": "mean: 1.4371709154588803 msec\nrounds: 1656"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 389.3938294691411,
            "unit": "iter/sec",
            "range": "stddev: 0.0015565050639620357",
            "extra": "mean: 2.5680941101796493 msec\nrounds: 2505"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 320.3085641548759,
            "unit": "iter/sec",
            "range": "stddev: 0.001980922112038552",
            "extra": "mean: 3.1219895810106375 msec\nrounds: 3265"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 172.88785727920612,
            "unit": "iter/sec",
            "range": "stddev: 0.00422151964378007",
            "extra": "mean: 5.784096209747368 msec\nrounds: 5540"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 441.3181113937364,
            "unit": "iter/sec",
            "range": "stddev: 0.0016217433843749049",
            "extra": "mean: 2.265939181244745 msec\nrounds: 2378"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 468.76285740827734,
            "unit": "iter/sec",
            "range": "stddev: 0.00029127006182366944",
            "extra": "mean: 2.133274819444648 msec\nrounds: 576"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 175357.23980344782,
            "unit": "iter/sec",
            "range": "stddev: 6.044351320396851e-7",
            "extra": "mean: 5.702644505130596 usec\nrounds: 58600"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 43.21595567657354,
            "unit": "iter/sec",
            "range": "stddev: 0.0006341908452584903",
            "extra": "mean: 23.139601666661253 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.0957848601577327,
            "unit": "iter/sec",
            "range": "stddev: 0.0016711384247021085",
            "extra": "mean: 912.5878960000003 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03931066865198454,
            "unit": "iter/sec",
            "range": "stddev: 2.9414691815075034",
            "extra": "mean: 25.438386938999994 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4521.630589840679,
            "unit": "iter/sec",
            "range": "stddev: 0.000032799076404452706",
            "extra": "mean: 221.15915489576415 usec\nrounds: 2931"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1165.286345295498,
            "unit": "iter/sec",
            "range": "stddev: 0.000014681105741863068",
            "extra": "mean: 858.1581720554839 usec\nrounds: 866"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 464.80960342684386,
            "unit": "iter/sec",
            "range": "stddev: 0.00002767585650780633",
            "extra": "mean: 2.1514185434797914 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 569.1241714644846,
            "unit": "iter/sec",
            "range": "stddev: 0.0004901045981886641",
            "extra": "mean: 1.7570857997944018 msec\nrounds: 974"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 15638.552757331558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026620969372347333",
            "extra": "mean: 63.94453601412617 usec\nrounds: 7983"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 15540.692080432145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027670164318465814",
            "extra": "mean: 64.34719862052583 usec\nrounds: 7975"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2600.870211797699,
            "unit": "iter/sec",
            "range": "stddev: 0.000008637691257001288",
            "extra": "mean: 384.4866981304725 usec\nrounds: 1070"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2608.3676479947385,
            "unit": "iter/sec",
            "range": "stddev: 0.000009351517572495009",
            "extra": "mean: 383.3815377862013 usec\nrounds: 2051"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1552.264985921003,
            "unit": "iter/sec",
            "range": "stddev: 0.000014887569010921206",
            "extra": "mean: 644.2199038630453 usec\nrounds: 1113"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 495.44546337777757,
            "unit": "iter/sec",
            "range": "stddev: 0.000050733881965854336",
            "extra": "mean: 2.018385622470619 msec\nrounds: 445"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.741048416322563,
            "unit": "iter/sec",
            "range": "stddev: 0.0005465449337742052",
            "extra": "mean: 45.99594190909526 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 604.0463953989947,
            "unit": "iter/sec",
            "range": "stddev: 0.00005582775776417336",
            "extra": "mean: 1.655501974048638 msec\nrounds: 578"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 501.59204843957514,
            "unit": "iter/sec",
            "range": "stddev: 0.00005973377907632974",
            "extra": "mean: 1.9936520188287357 msec\nrounds: 478"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 412.84188218507796,
            "unit": "iter/sec",
            "range": "stddev: 0.00007858497491190075",
            "extra": "mean: 2.422234863156877 msec\nrounds: 380"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 272.22931736855736,
            "unit": "iter/sec",
            "range": "stddev: 0.00016271924561936417",
            "extra": "mean: 3.6733736456685566 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1272.209949025961,
            "unit": "iter/sec",
            "range": "stddev: 0.00001687990738004366",
            "extra": "mean: 786.0337837835866 usec\nrounds: 1110"
          }
        ]
      }
    ]
  }
}