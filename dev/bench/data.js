window.BENCHMARK_DATA = {
  "lastUpdate": 1783377653617,
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
          "id": "612185aab256798170cf8fe6c3b2a7c6c742d5a4",
          "message": "docs(bench): pre-register addendum x2 es-memeval third-party replication (#102)\n\nPre-registration for Hx2: oracle-free query-appraised AFT vs naive cosine\non ES-MemEval/EvoEmo (WWW 2026, CC-BY-4.0, Zenodo 10.5281/zenodo.18338564),\nthe second released third-party corpus, reserved by the Addendum X closure.\n\n- Primary: upstream-verbatim nDCG@4, N=1,133 in-family queries (>=1\n  session-resolvable gold), paired bootstrap n=10k seed=0 one-tailed\n- Ex-ante regime prior declared: near-uniformly negative bank (393/401)\n  -> low D2 expected; three readings fixed before the run\n- Session-level documents, 50-candidate pools shared by both arms,\n  time-invariant DecayConfig (X Amendment A1 pattern, from the start)\n- Artifact-vs-paper QA count discrepancy (1,427 vs 1,209) declared;\n  hash-pinned artifact is ground truth\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T00:34:29+02:00",
          "tree_id": "2f2a24ffc5bdf7a73417ea169269e82d4b8f5aae",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/612185aab256798170cf8fe6c3b2a7c6c742d5a4"
        },
        "date": 1783377653011,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 618.9395136712212,
            "unit": "iter/sec",
            "range": "stddev: 0.0007669351562464963",
            "extra": "mean: 1.6156667621178842 msec\nrounds: 1341"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 420.448593757337,
            "unit": "iter/sec",
            "range": "stddev: 0.0012367536790461075",
            "extra": "mean: 2.3784120457236027 msec\nrounds: 1859"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 310.2119426816783,
            "unit": "iter/sec",
            "range": "stddev: 0.001995727718074333",
            "extra": "mean: 3.2236025194753473 msec\nrounds: 2516"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 177.01574294844954,
            "unit": "iter/sec",
            "range": "stddev: 0.003878583475865559",
            "extra": "mean: 5.649215054794418 msec\nrounds: 4307"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 433.56452904260215,
            "unit": "iter/sec",
            "range": "stddev: 0.001062729444085833",
            "extra": "mean: 2.306461744479424 msec\nrounds: 1902"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.5931052495744,
            "unit": "iter/sec",
            "range": "stddev: 0.0002643752709172993",
            "extra": "mean: 2.627477971111034 msec\nrounds: 450"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137338.7060937998,
            "unit": "iter/sec",
            "range": "stddev: 8.405788446447834e-7",
            "extra": "mean: 7.281268539962933 usec\nrounds: 47937"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 32.745891326623074,
            "unit": "iter/sec",
            "range": "stddev: 0.0004080317299431694",
            "extra": "mean: 30.538182333335346 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8452491222474576,
            "unit": "iter/sec",
            "range": "stddev: 0.002291382374347088",
            "extra": "mean: 1.1830831569999987 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.032514704643382034,
            "unit": "iter/sec",
            "range": "stddev: 0.3577592311325427",
            "extra": "mean: 30.755315509333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3458.387308748395,
            "unit": "iter/sec",
            "range": "stddev: 0.000009458490066175113",
            "extra": "mean: 289.1521136080922 usec\nrounds: 2447"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 887.7291939581596,
            "unit": "iter/sec",
            "range": "stddev: 0.000014640120707945999",
            "extra": "mean: 1.1264696562937773 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 356.4008561924577,
            "unit": "iter/sec",
            "range": "stddev: 0.000038749704447078765",
            "extra": "mean: 2.805829398625228 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 493.1932289257107,
            "unit": "iter/sec",
            "range": "stddev: 0.00045652276457262337",
            "extra": "mean: 2.027602856953718 msec\nrounds: 755"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 11997.892804191155,
            "unit": "iter/sec",
            "range": "stddev: 0.000004074118938846043",
            "extra": "mean: 83.34796920761583 usec\nrounds: 7372"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11820.041986099484,
            "unit": "iter/sec",
            "range": "stddev: 0.000013362756493743284",
            "extra": "mean: 84.60206834933517 usec\nrounds: 8676"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1956.9721136663168,
            "unit": "iter/sec",
            "range": "stddev: 0.00004048931427388204",
            "extra": "mean: 510.9934847904072 usec\nrounds: 1052"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1999.5159900168148,
            "unit": "iter/sec",
            "range": "stddev: 0.000013628550398360102",
            "extra": "mean: 500.1210317860927 usec\nrounds: 1573"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1160.1752635767502,
            "unit": "iter/sec",
            "range": "stddev: 0.000023456759234602956",
            "extra": "mean: 861.9387358053648 usec\nrounds: 863"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 382.42507135980765,
            "unit": "iter/sec",
            "range": "stddev: 0.00004501184897475297",
            "extra": "mean: 2.6148913209174567 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.205817569079002,
            "unit": "iter/sec",
            "range": "stddev: 0.0019401338476328252",
            "extra": "mean: 49.490697249999016 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 471.0644709033828,
            "unit": "iter/sec",
            "range": "stddev: 0.000059941916565395875",
            "extra": "mean: 2.122851672685593 msec\nrounds: 443"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 386.985928240943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000865326498586558",
            "extra": "mean: 2.5840732880017945 msec\nrounds: 375"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 318.8983451720789,
            "unit": "iter/sec",
            "range": "stddev: 0.00008842067488396926",
            "extra": "mean: 3.1357955133332402 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.5393838678257,
            "unit": "iter/sec",
            "range": "stddev: 0.00014034385047812818",
            "extra": "mean: 4.795257286430892 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 961.0042977564776,
            "unit": "iter/sec",
            "range": "stddev: 0.00003271126276330913",
            "extra": "mean: 1.0405780726835043 msec\nrounds: 853"
          }
        ]
      }
    ]
  }
}