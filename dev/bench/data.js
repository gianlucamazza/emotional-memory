window.BENCHMARK_DATA = {
  "lastUpdate": 1784022115316,
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
          "id": "a78d7fe60687afa75b2b6201e0ed36f0e28acd08",
          "message": "chore(metadata): add ORCID iD and Public API credential config (#120)\n\nRegistered the ORCID Public API (client-credentials, scope /read-public) and\nwired the credentials + identity into the repo:\n\n- add ORCID iD 0009-0005-1462-3019 to CITATION.cff, codemeta.json\n  (author + maintainer @id), and .zenodo.json (creators.orcid)\n- add ORCID_CLIENT_ID / ORCID_CLIENT_SECRET placeholders to .env.example\n  and document them in CONTRIBUTING.md\n\nSSOT guard (scripts/check_metadata_ssot.py) unaffected: it validates\nname/email/license/keywords only, not ORCID.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-14T11:35:48+02:00",
          "tree_id": "70636c67261955585c9d75806ecd8557be93a747",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/a78d7fe60687afa75b2b6201e0ed36f0e28acd08"
        },
        "date": 1784022114197,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 552.1971394521554,
            "unit": "iter/sec",
            "range": "stddev: 0.0008391000869376733",
            "extra": "mean: 1.810947447123898 msec\nrounds: 1617"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 428.2688251720611,
            "unit": "iter/sec",
            "range": "stddev: 0.00115905621993263",
            "extra": "mean: 2.3349820048149437 msec\nrounds: 1869"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 386.97002322892075,
            "unit": "iter/sec",
            "range": "stddev: 0.001406653590793479",
            "extra": "mean: 2.5841794970470557 msec\nrounds: 2201"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 214.61262152280608,
            "unit": "iter/sec",
            "range": "stddev: 0.0029675613195963403",
            "extra": "mean: 4.659558197949386 msec\nrounds: 3804"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 471.5035403526443,
            "unit": "iter/sec",
            "range": "stddev: 0.0009176351949639034",
            "extra": "mean: 2.1208748491094798 msec\nrounds: 1743"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 384.5536070543183,
            "unit": "iter/sec",
            "range": "stddev: 0.00032900543409008587",
            "extra": "mean: 2.600417683401809 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 140096.07433920103,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012214932990499637",
            "extra": "mean: 7.137958752354453 usec\nrounds: 41263"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.418898120133523,
            "unit": "iter/sec",
            "range": "stddev: 0.00021330555923260167",
            "extra": "mean: 35.187852666657214 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8261809239692467,
            "unit": "iter/sec",
            "range": "stddev: 0.0057994732526354086",
            "extra": "mean: 1.210388634000007 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.031619655458785666,
            "unit": "iter/sec",
            "range": "stddev: 0.20431487627569064",
            "extra": "mean: 31.625898052666646 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3193.7851313380474,
            "unit": "iter/sec",
            "range": "stddev: 0.000014595755756907942",
            "extra": "mean: 313.1081017905066 usec\nrounds: 2122"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 879.1574353766944,
            "unit": "iter/sec",
            "range": "stddev: 0.0000214508183094207",
            "extra": "mean: 1.1374527015989213 msec\nrounds: 687"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 361.77920495862986,
            "unit": "iter/sec",
            "range": "stddev: 0.00007753000094632301",
            "extra": "mean: 2.7641168599349206 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 502.3357186747189,
            "unit": "iter/sec",
            "range": "stddev: 0.00043422496224735665",
            "extra": "mean: 1.9907005670196771 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10440.614655780484,
            "unit": "iter/sec",
            "range": "stddev: 0.000005898103419026735",
            "extra": "mean: 95.77980157004897 usec\nrounds: 6496"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10606.179264134907,
            "unit": "iter/sec",
            "range": "stddev: 0.00000546011145366318",
            "extra": "mean: 94.2846594514509 usec\nrounds: 8222"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1810.938314817355,
            "unit": "iter/sec",
            "range": "stddev: 0.000009651746136881302",
            "extra": "mean: 552.1999241044588 usec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1791.8285623299212,
            "unit": "iter/sec",
            "range": "stddev: 0.000011492708398894347",
            "extra": "mean: 558.0891057455276 usec\nrounds: 1532"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1029.8484420036893,
            "unit": "iter/sec",
            "range": "stddev: 0.000018892400729076534",
            "extra": "mean: 971.0166653788244 usec\nrounds: 777"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 373.59925903924085,
            "unit": "iter/sec",
            "range": "stddev: 0.00003657885610901501",
            "extra": "mean: 2.6766648375364297 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.731789421902686,
            "unit": "iter/sec",
            "range": "stddev: 0.0016373023477815267",
            "extra": "mean: 43.99125741664989 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.9926538261913,
            "unit": "iter/sec",
            "range": "stddev: 0.00002758146792185917",
            "extra": "mean: 2.1231753656374996 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 379.3694473499599,
            "unit": "iter/sec",
            "range": "stddev: 0.000035250896133458356",
            "extra": "mean: 2.635952913407711 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.8259381743761,
            "unit": "iter/sec",
            "range": "stddev: 0.000037845088231681524",
            "extra": "mean: 3.2591768673471058 msec\nrounds: 294"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 190.39266274571114,
            "unit": "iter/sec",
            "range": "stddev: 0.00027501203958716914",
            "extra": "mean: 5.252303243090844 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 854.8476246334367,
            "unit": "iter/sec",
            "range": "stddev: 0.0000487643503716709",
            "extra": "mean: 1.1697991211343723 msec\nrounds: 776"
          }
        ]
      }
    ]
  }
}