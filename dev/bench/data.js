window.BENCHMARK_DATA = {
  "lastUpdate": 1789208695818,
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
          "id": "739162a2e4b71e4e2fa9ffdb7767b2429dd34003",
          "message": "fix(release): avoid exporting process environment (#135)",
          "timestamp": "2026-09-12T12:20:14+02:00",
          "tree_id": "b845973eb9bafc01d092eea5ddacd51b509870b8",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/739162a2e4b71e4e2fa9ffdb7767b2429dd34003"
        },
        "date": 1789208694459,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 554.7017884958071,
            "unit": "iter/sec",
            "range": "stddev: 0.0007369025465755566",
            "extra": "mean: 1.8027704628674706 msec\nrounds: 1737"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 504.29937466272185,
            "unit": "iter/sec",
            "range": "stddev: 0.0008868490734305336",
            "extra": "mean: 1.9829491176125398 msec\nrounds: 1726"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 453.9254718657407,
            "unit": "iter/sec",
            "range": "stddev: 0.0009791394900733642",
            "extra": "mean: 2.203004814622463 msec\nrounds: 2120"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 358.9392396969804,
            "unit": "iter/sec",
            "range": "stddev: 0.001406605559595502",
            "extra": "mean: 2.785986845139051 msec\nrounds: 2880"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 494.98959984753157,
            "unit": "iter/sec",
            "range": "stddev: 0.0010798377652892423",
            "extra": "mean: 2.020244466364593 msec\nrounds: 1769"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 410.6465028708798,
            "unit": "iter/sec",
            "range": "stddev: 0.00029487815066689193",
            "extra": "mean: 2.435184502994371 msec\nrounds: 501"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 196530.28242511424,
            "unit": "iter/sec",
            "range": "stddev: 6.774377035283728e-7",
            "extra": "mean: 5.088274375126079 usec\nrounds: 51364"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 15.824012422987877,
            "unit": "iter/sec",
            "range": "stddev: 0.0011827773922173236",
            "extra": "mean: 63.19509700000481 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.6819225260385142,
            "unit": "iter/sec",
            "range": "stddev: 0.032880458580431426",
            "extra": "mean: 1.466442245000015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.04425275032943091,
            "unit": "iter/sec",
            "range": "stddev: 0.39981541171597373",
            "extra": "mean: 22.597465525999997 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4680.415649784661,
            "unit": "iter/sec",
            "range": "stddev: 0.000011224250042584446",
            "extra": "mean: 213.65623799800954 usec\nrounds: 3416"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1213.309530467419,
            "unit": "iter/sec",
            "range": "stddev: 0.00003461381906477072",
            "extra": "mean: 824.1919929655188 usec\nrounds: 853"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 462.88469536905603,
            "unit": "iter/sec",
            "range": "stddev: 0.00009204629733125142",
            "extra": "mean: 2.1603652270306846 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 527.0433060078853,
            "unit": "iter/sec",
            "range": "stddev: 0.0008637478682166651",
            "extra": "mean: 1.8973772906339472 msec\nrounds: 726"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 15460.866747210435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037385930106071603",
            "extra": "mean: 64.67942686204364 usec\nrounds: 9359"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 15090.028274988645,
            "unit": "iter/sec",
            "range": "stddev: 0.0000039141694429369206",
            "extra": "mean: 66.268928180703 usec\nrounds: 10596"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2677.394997062543,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170821366029912",
            "extra": "mean: 373.4973737894978 usec\nrounds: 1137"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2688.91755265165,
            "unit": "iter/sec",
            "range": "stddev: 0.000015002836015159365",
            "extra": "mean: 371.89686199707376 usec\nrounds: 1942"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1608.5964684180572,
            "unit": "iter/sec",
            "range": "stddev: 0.000024870696394456872",
            "extra": "mean: 621.6599499210828 usec\nrounds: 639"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 488.5815356925704,
            "unit": "iter/sec",
            "range": "stddev: 0.00005011619589650676",
            "extra": "mean: 2.046741284609717 msec\nrounds: 1293"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 37.98736297895756,
            "unit": "iter/sec",
            "range": "stddev: 0.0010819064288388083",
            "extra": "mean: 26.324543784572054 msec\nrounds: 376"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 598.0880093543685,
            "unit": "iter/sec",
            "range": "stddev: 0.00005113343380387022",
            "extra": "mean: 1.6719947304736849 msec\nrounds: 3124"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 497.6303286556391,
            "unit": "iter/sec",
            "range": "stddev: 0.00005044390505185691",
            "extra": "mean: 2.009523822033768 msec\nrounds: 1416"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 411.2703029655276,
            "unit": "iter/sec",
            "range": "stddev: 0.00006702459378087461",
            "extra": "mean: 2.431490902186096 msec\nrounds: 869"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 273.09728992489295,
            "unit": "iter/sec",
            "range": "stddev: 0.00007413800321660564",
            "extra": "mean: 3.6616987311555507 msec\nrounds: 398"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1298.2874183858523,
            "unit": "iter/sec",
            "range": "stddev: 0.000020533676383166996",
            "extra": "mean: 770.245467866653 usec\nrounds: 1556"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[15]",
            "value": 7512.546676593847,
            "unit": "iter/sec",
            "range": "stddev: 0.000006904273942684034",
            "extra": "mean: 133.11065382337102 usec\nrounds: 6722"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_cosine_rank_fixed_pool[1000]",
            "value": 111.83634836674797,
            "unit": "iter/sec",
            "range": "stddev: 0.00022019645061586876",
            "extra": "mean: 8.941636727271108 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[15]",
            "value": 3031.1609304192166,
            "unit": "iter/sec",
            "range": "stddev: 0.000013470921949499295",
            "extra": "mean: 329.90660111922784 usec\nrounds: 2324"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_aft_plan_fixed_pool[1000]",
            "value": 40.574764721929945,
            "unit": "iter/sec",
            "range": "stddev: 0.006190119426798569",
            "extra": "mean: 24.64586071794318 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/perf/bench_scoring.py::bench_inmemory_search_cached",
            "value": 31143.308444613376,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029867860529677315",
            "extra": "mean: 32.1096264315798 usec\nrounds: 21921"
          }
        ]
      }
    ]
  }
}