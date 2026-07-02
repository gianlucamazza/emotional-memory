window.BENCHMARK_DATA = {
  "lastUpdate": 1782984375968,
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
          "id": "77ab6ec2083be9c0240d13b0dc9e575507bfb736",
          "message": "chore(release): v0.14.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.21129262\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T11:17:30+02:00",
          "tree_id": "e7b149f8c996bb08cc4bea3670dd4d5bb6a773c1",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/77ab6ec2083be9c0240d13b0dc9e575507bfb736"
        },
        "date": 1782984374812,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 580.278107622701,
            "unit": "iter/sec",
            "range": "stddev: 0.0008098526149774653",
            "extra": "mean: 1.7233116101808958 msec\nrounds: 1493"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 440.60925587775785,
            "unit": "iter/sec",
            "range": "stddev: 0.0011332429505292753",
            "extra": "mean: 2.269584641402629 msec\nrounds: 1768"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 310.93232534805037,
            "unit": "iter/sec",
            "range": "stddev: 0.002150442621200859",
            "extra": "mean: 3.2161339252219063 msec\nrounds: 2367"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 214.02898531721505,
            "unit": "iter/sec",
            "range": "stddev: 0.003285290924184042",
            "extra": "mean: 4.672264359511342 msec\nrounds: 3438"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 435.32650308095725,
            "unit": "iter/sec",
            "range": "stddev: 0.0010690485832501853",
            "extra": "mean: 2.2971263934602004 msec\nrounds: 1835"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 374.53617345780754,
            "unit": "iter/sec",
            "range": "stddev: 0.0003496379292426802",
            "extra": "mean: 2.669969073394863 msec\nrounds: 436"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137020.3958938951,
            "unit": "iter/sec",
            "range": "stddev: 0.000001183154242390623",
            "extra": "mean: 7.298183554909396 usec\nrounds: 41143"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.54879345661851,
            "unit": "iter/sec",
            "range": "stddev: 0.0002535399568036617",
            "extra": "mean: 35.02774999999758 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8096522629758477,
            "unit": "iter/sec",
            "range": "stddev: 0.0010707750145839414",
            "extra": "mean: 1.2350981350000012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.025896549522893123,
            "unit": "iter/sec",
            "range": "stddev: 1.991308040514589",
            "extra": "mean: 38.615183042666665 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3240.743900411262,
            "unit": "iter/sec",
            "range": "stddev: 0.00001198939565400917",
            "extra": "mean: 308.5711277195018 usec\nrounds: 2114"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 877.6422883277277,
            "unit": "iter/sec",
            "range": "stddev: 0.000015956228770524894",
            "extra": "mean: 1.1394163810239986 msec\nrounds: 664"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 355.86808681745396,
            "unit": "iter/sec",
            "range": "stddev: 0.000043344471220868725",
            "extra": "mean: 2.8100300000009835 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 488.84792546227965,
            "unit": "iter/sec",
            "range": "stddev: 0.001042018431684764",
            "extra": "mean: 2.045625946053364 msec\nrounds: 760"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10736.600250195996,
            "unit": "iter/sec",
            "range": "stddev: 0.000004820444984134321",
            "extra": "mean: 93.13935293267019 usec\nrounds: 6871"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10541.32141980279,
            "unit": "iter/sec",
            "range": "stddev: 0.000005204367033109711",
            "extra": "mean: 94.86476696568732 usec\nrounds: 8252"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1716.0733753114582,
            "unit": "iter/sec",
            "range": "stddev: 0.00003203253170283213",
            "extra": "mean: 582.7256656892689 usec\nrounds: 1023"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1800.7341689190814,
            "unit": "iter/sec",
            "range": "stddev: 0.000022381329986180675",
            "extra": "mean: 555.3290525942901 usec\nrounds: 1407"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1001.5481166965188,
            "unit": "iter/sec",
            "range": "stddev: 0.000028905198004550502",
            "extra": "mean: 998.4542762642049 usec\nrounds: 771"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 344.339245871569,
            "unit": "iter/sec",
            "range": "stddev: 0.00029713645785500845",
            "extra": "mean: 2.904112766666679 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.040825118940962,
            "unit": "iter/sec",
            "range": "stddev: 0.0010326074690424046",
            "extra": "mean: 47.52665327272739 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 463.94498458886636,
            "unit": "iter/sec",
            "range": "stddev: 0.00009084900264692266",
            "extra": "mean: 2.1554279779232206 msec\nrounds: 453"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 364.9555247883116,
            "unit": "iter/sec",
            "range": "stddev: 0.00013912110894887775",
            "extra": "mean: 2.740059903408885 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 295.2779508797156,
            "unit": "iter/sec",
            "range": "stddev: 0.00021544192711690064",
            "extra": "mean: 3.3866395950687154 msec\nrounds: 284"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 178.66486942797926,
            "unit": "iter/sec",
            "range": "stddev: 0.0005431224959433308",
            "extra": "mean: 5.5970712272739505 msec\nrounds: 176"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 844.801246489129,
            "unit": "iter/sec",
            "range": "stddev: 0.00002654744378138533",
            "extra": "mean: 1.1837103746660582 msec\nrounds: 750"
          }
        ]
      }
    ]
  }
}