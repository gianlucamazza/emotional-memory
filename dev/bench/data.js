window.BENCHMARK_DATA = {
  "lastUpdate": 1782557857908,
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
          "id": "40d60f79d84c6ae8f7f980d63819721108aeeb4a",
          "message": "chore(release): v0.13.0\n\nPrereserved Zenodo DOI: 10.5281/zenodo.20962443\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-06-27T12:50:32+02:00",
          "tree_id": "6d0c0d4df04cdc07f1f7b5cdc5f8587918e2f9ad",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/40d60f79d84c6ae8f7f980d63819721108aeeb4a"
        },
        "date": 1782557856045,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 515.4679035504313,
            "unit": "iter/sec",
            "range": "stddev: 0.0009361575884757313",
            "extra": "mean: 1.939984998313603 msec\nrounds: 1779"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 412.11731712829595,
            "unit": "iter/sec",
            "range": "stddev: 0.0011955482592724785",
            "extra": "mean: 2.426493521233641 msec\nrounds: 1978"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 360.6836691433,
            "unit": "iter/sec",
            "range": "stddev: 0.0016006905833264612",
            "extra": "mean: 2.772512551996633 msec\nrounds: 2404"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 172.09967712516746,
            "unit": "iter/sec",
            "range": "stddev: 0.003802965771044428",
            "extra": "mean: 5.810586148123356 msec\nrounds: 4503"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 419.3722098970629,
            "unit": "iter/sec",
            "range": "stddev: 0.0011024084871583163",
            "extra": "mean: 2.3845166093515244 msec\nrounds: 1989"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 380.6039236797334,
            "unit": "iter/sec",
            "range": "stddev: 0.00025094736710268403",
            "extra": "mean: 2.627403286681483 msec\nrounds: 443"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 145394.94770501487,
            "unit": "iter/sec",
            "range": "stddev: 6.957414101405378e-7",
            "extra": "mean: 6.877818079544648 usec\nrounds: 44459"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.36765047884377,
            "unit": "iter/sec",
            "range": "stddev: 0.0002865965458108821",
            "extra": "mean: 29.969146333333658 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8215854534936834,
            "unit": "iter/sec",
            "range": "stddev: 0.015343635076744246",
            "extra": "mean: 1.2171588429999975 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03267614840519028,
            "unit": "iter/sec",
            "range": "stddev: 0.3199388203178189",
            "extra": "mean: 30.603362048666668 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3615.185900029491,
            "unit": "iter/sec",
            "range": "stddev: 0.000010605336101508312",
            "extra": "mean: 276.610948275673 usec\nrounds: 2494"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 909.2585649290271,
            "unit": "iter/sec",
            "range": "stddev: 0.00004425083748403018",
            "extra": "mean: 1.0997971738413659 msec\nrounds: 604"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 368.01816210736547,
            "unit": "iter/sec",
            "range": "stddev: 0.00004022679392567499",
            "extra": "mean: 2.71725719805171 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 491.3872423527829,
            "unit": "iter/sec",
            "range": "stddev: 0.0004457843545604599",
            "extra": "mean: 2.035054868766958 msec\nrounds: 762"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13099.560340089687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022830697042618518",
            "extra": "mean: 76.33843991997317 usec\nrounds: 7515"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12249.158310888351,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026491328284659027",
            "extra": "mean: 81.63826237032906 usec\nrounds: 7619"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2106.6151543565657,
            "unit": "iter/sec",
            "range": "stddev: 0.000010572237877022403",
            "extra": "mean: 474.6951515714484 usec\nrounds: 1082"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2087.5154883994087,
            "unit": "iter/sec",
            "range": "stddev: 0.000009962953989460163",
            "extra": "mean: 479.0383618982126 usec\nrounds: 1412"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1154.7835817792386,
            "unit": "iter/sec",
            "range": "stddev: 0.00001671101502892959",
            "extra": "mean: 865.9631257133436 usec\nrounds: 875"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 378.1999832683979,
            "unit": "iter/sec",
            "range": "stddev: 0.000038396895638255946",
            "extra": "mean: 2.6441037658384245 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.328881207301826,
            "unit": "iter/sec",
            "range": "stddev: 0.0002580189497678277",
            "extra": "mean: 46.88478454545733 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 475.9642758493132,
            "unit": "iter/sec",
            "range": "stddev: 0.00003347074749469852",
            "extra": "mean: 2.1009980175835565 msec\nrounds: 455"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 390.83074550399584,
            "unit": "iter/sec",
            "range": "stddev: 0.00004101086281980835",
            "extra": "mean: 2.5586523360910354 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 324.8062956362448,
            "unit": "iter/sec",
            "range": "stddev: 0.00003503398113213026",
            "extra": "mean: 3.07875805806398 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 210.78818541353127,
            "unit": "iter/sec",
            "range": "stddev: 0.00010649418670432029",
            "extra": "mean: 4.744098906863147 msec\nrounds: 204"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 959.5127214268845,
            "unit": "iter/sec",
            "range": "stddev: 0.000019772128505367815",
            "extra": "mean: 1.0421956662678815 msec\nrounds: 836"
          }
        ]
      }
    ]
  }
}