window.BENCHMARK_DATA = {
  "lastUpdate": 1780042561782,
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
          "id": "5a6ee8337581b4effd0531375b0eb68da9a20d56",
          "message": "fix(release): sync README BibTeX DOI/version + gate it (#36)\n\nThe top-level README BibTeX block was orphaned from the release automation:\nsync_release_metadata.py only rewrote the README Zenodo badge (concept DOI),\nnever the BibTeX doi/version, so it drifted to a stale DOI (zenodo.20070143,\nversion 0.11.0) while the actual release is v0.11.1 (zenodo.20440996).\n\n- sync_release_metadata.py: also rewrite README BibTeX version + doi\n  (version_doi), mirroring the demo/README.md handling.\n- check_release_metadata.py: assert release.toml version_doi appears in\n  README.md, closing the gate blind spot that let the drift through G3b.\n- README.md: regenerated via make sync-metadata → 0.11.1 / zenodo.20440996.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-29T10:07:57+02:00",
          "tree_id": "914484c9de5f476254ebd76cbb4a0e46be380697",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/5a6ee8337581b4effd0531375b0eb68da9a20d56"
        },
        "date": 1780042561256,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 598.9479774407857,
            "unit": "iter/sec",
            "range": "stddev: 0.0007929668544040168",
            "extra": "mean: 1.6695940844025368 msec\nrounds: 1481"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 381.37728955910535,
            "unit": "iter/sec",
            "range": "stddev: 0.001478983991002689",
            "extra": "mean: 2.6220753762135627 msec\nrounds: 2060"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 336.6147655477512,
            "unit": "iter/sec",
            "range": "stddev: 0.0018338516175835766",
            "extra": "mean: 2.9707550064619577 msec\nrounds: 2476"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 146.8996701550173,
            "unit": "iter/sec",
            "range": "stddev: 0.005121219968245939",
            "extra": "mean: 6.807367225159459 msec\nrounds: 4388"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 430.5353902585913,
            "unit": "iter/sec",
            "range": "stddev: 0.0011149242438928811",
            "extra": "mean: 2.322689429548109 msec\nrounds: 1902"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 361.7866776516649,
            "unit": "iter/sec",
            "range": "stddev: 0.0005228795398235178",
            "extra": "mean: 2.7640597671836304 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137365.8426029766,
            "unit": "iter/sec",
            "range": "stddev: 8.910268716471298e-7",
            "extra": "mean: 7.279830131354147 usec\nrounds: 28993"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.361929696716494,
            "unit": "iter/sec",
            "range": "stddev: 0.0004470349936935467",
            "extra": "mean: 29.97428533333372 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8286861605570769,
            "unit": "iter/sec",
            "range": "stddev: 0.025435975804790126",
            "extra": "mean: 1.2067294563333348 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.020688184503309403,
            "unit": "iter/sec",
            "range": "stddev: 1.4343344163067988",
            "extra": "mean: 48.336769224000015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3545.1760062548833,
            "unit": "iter/sec",
            "range": "stddev: 0.000010146883243956703",
            "extra": "mean: 282.0734424005081 usec\nrounds: 2283"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 905.1629802517364,
            "unit": "iter/sec",
            "range": "stddev: 0.000047235024116282983",
            "extra": "mean: 1.1047734185084417 msec\nrounds: 724"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 362.69874840423074,
            "unit": "iter/sec",
            "range": "stddev: 0.00007469961081511317",
            "extra": "mean: 2.7571090454535887 msec\nrounds: 264"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 489.1696204393469,
            "unit": "iter/sec",
            "range": "stddev: 0.0005249881300270164",
            "extra": "mean: 2.0442806711950996 msec\nrounds: 736"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 11945.107818182723,
            "unit": "iter/sec",
            "range": "stddev: 0.000004073223987982332",
            "extra": "mean: 83.71628077544933 usec\nrounds: 5467"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 11797.589663383103,
            "unit": "iter/sec",
            "range": "stddev: 0.000004172597029161949",
            "extra": "mean: 84.76307691085076 usec\nrounds: 6423"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1980.300567222504,
            "unit": "iter/sec",
            "range": "stddev: 0.000013835958017188056",
            "extra": "mean: 504.973849198338 usec\nrounds: 935"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1966.6359740160087,
            "unit": "iter/sec",
            "range": "stddev: 0.000013072493876095435",
            "extra": "mean: 508.48251186920464 usec\nrounds: 1348"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1202.3899200781218,
            "unit": "iter/sec",
            "range": "stddev: 0.000027949876475070323",
            "extra": "mean: 831.6769654348299 usec\nrounds: 839"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 389.3497767608921,
            "unit": "iter/sec",
            "range": "stddev: 0.00008652131944085945",
            "extra": "mean: 2.568384675392073 msec\nrounds: 382"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 19.14319058700825,
            "unit": "iter/sec",
            "range": "stddev: 0.0008432472642044194",
            "extra": "mean: 52.2378960526393 msec\nrounds: 19"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 461.8887887118825,
            "unit": "iter/sec",
            "range": "stddev: 0.00011024623489216768",
            "extra": "mean: 2.1650233225811872 msec\nrounds: 434"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 376.78248113348576,
            "unit": "iter/sec",
            "range": "stddev: 0.0001604680616546342",
            "extra": "mean: 2.6540512101084706 msec\nrounds: 376"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 312.49465349906177,
            "unit": "iter/sec",
            "range": "stddev: 0.0001876661039739937",
            "extra": "mean: 3.200054749106299 msec\nrounds: 279"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 210.61131233381647,
            "unit": "iter/sec",
            "range": "stddev: 0.00020649924763991347",
            "extra": "mean: 4.748083039409638 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 978.3191568386217,
            "unit": "iter/sec",
            "range": "stddev: 0.000022715524253151584",
            "extra": "mean: 1.0221613192482488 msec\nrounds: 852"
          }
        ]
      }
    ]
  }
}