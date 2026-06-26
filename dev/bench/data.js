window.BENCHMARK_DATA = {
  "lastUpdate": 1782505390631,
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
          "id": "eaef8b6587ac73ee8c6084ddb32477d02c969386",
          "message": "docs(research): re-scope A7 from \"deterministic\" to near-deterministic (#60) (#64)\n\nA fresh `make bench-multiseed` falsifies the committed headline that cross-seed\nstdev = 0.0000 and retrieval is exactly deterministic. Repeating the sweep six\ntimes gave retrieval_deterministic=True in 5/6 runs and False in 1/6 (cross-seed\nstdev 0.0024), with the absolute aft mean drifting across sweeps (0.120-0.125).\n\nRoot cause (already documented, mislabelled as \"resolved\"): the engine stamps\nencode/retrieve with real wall-clock time and ACT-R decay tracks now - encoded_at,\nso a near-tie query can flip between seeds even with subprocess isolation — the\nisolation removes RNG coupling but not wall-clock timing. The variance is sub-CI\nand does not change the AFT-vs-baseline conclusion (Δ ≈ +0.075).\n\n- multiseed_runner.py: soften docstring + generated markdown; add a standing\n  \"near-deterministic, not bit-stable\" caveat to the artifact.\n- problem_register §A7: Resolved/deterministic -> Characterized/near-deterministic.\n- 08_limitations §2.9: drop \"exactly 0.0000\"; document timing-driven near-tie flips.\n- Regenerate multiseed_results.{md,json}.\n\nNo clock injection (sub-CI, benchmark-only effect; disproportionate per register).\n\nCloses #60\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-26T22:16:18+02:00",
          "tree_id": "c3552917aba9e00184b9f47e43c8d40f1ce77adc",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/eaef8b6587ac73ee8c6084ddb32477d02c969386"
        },
        "date": 1782505389315,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 601.3833421697113,
            "unit": "iter/sec",
            "range": "stddev: 0.0007684830807480777",
            "extra": "mean: 1.6628328885734227 msec\nrounds: 1409"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 392.6475131929146,
            "unit": "iter/sec",
            "range": "stddev: 0.0014750372489936358",
            "extra": "mean: 2.5468135322397485 msec\nrounds: 1768"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 292.66101206066634,
            "unit": "iter/sec",
            "range": "stddev: 0.0023795962106890247",
            "extra": "mean: 3.4169225103093264 msec\nrounds: 2328"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 159.52858127360398,
            "unit": "iter/sec",
            "range": "stddev: 0.004194153913858307",
            "extra": "mean: 6.268469211074609 msec\nrounds: 3648"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 486.0024506976089,
            "unit": "iter/sec",
            "range": "stddev: 0.000913677721454024",
            "extra": "mean: 2.0576027930818004 msec\nrounds: 1590"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 370.2644450265167,
            "unit": "iter/sec",
            "range": "stddev: 0.0003887125683686607",
            "extra": "mean: 2.7007724166666462 msec\nrounds: 456"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 133376.43215803523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015087243362494004",
            "extra": "mean: 7.497576474493776 usec\nrounds: 36777"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.51739619977548,
            "unit": "iter/sec",
            "range": "stddev: 0.00011822910481949539",
            "extra": "mean: 36.34064766666256 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8121438360465033,
            "unit": "iter/sec",
            "range": "stddev: 0.0018816804109836685",
            "extra": "mean: 1.2313089820000063 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.025541112424489094,
            "unit": "iter/sec",
            "range": "stddev: 6.2467582590203765",
            "extra": "mean: 39.15256247966667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3124.7716798093597,
            "unit": "iter/sec",
            "range": "stddev: 0.000017411666748324805",
            "extra": "mean: 320.0233816958458 usec\nrounds: 2240"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 852.0326178452123,
            "unit": "iter/sec",
            "range": "stddev: 0.00002932056420382944",
            "extra": "mean: 1.1736639878048294 msec\nrounds: 656"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 345.2537841027533,
            "unit": "iter/sec",
            "range": "stddev: 0.00006578078582225712",
            "extra": "mean: 2.8964201003583594 msec\nrounds: 279"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 491.85344736816785,
            "unit": "iter/sec",
            "range": "stddev: 0.001051667730666546",
            "extra": "mean: 2.033125934871141 msec\nrounds: 737"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10638.888165513039,
            "unit": "iter/sec",
            "range": "stddev: 0.000005457943462712185",
            "extra": "mean: 93.99478445892443 usec\nrounds: 7168"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10540.529744272862,
            "unit": "iter/sec",
            "range": "stddev: 0.000005885274317561236",
            "extra": "mean: 94.87189204540164 usec\nrounds: 8096"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1840.1983895314747,
            "unit": "iter/sec",
            "range": "stddev: 0.00004612060783752961",
            "extra": "mean: 543.4196691448067 usec\nrounds: 1076"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1830.8680851523836,
            "unit": "iter/sec",
            "range": "stddev: 0.000011208211282532734",
            "extra": "mean: 546.1889953239147 usec\nrounds: 1497"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1011.5577519088723,
            "unit": "iter/sec",
            "range": "stddev: 0.000024407800124933274",
            "extra": "mean: 988.574303457156 usec\nrounds: 781"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.4333944544076,
            "unit": "iter/sec",
            "range": "stddev: 0.00009364355264168824",
            "extra": "mean: 2.699540632595635 msec\nrounds: 362"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.784289279684344,
            "unit": "iter/sec",
            "range": "stddev: 0.0011189617960928122",
            "extra": "mean: 48.11326413636152 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 474.4468291517946,
            "unit": "iter/sec",
            "range": "stddev: 0.00003355820643612752",
            "extra": "mean: 2.1077177431826817 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 378.8419503432872,
            "unit": "iter/sec",
            "range": "stddev: 0.00004697120298682882",
            "extra": "mean: 2.639623196675688 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 306.4093807679044,
            "unit": "iter/sec",
            "range": "stddev: 0.000045702640049231806",
            "extra": "mean: 3.2636076529180063 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 191.8977497369638,
            "unit": "iter/sec",
            "range": "stddev: 0.00014383348346017042",
            "extra": "mean: 5.211108527175073 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 845.4666281995602,
            "unit": "iter/sec",
            "range": "stddev: 0.00002492432623320105",
            "extra": "mean: 1.1827787953375781 msec\nrounds: 772"
          }
        ]
      }
    ]
  }
}