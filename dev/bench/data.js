window.BENCHMARK_DATA = {
  "lastUpdate": 1782570487072,
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
          "id": "78a3c1962f7029f9ba101aff9ded665858a5c1e8",
          "message": "feat(bench): add paired condition-significance test to human-eval summary (#83)\n\nThe Gate 2 effectiveness criterion is \"AFT rated higher than baseline, p<0.05\",\nbut `make human-eval-summary` computed only Krippendorff's alpha (agreement) and\ndimension means — no significance test. When real ratings arrive the analyst could\nnot evaluate the effect gate with the provided tool. This closes that latent gap.\n\n- `_paired_condition_significance`: per-dimension paired test of `aft - naive_cosine`,\n  pairs formed by shared (scenario, rater). Uses the repo-standard paired bootstrap\n  (`benchmarks.common.statistics`, numpy-only) rather than scipy/Wilcoxon, matching the\n  project's stats convention. Reports mean Δ, 95% CI, two-sided p, directional flag.\n- Wired into `summarize_ratings` output + rendered in `write_summary` markdown.\n- protocol.md documents the test and that agreement (α) and effect (Δ, p) are distinct gates.\n- New lightweight test file (imports only pipeline → runs in the fast job, unlike the\n  slow benchmark-backed test): aft-advantage detection, tie null, unpaired graceful, determinism.\n\nruff clean; make typecheck green (src scope); 4/4 new tests pass; import chain pulls\nneither tqdm nor httpx.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T16:22:29+02:00",
          "tree_id": "5e5e9bbbcbe024150fc144c02840d55b3ce2c912",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/78a3c1962f7029f9ba101aff9ded665858a5c1e8"
        },
        "date": 1782570486278,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 520.6975482096632,
            "unit": "iter/sec",
            "range": "stddev: 0.0008972567718156953",
            "extra": "mean: 1.9205006888131952 msec\nrounds: 1761"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 452.39985878570394,
            "unit": "iter/sec",
            "range": "stddev: 0.0010438428893286042",
            "extra": "mean: 2.210433934891406 msec\nrounds: 1797"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 358.06323823383246,
            "unit": "iter/sec",
            "range": "stddev: 0.0015561607770457652",
            "extra": "mean: 2.7928027600167993 msec\nrounds: 2371"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 227.66517107023898,
            "unit": "iter/sec",
            "range": "stddev: 0.0026739994573202006",
            "extra": "mean: 4.3924153848349565 msec\nrounds: 3851"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 468.9498599022478,
            "unit": "iter/sec",
            "range": "stddev: 0.0009207628538410689",
            "extra": "mean: 2.1324241363638516 msec\nrounds: 1782"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 389.3576764152134,
            "unit": "iter/sec",
            "range": "stddev: 0.000271010036916502",
            "extra": "mean: 2.5683325655909086 msec\nrounds: 465"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135581.6710017128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011821674486307515",
            "extra": "mean: 7.375628229182741 usec\nrounds: 52258"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.37297233025985,
            "unit": "iter/sec",
            "range": "stddev: 0.0003099995528435047",
            "extra": "mean: 35.24480933333507 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8373542311285425,
            "unit": "iter/sec",
            "range": "stddev: 0.003124732052407366",
            "extra": "mean: 1.1942377106666697 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03873991013470065,
            "unit": "iter/sec",
            "range": "stddev: 1.4783219735609439",
            "extra": "mean: 25.813172940333335 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3241.816869717353,
            "unit": "iter/sec",
            "range": "stddev: 0.00001113396416351182",
            "extra": "mean: 308.4689975369238 usec\nrounds: 2436"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 882.8966292969491,
            "unit": "iter/sec",
            "range": "stddev: 0.000015427685265420678",
            "extra": "mean: 1.1326354261838107 msec\nrounds: 718"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 363.7148134477148,
            "unit": "iter/sec",
            "range": "stddev: 0.00001941513746459167",
            "extra": "mean: 2.7494068512657743 msec\nrounds: 316"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 504.40871019099,
            "unit": "iter/sec",
            "range": "stddev: 0.00042364193164024134",
            "extra": "mean: 1.9825192939696832 msec\nrounds: 796"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10304.07147395033,
            "unit": "iter/sec",
            "range": "stddev: 0.000004965394024592382",
            "extra": "mean: 97.0490162580971 usec\nrounds: 7258"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10284.61898110221,
            "unit": "iter/sec",
            "range": "stddev: 0.000004812692118417462",
            "extra": "mean: 97.23257631979179 usec\nrounds: 8032"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1820.5819432921094,
            "unit": "iter/sec",
            "range": "stddev: 0.000007670531944098085",
            "extra": "mean: 549.2749193105402 usec\nrounds: 1103"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1808.3548515078455,
            "unit": "iter/sec",
            "range": "stddev: 0.000010543263902239272",
            "extra": "mean: 552.9888114416141 usec\nrounds: 1713"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1027.5938000841206,
            "unit": "iter/sec",
            "range": "stddev: 0.000017305324099055317",
            "extra": "mean: 973.1471714972768 usec\nrounds: 828"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 384.3108077424389,
            "unit": "iter/sec",
            "range": "stddev: 0.00003153410613082223",
            "extra": "mean: 2.6020605714273577 msec\nrounds: 371"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 23.208318893487444,
            "unit": "iter/sec",
            "range": "stddev: 0.0005439374714918315",
            "extra": "mean: 43.08799808333438 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.4996155557338,
            "unit": "iter/sec",
            "range": "stddev: 0.0000535693957835105",
            "extra": "mean: 2.1254002488797856 msec\nrounds: 446"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 379.128411125886,
            "unit": "iter/sec",
            "range": "stddev: 0.00006840070981644963",
            "extra": "mean: 2.6376287575767026 msec\nrounds: 363"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 307.9235562757112,
            "unit": "iter/sec",
            "range": "stddev: 0.00002628068312537973",
            "extra": "mean: 3.247559271186813 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 197.12991205752832,
            "unit": "iter/sec",
            "range": "stddev: 0.00005078532995560343",
            "extra": "mean: 5.0727968655927285 msec\nrounds: 186"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 853.0779722899224,
            "unit": "iter/sec",
            "range": "stddev: 0.000019694839819083255",
            "extra": "mean: 1.1722257900009936 msec\nrounds: 800"
          }
        ]
      }
    ]
  }
}