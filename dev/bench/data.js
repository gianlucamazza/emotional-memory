window.BENCHMARK_DATA = {
  "lastUpdate": 1783417283500,
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
          "id": "3ef78c2a019b926c59cc92c188e5eca015ab8cf5",
          "message": "docs(paper): sharpen abstract boundary claim after Addendum X2 + refresh dated review docs (#109)\n\nAddendum X2 falsified the discrimination-based framing of the paper's central\nboundary claim: X2's corpus was affect-discriminative (D2=68.2%) yet AFT still\nfailed, because the gold relation is affect-orthogonal. The precise boundary that\nsurvives X/X2 is 'relevance is itself affect-conditioned', not 'affect\ndiscriminates among candidate memories'.\n\n- paper/main.tex abstract (L79): boundary claim corrected (abstract ~1890->1887\n  chars, still <1920; page count 19pp unchanged); bundle + sha256 regenerated,\n  check-arxiv-bundle green\n- README.md 'When NOT to use' headline: same refinement (oracle-affect condition\n  preserved), consistent with the X2 affect-orthogonal bullet\n- validation_report_2026-06.md, review_response_2026-06.md: appended\n  Update (2026-07-07) blocks — the two dated review docs mentioned X but not X2,\n  understating the third-party bound; now note both corpora + the sharpened boundary\n- ARXIV_CHECKLIST footer updated\n\nZenodo v0.15.0 keeps the prior abstract (immutable snapshot); the pending arXiv\nupload uses this sharpened bundle.\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T11:34:15+02:00",
          "tree_id": "f2a902a027d872907e7b10801bb52b681f56baae",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/3ef78c2a019b926c59cc92c188e5eca015ab8cf5"
        },
        "date": 1783417282897,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 647.3746250777513,
            "unit": "iter/sec",
            "range": "stddev: 0.0006649718202883749",
            "extra": "mean: 1.54470064358778 msec\nrounds: 1271"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 429.9291950100485,
            "unit": "iter/sec",
            "range": "stddev: 0.001183619388449476",
            "extra": "mean: 2.3259643950828406 msec\nrounds: 1749"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 302.2963854374298,
            "unit": "iter/sec",
            "range": "stddev: 0.002308640814163369",
            "extra": "mean: 3.3080117665084776 msec\nrounds: 2317"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 175.98873159730923,
            "unit": "iter/sec",
            "range": "stddev: 0.004124977343378797",
            "extra": "mean: 5.682181983606554 msec\nrounds: 3660"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 418.88166040041216,
            "unit": "iter/sec",
            "range": "stddev: 0.001210449586164554",
            "extra": "mean: 2.3873091007233223 msec\nrounds: 1797"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 368.5341969608428,
            "unit": "iter/sec",
            "range": "stddev: 0.0003582654989567511",
            "extra": "mean: 2.7134523966747413 msec\nrounds: 421"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 131798.7074834911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011965263343192363",
            "extra": "mean: 7.5873278205346475 usec\nrounds: 34586"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.372485300043405,
            "unit": "iter/sec",
            "range": "stddev: 0.00012211348088898328",
            "extra": "mean: 36.533036333329015 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8015391385381461,
            "unit": "iter/sec",
            "range": "stddev: 0.009554497507017723",
            "extra": "mean: 1.2475997140000032 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02355982711217,
            "unit": "iter/sec",
            "range": "stddev: 0.16631774014962816",
            "extra": "mean: 42.44513320233333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3154.6737406019965,
            "unit": "iter/sec",
            "range": "stddev: 0.000015432434800827667",
            "extra": "mean: 316.9899907966943 usec\nrounds: 2173"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 852.9518579976601,
            "unit": "iter/sec",
            "range": "stddev: 0.000024485584457977827",
            "extra": "mean: 1.1723991109504603 msec\nrounds: 694"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 348.4773610953978,
            "unit": "iter/sec",
            "range": "stddev: 0.00019119361949144155",
            "extra": "mean: 2.8696268729096692 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 484.4251851565229,
            "unit": "iter/sec",
            "range": "stddev: 0.0010127320070538643",
            "extra": "mean: 2.0643022506703264 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10296.658943334918,
            "unit": "iter/sec",
            "range": "stddev: 0.00000537902513014558",
            "extra": "mean: 97.11888152295316 usec\nrounds: 6094"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10409.69341337027,
            "unit": "iter/sec",
            "range": "stddev: 0.000005168482283636483",
            "extra": "mean: 96.06430855260294 usec\nrounds: 8138"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1790.6189857874706,
            "unit": "iter/sec",
            "range": "stddev: 0.000018708741194285876",
            "extra": "mean: 558.4660991183583 usec\nrounds: 908"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1786.6023695616195,
            "unit": "iter/sec",
            "range": "stddev: 0.000016883465789469696",
            "extra": "mean: 559.7216353437228 usec\nrounds: 1426"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 960.2418333524478,
            "unit": "iter/sec",
            "range": "stddev: 0.00005833102595205289",
            "extra": "mean: 1.0414043267712534 msec\nrounds: 762"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 353.7138840810075,
            "unit": "iter/sec",
            "range": "stddev: 0.00012260271086372138",
            "extra": "mean: 2.8271437594204816 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.93691621303828,
            "unit": "iter/sec",
            "range": "stddev: 0.00039857052120962077",
            "extra": "mean: 47.762525761900825 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 431.9439113029095,
            "unit": "iter/sec",
            "range": "stddev: 0.00016351262443526087",
            "extra": "mean: 2.3151153976997016 msec\nrounds: 435"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 362.495615291529,
            "unit": "iter/sec",
            "range": "stddev: 0.0001048330024864831",
            "extra": "mean: 2.7586540576381107 msec\nrounds: 347"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 285.60995602597484,
            "unit": "iter/sec",
            "range": "stddev: 0.00013981679663484828",
            "extra": "mean: 3.5012785055331017 msec\nrounds: 271"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 171.11595667759414,
            "unit": "iter/sec",
            "range": "stddev: 0.00029898424850647535",
            "extra": "mean: 5.843990352601287 msec\nrounds: 173"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 805.9441968906724,
            "unit": "iter/sec",
            "range": "stddev: 0.000055369456806786914",
            "extra": "mean: 1.2407806940703756 msec\nrounds: 742"
          }
        ]
      }
    ]
  }
}