window.BENCHMARK_DATA = {
  "lastUpdate": 1782573565956,
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
          "id": "3df9e0c07825242b3355dc7776947ac0f23c0a38",
          "message": "docs: propagate Addendum W (arousal calibration) to ledger and paper (#85)\n\nSurfaces the executed Addendum W result (calibrated direct-VAD arousal dominates\nthe SEC->projection on both r and MAE) across the evidence ledger and paper:\n\n- docs/research/08_limitations.md: append the W update to the direct-VAD/V paragraph\n  (arousal MAE caveat resolved by affine calibration; narrow-variance caveat noted).\n- docs/research/claim_validation_matrix.json (appraisal_human_validated): append W to\n  not_yet_shown, add W closure to evidence_refs and W results to benchmark_refs.\n  Surgical edits — allowed_public_wording unchanged (09 verbatim coupling intact).\n- CHANGELOG.md [Unreleased]: Research entry for Addendum W; Docs entries for the paper\n  reframe (#81) and the Gate 2 condition-significance test (#83).\n- paper/main.tex: one sentence + footnote in the V limitations paragraph (calibrated\n  direct-VAD dominates on both r and MAE); arXiv bundle regenerated.\n\nclaim-matrix tests green; paper compiles (0 errors); reproduce-paper-check clean.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T17:12:47+02:00",
          "tree_id": "0f8a74ae5ca27a4575a7ad68e376e190c69f820f",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/3df9e0c07825242b3355dc7776947ac0f23c0a38"
        },
        "date": 1782573564474,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 524.3351533573665,
            "unit": "iter/sec",
            "range": "stddev: 0.0009443212047629283",
            "extra": "mean: 1.907177105324538 msec\nrounds: 1709"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 420.0202356649285,
            "unit": "iter/sec",
            "range": "stddev: 0.0012182094218214363",
            "extra": "mean: 2.380837671825295 msec\nrounds: 1874"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 305.0957972077868,
            "unit": "iter/sec",
            "range": "stddev: 0.0021368060882375503",
            "extra": "mean: 3.2776590472629343 msec\nrounds: 2539"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 160.3333571691698,
            "unit": "iter/sec",
            "range": "stddev: 0.004145313981012472",
            "extra": "mean: 6.237005309786453 msec\nrounds: 4445"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 412.0511381236343,
            "unit": "iter/sec",
            "range": "stddev: 0.0011820034245759545",
            "extra": "mean: 2.4268832372450673 msec\nrounds: 1960"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 376.741506218078,
            "unit": "iter/sec",
            "range": "stddev: 0.0002961358857053952",
            "extra": "mean: 2.654339868305211 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 146794.23612254905,
            "unit": "iter/sec",
            "range": "stddev: 6.041735035453679e-7",
            "extra": "mean: 6.812256573651601 usec\nrounds: 39057"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.12365215027252,
            "unit": "iter/sec",
            "range": "stddev: 0.00020571723135262725",
            "extra": "mean: 30.189907666681393 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8204839098576202,
            "unit": "iter/sec",
            "range": "stddev: 0.016259950564262576",
            "extra": "mean: 1.2187929440000005 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.02881635232563942,
            "unit": "iter/sec",
            "range": "stddev: 0.15205441374323939",
            "extra": "mean: 34.702518511 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3677.332036299781,
            "unit": "iter/sec",
            "range": "stddev: 0.000008820510917248413",
            "extra": "mean: 271.93628155651226 usec\nrounds: 2543"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 939.5256131641935,
            "unit": "iter/sec",
            "range": "stddev: 0.000022189438345965097",
            "extra": "mean: 1.0643669379402412 msec\nrounds: 709"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 378.42171346034564,
            "unit": "iter/sec",
            "range": "stddev: 0.00003743652442814534",
            "extra": "mean: 2.6425544952371998 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 500.1994067490382,
            "unit": "iter/sec",
            "range": "stddev: 0.0004605818859100466",
            "extra": "mean: 1.999202690981446 msec\nrounds: 754"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 13380.71244403172,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026101983761238528",
            "extra": "mean: 74.7344361656943 usec\nrounds: 6979"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13553.038181476213,
            "unit": "iter/sec",
            "range": "stddev: 0.000002556645762113943",
            "extra": "mean: 73.78419411278297 usec\nrounds: 8459"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2224.211012848518,
            "unit": "iter/sec",
            "range": "stddev: 0.000008706899999183924",
            "extra": "mean: 449.59763000153174 usec\nrounds: 1100"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2192.86519018428,
            "unit": "iter/sec",
            "range": "stddev: 0.000013370507979316738",
            "extra": "mean: 456.02438511779366 usec\nrounds: 1532"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1176.414925537184,
            "unit": "iter/sec",
            "range": "stddev: 0.000015760606716493003",
            "extra": "mean: 850.0402182022402 usec\nrounds: 912"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 385.5714282101553,
            "unit": "iter/sec",
            "range": "stddev: 0.00004014150198580384",
            "extra": "mean: 2.593553170270052 msec\nrounds: 370"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 20.90054631788305,
            "unit": "iter/sec",
            "range": "stddev: 0.00031250043130296687",
            "extra": "mean: 47.84563928572403 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 467.4095320015583,
            "unit": "iter/sec",
            "range": "stddev: 0.00003880779538025527",
            "extra": "mean: 2.1394514478935918 msec\nrounds: 451"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 381.7625601576617,
            "unit": "iter/sec",
            "range": "stddev: 0.00006937912706361586",
            "extra": "mean: 2.6194292064340106 msec\nrounds: 373"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 314.32055770260865,
            "unit": "iter/sec",
            "range": "stddev: 0.00005332605358110428",
            "extra": "mean: 3.1814654673212317 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 199.77178573124354,
            "unit": "iter/sec",
            "range": "stddev: 0.0001468414324149124",
            "extra": "mean: 5.005711874375081 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 954.5764356509244,
            "unit": "iter/sec",
            "range": "stddev: 0.000014453738894838044",
            "extra": "mean: 1.0475850467837093 msec\nrounds: 855"
          }
        ]
      }
    ]
  }
}