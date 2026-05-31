window.BENCHMARK_DATA = {
  "lastUpdate": 1780191993321,
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
          "id": "9ec4752ac9f3abd192a103d67d24739325109b8a",
          "message": "feat(appraisal): recalibrate Scherer SEC→affect mapping (Addendum O) (#46)\n\n* docs(bench): pre-register Addendum O (SEC mapping recalibration) + tooling\n\nFrozen protocol for numeric recalibration of the SEC->valence/arousal mapping, the\npre-registered fallback after Addendum N (prompt-only fixed valence bias but not\narousal). Adds the by-scenario 70/30 split (seed 42, 87 train / 38 test scenarios),\ndump_sec.py (per-event SEC + oracle, the single LLM-costed step) and fit.py (pure\nnumpy lstsq; M0/M1/M2; valence intercept constrained to 0 to keep neutral->0).\nLeakage: fit/test split is within v3 for the calibration claim; a leakage-free Hg1\nretrieval re-run needs scenarios disjoint from v3 (v3_noAF is contained in v3) and\nis logged as next_study, not run here.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(appraisal): recalibrate Scherer SEC→affect mapping (Addendum O, M1)\n\nNumerically recalibrate _scherer_project against oracle affect on a held-out\nscenario split (gpt-5-mini, N=750 SEC dump, by-scenario 70/30 split seed 42).\nPromote model M1: Scherer feature basis preserved (coping_signed, |novelty|,\n1-coping_potential), valence intercept constrained to 0 (G1 invariant).\n\nHeld-out test (228 events): valence bias +0.200→+0.072, arousal bias\n-0.144→-0.023; MAE improves on both axes; Pearson r within the -0.05 guardrail.\nBoth pre-registered hypotheses PASS (Ho1 bias, Ho2 guardrail).\n\n- appraisal_schema.py: M1 weights live in _scherer_project; neutral arousal\n  shifts 0.15 → 0.20775 (free intercept 0.1399 + 0.1357·0.5).\n- appraisal.py: AppraisalVector.to_core_affect() delegates to the schema\n  projection — single source of truth for the mapping.\n- tests/test_appraisal.py: neutral-affect expectation updated to 0.20775.\n- claim_validation_matrix.json: appraisal_llm_real_dual_path updated with the\n  calibration result + refs; next_study now the Hg1 re-run on a disjoint set.\n- benchmarks/appraisal_calibration/: dump_sec.py (LLM SEC dump), fit.py (numpy\n  lstsq M0/M1/M2 + 5-fold CV), frozen split + results; closure doc.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-05-31T03:40:21+02:00",
          "tree_id": "296cca8e940ba1a0f444ee40c4b4b8c33bbc9123",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/9ec4752ac9f3abd192a103d67d24739325109b8a"
        },
        "date": 1780191992846,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 608.5731259702326,
            "unit": "iter/sec",
            "range": "stddev: 0.0007662313726481231",
            "extra": "mean: 1.6431879051604943 msec\nrounds: 1434"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 410.9933900071179,
            "unit": "iter/sec",
            "range": "stddev: 0.001290794742529612",
            "extra": "mean: 2.43312915563601 msec\nrounds: 1934"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 319.6317364523687,
            "unit": "iter/sec",
            "range": "stddev: 0.0018544207374066677",
            "extra": "mean: 3.1286004672099237 msec\nrounds: 2577"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 180.10354925816662,
            "unit": "iter/sec",
            "range": "stddev: 0.003984471223580128",
            "extra": "mean: 5.552361428294595 msec\nrounds: 4128"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 798.8278621816723,
            "unit": "iter/sec",
            "range": "stddev: 0.00045059933547081024",
            "extra": "mean: 1.251834152690804 msec\nrounds: 799"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 378.76923847628404,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348914894302753",
            "extra": "mean: 2.6401299219092027 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138205.46225069067,
            "unit": "iter/sec",
            "range": "stddev: 9.155019685749461e-7",
            "extra": "mean: 7.235604032683612 usec\nrounds: 46966"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 33.15258763571796,
            "unit": "iter/sec",
            "range": "stddev: 0.000451555595189897",
            "extra": "mean: 30.163557999998147 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8488266927376965,
            "unit": "iter/sec",
            "range": "stddev: 0.003505358047535138",
            "extra": "mean: 1.1780967876666655 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.034305840734419,
            "unit": "iter/sec",
            "range": "stddev: 0.9998328595314026",
            "extra": "mean: 29.149555253333332 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3522.129115038671,
            "unit": "iter/sec",
            "range": "stddev: 0.000009501623183875701",
            "extra": "mean: 283.91917710518703 usec\nrounds: 2411"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 900.6023262744915,
            "unit": "iter/sec",
            "range": "stddev: 0.000018348756232443936",
            "extra": "mean: 1.1103679957575563 msec\nrounds: 707"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 359.2611350965117,
            "unit": "iter/sec",
            "range": "stddev: 0.00015031730943498514",
            "extra": "mean: 2.783490620913839 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 484.9092751767718,
            "unit": "iter/sec",
            "range": "stddev: 0.0009990689589022056",
            "extra": "mean: 2.0622414360613206 msec\nrounds: 782"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 12212.570456930516,
            "unit": "iter/sec",
            "range": "stddev: 0.000003774553420210413",
            "extra": "mean: 81.8828438719475 usec\nrounds: 7955"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 12499.831558856584,
            "unit": "iter/sec",
            "range": "stddev: 0.000003965562536588658",
            "extra": "mean: 80.00107803784474 usec\nrounds: 8765"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2030.0073721284944,
            "unit": "iter/sec",
            "range": "stddev: 0.000012236113713865589",
            "extra": "mean: 492.6090484841365 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2006.6844620854556,
            "unit": "iter/sec",
            "range": "stddev: 0.000014767231469347024",
            "extra": "mean: 498.33445112778 usec\nrounds: 1729"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1174.0195490974068,
            "unit": "iter/sec",
            "range": "stddev: 0.000018918849386878436",
            "extra": "mean: 851.7745728925944 usec\nrounds: 878"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 386.49736746027577,
            "unit": "iter/sec",
            "range": "stddev: 0.000042918439347449926",
            "extra": "mean: 2.5873397445657376 msec\nrounds: 368"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.26086053765866,
            "unit": "iter/sec",
            "range": "stddev: 0.000660106402487379",
            "extra": "mean: 44.921893217393894 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 465.32316172290575,
            "unit": "iter/sec",
            "range": "stddev: 0.00013001558529785148",
            "extra": "mean: 2.149044110113495 msec\nrounds: 445"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 380.90345662942656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000862513055140677",
            "extra": "mean: 2.6253371624634014 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 317.3448764742107,
            "unit": "iter/sec",
            "range": "stddev: 0.00008729539808650179",
            "extra": "mean: 3.1511458798713763 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 208.16626994477,
            "unit": "iter/sec",
            "range": "stddev: 0.00010744866810455492",
            "extra": "mean: 4.80385222959184 msec\nrounds: 196"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 965.2995172185551,
            "unit": "iter/sec",
            "range": "stddev: 0.000020135717810604758",
            "extra": "mean: 1.0359478919884182 msec\nrounds: 824"
          }
        ]
      }
    ]
  }
}