window.BENCHMARK_DATA = {
  "lastUpdate": 1783439475918,
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
          "id": "1810587e5c2d2fa016319605f7ee376f1509a9f5",
          "message": "fix(bench): esmemeval appraiser closed in ingest + X2 direct-VAD data correction (#112)\n\n* fix(bench): esmemeval adapter closed the shared appraiser in ingest\n\nAFTQueryAppraisedEsmemAdapter.ingest built a temporary EmotionalMemory with\nappraisal_engine=self._appraiser and called engine.close() in a finally block.\nEmotionalMemory.close() closes the appraisal engine (its httpx client), so every\nsubsequent retrieve()-time query appraisal hit a closed client and — with\nfallback_on_error=True — silently fell back to the KEYWORD appraiser instead of\ndirect-VAD.\n\nImpact: the Addendum X2 scored run (and the aborted Addendum Y gate run) appraised\nthe 401 session documents correctly at encode time (D1/D2 diagnostics valid) but\nappraised all ~1,133 queries with the keyword fallback, not the pre-registered\nDIRECT_VAD_SCHEMA. MADial-Bench (Addendum X) is unaffected (persistent engine, no\nclose in ingest). The X2 verdict (Hx2 FAIL, affect-orthogonal) is expected to\nstand; corrected numbers land in a follow-up.\n\nFix: ingest no longer closes the engine (its InMemoryStore is GC'd; the shared\nappraiser stays open); a new close() releases the appraiser's httpx client once at\nend of run, mirroring the MADial adapter.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n* docs(bench): correct X2 numbers to direct-VAD after the appraiser-close fix\n\nRe-ran Addendum X2 with the fixed adapter (PR #112) so the retrieve-time query\nappraisals use the pre-registered DIRECT_VAD_SCHEMA instead of the keyword\nfallback. Verdict unchanged (Hx2 FAIL, inverted, powered); numbers corrected\nthroughout:\n\n- u_nDCG@4: AFT 0.120 -> 0.133 vs cosine 0.284 (unchanged)\n- Δ=−0.164 -> −0.150 [−0.165, −0.136], d=−0.653 -> −0.622, MDE 0.019 -> 0.018\n- D1 AUC 0.971 -> 0.950, D2 68.2% -> 63.0%\n- corrected run's neutral fraction 45.5% matches the original post-hoc 45.2%\n\nCommitted the corrected results.{json,md,protocol.json}; added a Data-correction\nnote to the closure and prereg Status; propagated the corrected numbers to the\nclaim matrix (+ verbatim 09_current_evidence mirror, test green), 08_limitations,\npaper (abstract + X2 paragraph + regenerated arXiv bundle, 19pp), README,\ncomparison, benchmarks/README, index, ROADMAP, the two dated review docs, and a\nCHANGELOG Unreleased entry. v0.15.0 carried the pre-correction numbers.\n\nmake check green; claim-matrix mirror test green; check-arxiv-bundle OK.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T17:45:11+02:00",
          "tree_id": "481f3cf229a1ecc8eefe408b02d776bd106baee1",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/1810587e5c2d2fa016319605f7ee376f1509a9f5"
        },
        "date": 1783439475047,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 540.4387769118238,
            "unit": "iter/sec",
            "range": "stddev: 0.0008727135886317816",
            "extra": "mean: 1.850348351600901 msec\nrounds: 1624"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 420.0192453765687,
            "unit": "iter/sec",
            "range": "stddev: 0.0011982703077511629",
            "extra": "mean: 2.3808432851772046 msec\nrounds: 1862"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 384.298933836723,
            "unit": "iter/sec",
            "range": "stddev: 0.0014262424675816647",
            "extra": "mean: 2.602140968792981 msec\nrounds: 2179"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 212.01368525562864,
            "unit": "iter/sec",
            "range": "stddev: 0.002963500495794401",
            "extra": "mean: 4.7166766560105895 msec\nrounds: 3785"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 452.05828364131276,
            "unit": "iter/sec",
            "range": "stddev: 0.0009760551126669968",
            "extra": "mean: 2.212104138309417 msec\nrounds: 1822"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 382.68845472691726,
            "unit": "iter/sec",
            "range": "stddev: 0.0003068830159219068",
            "extra": "mean: 2.613091635371102 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 135018.0071098362,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013751665718692353",
            "extra": "mean: 7.406419494745667 usec\nrounds: 36656"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.85181512321352,
            "unit": "iter/sec",
            "range": "stddev: 0.0005311138584253798",
            "extra": "mean: 35.90430266667018 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8192891743455956,
            "unit": "iter/sec",
            "range": "stddev: 0.007409148495681788",
            "extra": "mean: 1.220570259333338 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.032344758057422,
            "unit": "iter/sec",
            "range": "stddev: 2.2188007144315636",
            "extra": "mean: 30.916910809 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3123.0251875294707,
            "unit": "iter/sec",
            "range": "stddev: 0.000013482065103142846",
            "extra": "mean: 320.2023486691983 usec\nrounds: 2217"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 862.9122095826457,
            "unit": "iter/sec",
            "range": "stddev: 0.000020574277943769803",
            "extra": "mean: 1.158866439592572 msec\nrounds: 687"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 356.24156339088734,
            "unit": "iter/sec",
            "range": "stddev: 0.00002401611106474866",
            "extra": "mean: 2.807084020408776 msec\nrounds: 294"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 502.4833426194692,
            "unit": "iter/sec",
            "range": "stddev: 0.00043727651866762253",
            "extra": "mean: 1.9901157216216425 msec\nrounds: 740"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10105.322662355255,
            "unit": "iter/sec",
            "range": "stddev: 0.000005782374666230176",
            "extra": "mean: 98.95775062435555 usec\nrounds: 6412"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10114.074833264789,
            "unit": "iter/sec",
            "range": "stddev: 0.000005586833038968853",
            "extra": "mean: 98.87211796288474 usec\nrounds: 8011"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1794.2491641223212,
            "unit": "iter/sec",
            "range": "stddev: 0.000009197078762962002",
            "extra": "mean: 557.3361938776002 usec\nrounds: 1078"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1779.5024940653875,
            "unit": "iter/sec",
            "range": "stddev: 0.00001012723074087794",
            "extra": "mean: 561.954817897128 usec\nrounds: 1598"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 997.7244810948598,
            "unit": "iter/sec",
            "range": "stddev: 0.000018185966130659005",
            "extra": "mean: 1.0022807087009062 msec\nrounds: 793"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 370.5864213961121,
            "unit": "iter/sec",
            "range": "stddev: 0.0000795367401210934",
            "extra": "mean: 2.69842590625068 msec\nrounds: 352"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.253540888120046,
            "unit": "iter/sec",
            "range": "stddev: 0.0008545431396042078",
            "extra": "mean: 44.93666895652752 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.8880806989499,
            "unit": "iter/sec",
            "range": "stddev: 0.000030330184629570587",
            "extra": "mean: 2.1236468727678925 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 375.3509263426666,
            "unit": "iter/sec",
            "range": "stddev: 0.00003518711911917216",
            "extra": "mean: 2.664173523544409 msec\nrounds: 361"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.8550199119504,
            "unit": "iter/sec",
            "range": "stddev: 0.000057877120696029405",
            "extra": "mean: 3.301909937932453 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.3808689437072,
            "unit": "iter/sec",
            "range": "stddev: 0.00008413203172956987",
            "extra": "mean: 5.171142344443069 msec\nrounds: 180"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 841.3531685588149,
            "unit": "iter/sec",
            "range": "stddev: 0.00001956665448478237",
            "extra": "mean: 1.1885615189551577 msec\nrounds: 765"
          }
        ]
      }
    ]
  }
}