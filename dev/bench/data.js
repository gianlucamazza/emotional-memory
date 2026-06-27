window.BENCHMARK_DATA = {
  "lastUpdate": 1782556779159,
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
          "id": "367a3ad58f37a30c654b47e6400a9a85e52b3e74",
          "message": "feat(bench): add Addendum T2A — naturalistic query-appraisal re-test (DailyDialog, Ht2a FAIL) (#77)\n\n* docs(bench): pre-register Addendum T2A (naturalistic query appraisal, DailyDialog)\n\nHt2a: does retrieve-time query appraisal (direct-VAD, no oracle) recover an\nadvantage on naturalistic affect-conditioned dialogue (DailyDialog N=120/480),\nwhere stale-state AFT tied cosine in Hk1? Arms, decision rule (Holm m=4, paired\nbootstrap n=10k seed=0), and downstream consequences fixed before execution.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): add Addendum T2A — naturalistic query-appraisal re-test (DailyDialog, Ht2a FAIL)\n\nAdds the aft_query_appraised arm (direct-VAD query appraisal at retrieve-time via\nthe public query_affect API; encode path unchanged) and a dedicated runner.\n\nResult (N=120 personas / 396 queries, multilingual-e5-small, bootstrap n=10k seed=0):\n- aft_query_appraised top1=0.212 vs naive_cosine 0.220 — delta=-0.008 [-0.056,+0.040],\n  p_holm=1.000, 0/3 directional types -> Ht2a FAIL (reproduces the Hk1 null).\n- +0.010 (ns) over the stale-state AFT arm.\n- Diagnostic: appraised query affect vs target-session PAD valence r=0.69, arousal r=0.74.\n\nThe query appraisal is faithful, so this is a regime limit, not an appraisal-quality\nfailure: Addendum T's production-reachable recovery is bounded to the affect-discriminative\nregime (Addendum U) and does not extend to naturalistic affect-conditioned dialogue.\n\nPropagated to claim_validation_matrix (downstream_value, cross_domain_affect_replication),\n08_limitations 2.4, 09_current_evidence, problem_register, CHANGELOG. Pre-registration\ncommitted before execution (260318d).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-27T12:34:22+02:00",
          "tree_id": "37bdcc3ecb3cb9966fe998d738511306511b0e63",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/367a3ad58f37a30c654b47e6400a9a85e52b3e74"
        },
        "date": 1782556778442,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 509.5640849462119,
            "unit": "iter/sec",
            "range": "stddev: 0.0009319278025887009",
            "extra": "mean: 1.9624616992101338 msec\nrounds: 1772"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 470.5392936846159,
            "unit": "iter/sec",
            "range": "stddev: 0.0009369319205810767",
            "extra": "mean: 2.1252210249422037 msec\nrounds: 1724"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 385.0454319139132,
            "unit": "iter/sec",
            "range": "stddev: 0.0014090606619332818",
            "extra": "mean: 2.5970961271489013 msec\nrounds: 2210"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 250.270419678175,
            "unit": "iter/sec",
            "range": "stddev: 0.002401126377375096",
            "extra": "mean: 3.9956779602076384 msec\nrounds: 3468"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 466.71078621813353,
            "unit": "iter/sec",
            "range": "stddev: 0.0009302833294401976",
            "extra": "mean: 2.14265457223141 msec\nrounds: 1779"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 387.2189277331274,
            "unit": "iter/sec",
            "range": "stddev: 0.00024297150781805988",
            "extra": "mean: 2.5825183852820426 msec\nrounds: 462"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 138421.2096636659,
            "unit": "iter/sec",
            "range": "stddev: 0.000001141778310550061",
            "extra": "mean: 7.224326405106465 usec\nrounds: 52588"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.658236279666927,
            "unit": "iter/sec",
            "range": "stddev: 0.00024593367514304427",
            "extra": "mean: 34.89398266666891 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8327852496010614,
            "unit": "iter/sec",
            "range": "stddev: 0.0019406484388509312",
            "extra": "mean: 1.2007897600000017 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.039217373700184834,
            "unit": "iter/sec",
            "range": "stddev: 1.1828420117709735",
            "extra": "mean: 25.498902798666677 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3147.338474455287,
            "unit": "iter/sec",
            "range": "stddev: 0.000012103360769409625",
            "extra": "mean: 317.7287756357603 usec\nrounds: 2594"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 855.5361203525528,
            "unit": "iter/sec",
            "range": "stddev: 0.000016847759074644483",
            "extra": "mean: 1.1688577211537439 msec\nrounds: 624"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.0966641330915,
            "unit": "iter/sec",
            "range": "stddev: 0.000021764251581855622",
            "extra": "mean: 2.8240876045760714 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 494.69100066133296,
            "unit": "iter/sec",
            "range": "stddev: 0.0007822355652059707",
            "extra": "mean: 2.0214639010273876 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10677.633553632197,
            "unit": "iter/sec",
            "range": "stddev: 0.000005140998478492865",
            "extra": "mean: 93.65371034481993 usec\nrounds: 6960"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10684.415390254562,
            "unit": "iter/sec",
            "range": "stddev: 0.000005129114536764948",
            "extra": "mean: 93.59426449407023 usec\nrounds: 8193"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1831.0294203976946,
            "unit": "iter/sec",
            "range": "stddev: 0.000009817056661189027",
            "extra": "mean: 546.1408696441385 usec\nrounds: 1097"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1808.4128426179782,
            "unit": "iter/sec",
            "range": "stddev: 0.000014443818673956474",
            "extra": "mean: 552.9710785244889 usec\nrounds: 1681"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1004.4081487591665,
            "unit": "iter/sec",
            "range": "stddev: 0.000018348503095605544",
            "extra": "mean: 995.6111977340962 usec\nrounds: 794"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 377.57810316001985,
            "unit": "iter/sec",
            "range": "stddev: 0.000029572768117195043",
            "extra": "mean: 2.6484586675731934 msec\nrounds: 367"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 24.16131265991638,
            "unit": "iter/sec",
            "range": "stddev: 0.0006862860630197985",
            "extra": "mean: 41.38847976000079 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 472.6016439706599,
            "unit": "iter/sec",
            "range": "stddev: 0.00003401787231553244",
            "extra": "mean: 2.115946934924505 msec\nrounds: 461"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 377.42283107370196,
            "unit": "iter/sec",
            "range": "stddev: 0.00002898977154815607",
            "extra": "mean: 2.649548245810077 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.9120502174998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000667011958467112",
            "extra": "mean: 3.301288275860833 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.8629682826109,
            "unit": "iter/sec",
            "range": "stddev: 0.00006915567428199432",
            "extra": "mean: 5.158282723403952 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 845.5228974095004,
            "unit": "iter/sec",
            "range": "stddev: 0.00001777145283726478",
            "extra": "mean: 1.1827000818828017 msec\nrounds: 806"
          }
        ]
      }
    ]
  }
}