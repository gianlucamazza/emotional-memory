window.BENCHMARK_DATA = {
  "lastUpdate": 1782946183988,
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
          "id": "216e363939162d81a32526c9754dd74ab4892228",
          "message": "docs(bench): pre-register Addendum X — third-party retrieval on MADial-Bench (#91)\n\n* docs(bench): pre-register Addendum X — third-party retrieval on MADial-Bench\n\nFirst test of the production-reachable query-appraisal mechanism (Addendum T)\non a third-party affect-discriminative corpus, oracle-free:\n\n- Dataset: MADial-Bench EN (NAACL 2025, MIT), 160 queries with gold memory\n  sets and per-memory emotion labels; data files pinned by sha256 + commit\n- Selection audit: HLME/MemEmo not released, ENPMR-Bench repo 404,\n  EvoEmo (CC-BY-4.0) reserved as Addendum X2 replication corpus\n- Hx1 (single family member): aft_query_appraised nDCG@5 > naive_cosine,\n  one-tailed paired bootstrap n=10k seed=0; decision rule ex-ante\n- Protocol replicates the repo's own evaluation (query = context before\n  first test-turn, full 160-memory bank, binary relevance); event-only\n  document text keeps third-party emotion labels out of both arms\n- Diagnostics: D1 appraisal-vs-label AUC (Happy vs negative), D2 corpus\n  affect-discriminativeness; exploratory arms: full-stack decay on real\n  dates, Mem0 if adapter runs unmodified\n\nCommitted before any scored run, per pre-registration integrity policy.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n* docs(bench): amend Addendum X prereg pre-run (A1: s5 semantics, arm-3 feasibility)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-02T00:43:49+02:00",
          "tree_id": "32d428cc50af9c1abbcbcb7051a6684f15d9231c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/216e363939162d81a32526c9754dd74ab4892228"
        },
        "date": 1782946182222,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 560.3650590047755,
            "unit": "iter/sec",
            "range": "stddev: 0.0008164728453780321",
            "extra": "mean: 1.7845509528663845 msec\nrounds: 1570"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 455.0888303494239,
            "unit": "iter/sec",
            "range": "stddev: 0.0010324247943211701",
            "extra": "mean: 2.197373201254325 msec\nrounds: 1754"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 353.0370534075797,
            "unit": "iter/sec",
            "range": "stddev: 0.0016013755407892308",
            "extra": "mean: 2.8325638636166173 msec\nrounds: 2317"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 238.93260163479167,
            "unit": "iter/sec",
            "range": "stddev: 0.0025619025380079203",
            "extra": "mean: 4.185280673955493 msec\nrounds: 3567"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 447.74927342236253,
            "unit": "iter/sec",
            "range": "stddev: 0.000994839756238542",
            "extra": "mean: 2.2333927922574173 msec\nrounds: 1834"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 383.7569135208049,
            "unit": "iter/sec",
            "range": "stddev: 0.0002597156817054318",
            "extra": "mean: 2.605816246606294 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 137313.96555256352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012702607564998586",
            "extra": "mean: 7.282580442389176 usec\nrounds: 48196"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.739405960254704,
            "unit": "iter/sec",
            "range": "stddev: 0.00035512032304582837",
            "extra": "mean: 36.049798666662504 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8154257955246993,
            "unit": "iter/sec",
            "range": "stddev: 0.0031221892281955432",
            "extra": "mean: 1.2263531586666734 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.035281406595959776,
            "unit": "iter/sec",
            "range": "stddev: 1.411204213309943",
            "extra": "mean: 28.343541159 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3100.3668099246797,
            "unit": "iter/sec",
            "range": "stddev: 0.000012254465362302716",
            "extra": "mean: 322.54248007005793 usec\nrounds: 2283"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 837.045536501002,
            "unit": "iter/sec",
            "range": "stddev: 0.000015714795801040953",
            "extra": "mean: 1.1946781344539228 msec\nrounds: 595"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 345.08345686393153,
            "unit": "iter/sec",
            "range": "stddev: 0.00009426537077797284",
            "extra": "mean: 2.897849723333176 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 492.72464371938855,
            "unit": "iter/sec",
            "range": "stddev: 0.0007758694832144607",
            "extra": "mean: 2.029531124019666 msec\nrounds: 766"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10513.937474292608,
            "unit": "iter/sec",
            "range": "stddev: 0.00000775459612829492",
            "extra": "mean: 95.11184581847452 usec\nrounds: 5273"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10182.366033526712,
            "unit": "iter/sec",
            "range": "stddev: 0.000005391558084790253",
            "extra": "mean: 98.20900139588137 usec\nrounds: 7162"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1820.6550851853194,
            "unit": "iter/sec",
            "range": "stddev: 0.000011204081389623302",
            "extra": "mean: 549.2528530730535 usec\nrounds: 1123"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1803.9737318949472,
            "unit": "iter/sec",
            "range": "stddev: 0.000011867628776245822",
            "extra": "mean: 554.3317966994844 usec\nrounds: 1697"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 982.6165712858016,
            "unit": "iter/sec",
            "range": "stddev: 0.0000839692615720279",
            "extra": "mean: 1.0176909582253955 msec\nrounds: 766"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 374.3976358908312,
            "unit": "iter/sec",
            "range": "stddev: 0.00003605590775792274",
            "extra": "mean: 2.6709570364156496 msec\nrounds: 357"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.799122788483334,
            "unit": "iter/sec",
            "range": "stddev: 0.0009525197274227519",
            "extra": "mean: 43.86133665217753 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 470.553987180297,
            "unit": "iter/sec",
            "range": "stddev: 0.000022817924677187374",
            "extra": "mean: 2.1251546628949103 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 374.7215751971122,
            "unit": "iter/sec",
            "range": "stddev: 0.00004028879230992462",
            "extra": "mean: 2.6686480474842607 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 302.2873737779588,
            "unit": "iter/sec",
            "range": "stddev: 0.00004904626549560853",
            "extra": "mean: 3.3081103835138572 msec\nrounds: 279"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 193.15567134332318,
            "unit": "iter/sec",
            "range": "stddev: 0.00008017782235567772",
            "extra": "mean: 5.177171309780271 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 834.8977382979284,
            "unit": "iter/sec",
            "range": "stddev: 0.000050827242997460295",
            "extra": "mean: 1.1977514779698153 msec\nrounds: 749"
          }
        ]
      }
    ]
  }
}