window.BENCHMARK_DATA = {
  "lastUpdate": 1783462323170,
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
          "id": "8b15deca1438cd422e63791468b4b091b597f3ca",
          "message": "perf(engine): parallelise per-item appraisal in encode_batch (#116)\n\n* perf(engine): parallelise per-item appraisal in encode_batch\n\nencode_batch appraised items in a sequential loop (both sync and async), so a\nbatch of N with LLM appraisal cost N serial round-trips — the dominant encode\nlatency (200-2000 ms/item). Appraisal depends only on content/context, not on\nevolving engine state, so the batch's appraisals can be computed concurrently\nup front and then consumed by the order-preserving state-evolution loop.\n\n- async: `_appraise_many` runs the calls under an `asyncio.Semaphore` via\n  `asyncio.gather` (order preserved).\n- sync: `_appraise_many` uses a bounded `ThreadPoolExecutor` (`pool.map`,\n  order preserved); the `llm` callable must be thread-safe (httpx.Client and\n  KeywordAppraisalEngine are).\n- New `EmotionalMemoryConfig.appraisal_max_concurrency` (default 8, ge=1);\n  set to 1 to force fully sequential appraisal.\n\nResults are identical to the sequential path (state still evolves item-by-item\nin input order); only the appraisal I/O waits overlap. Tests assert real\nconcurrency, order/result preservation, and the concurrency=1 sequential path.\n\nelaborate_pending() is left sequential (it interleaves appraisal with per-memory\nstate blending) — a possible follow-up.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* docs: document parallel batch appraisal (changelog, config, limitations)\n\nCHANGELOG [Unreleased] Added/Changed entries; CLAUDE.md encode_batch data-flow +\nappraisal_max_concurrency config flag; 08_limitations §3.1 batch-throughput note.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-08T00:05:51+02:00",
          "tree_id": "d74375eff2f7079cc3bda5485009a4a6380dc03e",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/8b15deca1438cd422e63791468b4b091b597f3ca"
        },
        "date": 1783462321556,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 548.5436256034036,
            "unit": "iter/sec",
            "range": "stddev: 0.0008428982410737496",
            "extra": "mean: 1.8230090613121055 msec\nrounds: 1631"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 419.18437518526184,
            "unit": "iter/sec",
            "range": "stddev: 0.0012354192493394137",
            "extra": "mean: 2.38558510096671 msec\nrounds: 1862"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 382.15458386790647,
            "unit": "iter/sec",
            "range": "stddev: 0.0014867510942200562",
            "extra": "mean: 2.616742130576287 msec\nrounds: 2152"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 201.32382109600823,
            "unit": "iter/sec",
            "range": "stddev: 0.0039037419008520445",
            "extra": "mean: 4.967122094921471 msec\nrounds: 3308"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 434.4850550822732,
            "unit": "iter/sec",
            "range": "stddev: 0.001126264011373135",
            "extra": "mean: 2.301575136596222 msec\nrounds: 1757"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 364.11497009492484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003471435274895533",
            "extra": "mean: 2.7463852962137203 msec\nrounds: 449"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136504.6413874686,
            "unit": "iter/sec",
            "range": "stddev: 0.000001330568850253416",
            "extra": "mean: 7.325758229432642 usec\nrounds: 31290"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 27.951749206020555,
            "unit": "iter/sec",
            "range": "stddev: 0.000542859139020363",
            "extra": "mean: 35.7759363333372 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8104591589339948,
            "unit": "iter/sec",
            "range": "stddev: 0.003319298438794064",
            "extra": "mean: 1.233868467000003 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.03256913900158444,
            "unit": "iter/sec",
            "range": "stddev: 4.211417330245443",
            "extra": "mean: 30.703912680999995 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3207.955263215309,
            "unit": "iter/sec",
            "range": "stddev: 0.000011926054171416533",
            "extra": "mean: 311.7250453791265 usec\nrounds: 2424"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 872.4903539041476,
            "unit": "iter/sec",
            "range": "stddev: 0.000049115327586755004",
            "extra": "mean: 1.1461444765839333 msec\nrounds: 726"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 359.49500415933926,
            "unit": "iter/sec",
            "range": "stddev: 0.0000299488712247593",
            "extra": "mean: 2.7816798242814222 msec\nrounds: 313"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 481.8326863217884,
            "unit": "iter/sec",
            "range": "stddev: 0.0009167098251837323",
            "extra": "mean: 2.0754092206441914 msec\nrounds: 775"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10511.566535331505,
            "unit": "iter/sec",
            "range": "stddev: 0.000004993641725191191",
            "extra": "mean: 95.13329879412335 usec\nrounds: 6553"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10155.016337093011,
            "unit": "iter/sec",
            "range": "stddev: 0.000005073188922983186",
            "extra": "mean: 98.4734998748669 usec\nrounds: 7996"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1855.667250927337,
            "unit": "iter/sec",
            "range": "stddev: 0.000008346248892540571",
            "extra": "mean: 538.8897171625289 usec\nrounds: 1043"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1839.7512164037425,
            "unit": "iter/sec",
            "range": "stddev: 0.000043111776770071876",
            "extra": "mean: 543.5517536739299 usec\nrounds: 1701"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1021.9424652667958,
            "unit": "iter/sec",
            "range": "stddev: 0.000022033375910657495",
            "extra": "mean: 978.5286686750342 usec\nrounds: 830"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 378.0294502232302,
            "unit": "iter/sec",
            "range": "stddev: 0.00003843931516414806",
            "extra": "mean: 2.6452965487463738 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 22.497694250444503,
            "unit": "iter/sec",
            "range": "stddev: 0.0007517309216709299",
            "extra": "mean: 44.448999478257306 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 477.2404752742452,
            "unit": "iter/sec",
            "range": "stddev: 0.000028993864193795926",
            "extra": "mean: 2.095379691811245 msec\nrounds: 464"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 383.6461655671869,
            "unit": "iter/sec",
            "range": "stddev: 0.00002943787194211124",
            "extra": "mean: 2.606568473117901 msec\nrounds: 372"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 303.34689713105,
            "unit": "iter/sec",
            "range": "stddev: 0.0003628205811061224",
            "extra": "mean: 3.2965558885146145 msec\nrounds: 296"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 190.5423670484994,
            "unit": "iter/sec",
            "range": "stddev: 0.000927496435006828",
            "extra": "mean: 5.2481766417096445 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 817.7399076388956,
            "unit": "iter/sec",
            "range": "stddev: 0.00025929813640900506",
            "extra": "mean: 1.2228827169354541 msec\nrounds: 809"
          }
        ]
      }
    ]
  }
}