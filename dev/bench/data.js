window.BENCHMARK_DATA = {
  "lastUpdate": 1781132480934,
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
          "id": "70aaa1d3613a123d4c122f7d6061799248f62c59",
          "message": "docs(research): record Hl3 follow-up resolution in evidence doc, claim matrix, changelog (#53)\n\n* docs(research): record Hl3 follow-up resolution in evidence doc, claim matrix, changelog\n\nAlign the three places that still said 'Hl3 data-collection issue\n(classifier log bug)' with the resolution merged in PR #52: bug fixed\n(log matched by query text instead of list index across the resume\nboundary) and ground-truth heuristic classifier accuracy measured\noffline on the exact 200-QA subset — 0.465 overall, 0.600 excluding\nadversarial, multi_hop essentially undetected (2/28).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n* docs: record paper O/P refresh and retroactive v0.11.3 DOI in changelog; add mem0 row to CLAUDE.md\n\nCHANGELOG [Unreleased] was missing two merged main commits: #48\n(paper/main.tex Addendum O/P scope discussion + bundle regen) and #49\n(Zenodo record 20475352 minted retroactively for v0.11.3 and propagated\nvia sync-metadata). CLAUDE.md's module table was missing\nintegrations/mem0.py, shipped in v0.11.0 with a top-level export.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-06-11T00:55:58+02:00",
          "tree_id": "ac5fe840884f52d9d69cca5a814f4d3a47e00e7c",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/70aaa1d3613a123d4c122f7d6061799248f62c59"
        },
        "date": 1781132480153,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 712.3217612250132,
            "unit": "iter/sec",
            "range": "stddev: 0.0006759677912756776",
            "extra": "mean: 1.4038599610943419 msec\nrounds: 1645"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 401.8450265876236,
            "unit": "iter/sec",
            "range": "stddev: 0.0014269788317365596",
            "extra": "mean: 2.4885215290376297 msec\nrounds: 2514"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 320.2544979238395,
            "unit": "iter/sec",
            "range": "stddev: 0.0020106844212462566",
            "extra": "mean: 3.122516643740668 msec\nrounds: 3315"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 222.59202688016012,
            "unit": "iter/sec",
            "range": "stddev: 0.0031022790473248392",
            "extra": "mean: 4.49252389681677 msec\nrounds: 4681"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 436.96532456180546,
            "unit": "iter/sec",
            "range": "stddev: 0.0011439272554506965",
            "extra": "mean: 2.288511110127132 msec\nrounds: 2597"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 466.6962653472242,
            "unit": "iter/sec",
            "range": "stddev: 0.0002684413635965294",
            "extra": "mean: 2.142721239168253 msec\nrounds: 577"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 173210.33897606385,
            "unit": "iter/sec",
            "range": "stddev: 5.869897122764371e-7",
            "extra": "mean: 5.773327423244586 usec\nrounds: 60405"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 42.52525450833172,
            "unit": "iter/sec",
            "range": "stddev: 0.0006919994946394708",
            "extra": "mean: 23.515438333333805 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.087554602690809,
            "unit": "iter/sec",
            "range": "stddev: 0.0020217689502186223",
            "extra": "mean: 919.4940626666627 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.04362227250721297,
            "unit": "iter/sec",
            "range": "stddev: 0.22835936461032047",
            "extra": "mean: 22.92406934633333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4469.383870743128,
            "unit": "iter/sec",
            "range": "stddev: 0.000006816079549841973",
            "extra": "mean: 223.7444866944779 usec\nrounds: 3119"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1159.5807225835208,
            "unit": "iter/sec",
            "range": "stddev: 0.00001235917168133828",
            "extra": "mean: 862.3806696027351 usec\nrounds: 908"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 463.87217947000863,
            "unit": "iter/sec",
            "range": "stddev: 0.000027600894890885095",
            "extra": "mean: 2.1557662741114108 msec\nrounds: 394"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 583.4033578439212,
            "unit": "iter/sec",
            "range": "stddev: 0.00045568030676053605",
            "extra": "mean: 1.7140799526689243 msec\nrounds: 993"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 15104.358173452614,
            "unit": "iter/sec",
            "range": "stddev: 0.000002850816512512371",
            "extra": "mean: 66.20605712049371 usec\nrounds: 10259"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 14932.730539768729,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025151339706536986",
            "extra": "mean: 66.96698887968333 usec\nrounds: 9622"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2522.0390383570793,
            "unit": "iter/sec",
            "range": "stddev: 0.000008998220691567032",
            "extra": "mean: 396.5045682446793 usec\nrounds: 1436"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2511.584008142602,
            "unit": "iter/sec",
            "range": "stddev: 0.000010218839908536602",
            "extra": "mean: 398.15510719847765 usec\nrounds: 1931"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1541.2067503806372,
            "unit": "iter/sec",
            "range": "stddev: 0.000013668656723222846",
            "extra": "mean: 648.8422139035055 usec\nrounds: 1122"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 500.0498008112595,
            "unit": "iter/sec",
            "range": "stddev: 0.00006069598726186718",
            "extra": "mean: 1.9998008165939525 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 25.19195572445013,
            "unit": "iter/sec",
            "range": "stddev: 0.0005517671382906157",
            "extra": "mean: 39.695211079997534 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 607.2694409352613,
            "unit": "iter/sec",
            "range": "stddev: 0.000049880507958654404",
            "extra": "mean: 1.6467154982471879 msec\nrounds: 570"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 503.19029615847364,
            "unit": "iter/sec",
            "range": "stddev: 0.00003825152931711527",
            "extra": "mean: 1.9873197230438289 msec\nrounds: 473"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 414.1921397650942,
            "unit": "iter/sec",
            "range": "stddev: 0.000041953589122022436",
            "extra": "mean: 2.4143384289405927 msec\nrounds: 387"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 270.06709892825177,
            "unit": "iter/sec",
            "range": "stddev: 0.0000635623803042497",
            "extra": "mean: 3.7027835081298375 msec\nrounds: 246"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1253.7984951407266,
            "unit": "iter/sec",
            "range": "stddev: 0.000016209802501548734",
            "extra": "mean: 797.5763281545171 usec\nrounds: 1094"
          }
        ]
      }
    ]
  }
}