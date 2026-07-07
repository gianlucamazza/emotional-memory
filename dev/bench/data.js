window.BENCHMARK_DATA = {
  "lastUpdate": 1783388529219,
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
          "id": "bfc4bf2cc1c57dd6daa13ceee9d45f38c1708264",
          "message": "docs(bench): addendum x2 executed — hx2 fail inverted, closure + propagation (#104)\n\nScored run (make bench-x2-esmem, gpt-5-mini, N=1,133): Hx2 FAIL, decisive —\ncosine significantly ahead on upstream-verbatim nDCG@4 (0.284 vs 0.120,\ndelta=-0.164 [-0.179, -0.149], d=-0.653, powered: MDE 0.019), uniform across\nall 31 grid contrasts and capabilities. D1 AUC=0.971 (appraisal faithful);\nD2=68.2% (ex-ante low-D2 prior wrong -> pre-declared reading iii). Post-hoc:\nfailure mode distinct from X — affect-orthogonal content-determined gold\n(45.2% neutral queries, corr(query,gold)=+0.25), not counter-congruence.\nProvenance bound hardened: operative boundary = the gold relation itself\nmust be affect-conditioned (to date only author-crafted corpora).\n\n- results.{json,md,protocol.json} committed; prereg Status updated\n- closure: verdict, diagnostics, post-hoc, bound update, dropped arms\n- propagation: 08_limitations §2.4, claim matrix + 09_current_evidence\n  (verbatim mirror), CHANGELOG, ROADMAP, paper (X2 paragraph, abstract\n  both-corpora rewording ~1890 chars, conclusion, acknowledgments,\n  refs.bib chen2026esmemeval, A--X2 range, arXiv bundle regenerated)\n\nCo-authored-by: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-07T03:37:35+02:00",
          "tree_id": "81f552c7ba90ca5ce1cb6321cd62ee305228a209",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/bfc4bf2cc1c57dd6daa13ceee9d45f38c1708264"
        },
        "date": 1783388527706,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 692.0029034077737,
            "unit": "iter/sec",
            "range": "stddev: 0.0006967711838692828",
            "extra": "mean: 1.445080642112182 msec\nrounds: 1534"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 398.1041545052029,
            "unit": "iter/sec",
            "range": "stddev: 0.001330155266848731",
            "extra": "mean: 2.511905461631978 msec\nrounds: 2463"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 361.62708702726803,
            "unit": "iter/sec",
            "range": "stddev: 0.0015888012729492972",
            "extra": "mean: 2.7652795818489015 msec\nrounds: 2975"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 200.38633205552006,
            "unit": "iter/sec",
            "range": "stddev: 0.0029244664335424794",
            "extra": "mean: 4.990360319200487 msec\nrounds: 5401"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 421.51199184867676,
            "unit": "iter/sec",
            "range": "stddev: 0.0011358588827114903",
            "extra": "mean: 2.372411744714966 msec\nrounds: 2507"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 443.6711569040269,
            "unit": "iter/sec",
            "range": "stddev: 0.0002723952621849063",
            "extra": "mean: 2.253921591338235 msec\nrounds: 531"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 178006.058784772,
            "unit": "iter/sec",
            "range": "stddev: 7.743453195373828e-7",
            "extra": "mean: 5.617786309223918 usec\nrounds: 48982"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 44.73252922240203,
            "unit": "iter/sec",
            "range": "stddev: 0.00031674501809883553",
            "extra": "mean: 22.35509633332337 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 1.011429459012835,
            "unit": "iter/sec",
            "range": "stddev: 0.0031520221091288964",
            "extra": "mean: 988.6996973333263 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.048101506207322155,
            "unit": "iter/sec",
            "range": "stddev: 0.08350533584939951",
            "extra": "mean: 20.789369789999984 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 4393.54503513287,
            "unit": "iter/sec",
            "range": "stddev: 0.000006974559674174695",
            "extra": "mean: 227.60663473425802 usec\nrounds: 3236"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 1147.9307093630582,
            "unit": "iter/sec",
            "range": "stddev: 0.000019545480713793072",
            "extra": "mean: 871.1327189381151 usec\nrounds: 829"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 445.8586499557023,
            "unit": "iter/sec",
            "range": "stddev: 0.00003125661162841924",
            "extra": "mean: 2.2428632933315384 msec\nrounds: 375"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 550.6087937222593,
            "unit": "iter/sec",
            "range": "stddev: 0.00043679464720092345",
            "extra": "mean: 1.8161715021653375 msec\nrounds: 924"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 14072.425627044064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034056096055941357",
            "extra": "mean: 71.06095469982255 usec\nrounds: 9426"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 13954.944404749449,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030492502099581894",
            "extra": "mean: 71.65918910143837 usec\nrounds: 9561"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 2506.4494961955397,
            "unit": "iter/sec",
            "range": "stddev: 0.000013916392979351763",
            "extra": "mean: 398.97073590266564 usec\nrounds: 1064"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 2520.3413096630143,
            "unit": "iter/sec",
            "range": "stddev: 0.000013603540287538357",
            "extra": "mean: 396.7716579361652 usec\nrounds: 1871"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1502.982242391935,
            "unit": "iter/sec",
            "range": "stddev: 0.00001694052574542223",
            "extra": "mean: 665.3438555658121 usec\nrounds: 1087"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 462.71714035306013,
            "unit": "iter/sec",
            "range": "stddev: 0.00003138986063077481",
            "extra": "mean: 2.1611475192749183 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 35.285485320726046,
            "unit": "iter/sec",
            "range": "stddev: 0.0003606064029081127",
            "extra": "mean: 28.34026486841654 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 562.1054453240054,
            "unit": "iter/sec",
            "range": "stddev: 0.000031749309455591154",
            "extra": "mean: 1.7790256406848828 msec\nrounds: 526"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 468.8754784144384,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367205282018028",
            "extra": "mean: 2.13276241995343 msec\nrounds: 431"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 387.0528771942194,
            "unit": "iter/sec",
            "range": "stddev: 0.000040227709464991624",
            "extra": "mean: 2.583626318060438 msec\nrounds: 371"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 254.09041861819836,
            "unit": "iter/sec",
            "range": "stddev: 0.00005880056020498145",
            "extra": "mean: 3.9356068813543934 msec\nrounds: 236"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 1198.6566257174898,
            "unit": "iter/sec",
            "range": "stddev: 0.000023103373156758486",
            "extra": "mean: 834.267277671303 usec\nrounds: 1048"
          }
        ]
      }
    ]
  }
}