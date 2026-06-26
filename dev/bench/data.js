window.BENCHMARK_DATA = {
  "lastUpdate": 1782510315282,
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
          "id": "c746f80e75e7970b16cbb9e3907cf673032f2536",
          "message": "feat(bench): add A5 human-gold appraisal vs EmoBank, LLM valence validated (#62) (#66)\n\n* feat(bench): add A5 human-gold appraisal benchmark vs EmoBank (Addendum S, #62)\n\nPre-registers and implements the construct-validity check A5 recorded as future\nwork: how well does the appraisal pipeline reproduce human-annotated affect?\n\n- benchmarks/datasets/emobank_v1.json: 300 rows sampled (seed=42) from EmoBank\n  (CC-BY-SA 4.0, Buechel & Hahn 2017), human VAD 1-5 mapped to CoreAffect ranges.\n- preregistration_addendum_s_human_gold_appraisal.md: Hs1 (Pearson r per\n  dimension vs human gold), bias + MAE, pre-declared bootstrap plan, decision rule.\n- benchmarks/human_gold_appraisal/runner.py: LLMAppraisalEngine + KeywordAppraisalEngine\n  -> to_core_affect(), per-dimension Pearson r / bias / MAE with bootstrap CIs.\n- Makefile: bench-human-gold (llm-config-strict) + bench-human-gold-dry.\n\nResults + closure + appraisal-claim scoping land in a follow-up commit once the\nconfirmatory run completes.\n\nRefs #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n* feat(bench): A5 human-gold appraisal — LLM valence human-validated (#62)\n\nConfirmatory run of Addendum S (appraisal vs human-annotated affect) on EmoBank\n(human VAD, N=300, CC-BY-SA 4.0), bootstrap n=2000 seed=42:\n\n- LLM engine: valence r=0.703 [0.655, 0.746] (human-validated), arousal r=0.276\n  [0.172, 0.373], dominance r=0.333 [0.224, 0.432] — all CIs exclude 0.\n- The +0.169 positive valence bias (Addendum N, vs LLM-gold) persists vs HUMAN\n  gold (+0.152) — a standing limitation (Addendum O recalibration not in path).\n- KeywordAppraisalEngine is NOT human-validated (valence r=0.070).\n\nThis is the first non-circular validation of the affect signal against human\nlabels. Valence is the trustworthy dimension; arousal/dominance are weak-positive;\nno \"accurate appraisal\" wording beyond the validated valence dim, bias stated.\n\n- results.{json,md} + Addendum S closure.\n- claim_validation_matrix.json: new claim appraisal_human_validated =\n  early_controlled_evidence; appraisal_directionally_useful evidence extended.\n- 09_current_evidence.md: new claim row (id + exact allowed wording).\n- problem_register §A5: future-work item now executed, scoped honestly.\n\nCloses #62\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-26T23:38:39+02:00",
          "tree_id": "bb04cf8f1496143a32b1ff178976135e1e0d83a9",
          "url": "https://github.com/gianlucamazza/emotional-memory/commit/c746f80e75e7970b16cbb9e3907cf673032f2536"
        },
        "date": 1782510313668,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_single",
            "value": 560.7999171894597,
            "unit": "iter/sec",
            "range": "stddev: 0.0008405470735271737",
            "extra": "mean: 1.78316716773366 msec\nrounds: 1562"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_with_resonance",
            "value": 434.1633130004299,
            "unit": "iter/sec",
            "range": "stddev: 0.0011377729890550009",
            "extra": "mean: 2.303280747258831 msec\nrounds: 1824"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_no_resonance",
            "value": 327.8747325048106,
            "unit": "iter/sec",
            "range": "stddev: 0.0019452366855648832",
            "extra": "mean: 3.0499453018549634 msec\nrounds: 2372"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[10]",
            "value": 196.773973739463,
            "unit": "iter/sec",
            "range": "stddev: 0.0036435887540902977",
            "extra": "mean: 5.081972889992261 msec\nrounds: 3827"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[100]",
            "value": 455.52862755875293,
            "unit": "iter/sec",
            "range": "stddev: 0.0010175876427604177",
            "extra": "mean: 2.1952517130682914 msec\nrounds: 1760"
          },
          {
            "name": "benchmarks/perf/bench_encode.py::bench_encode_scaling[1000]",
            "value": 387.4983305421216,
            "unit": "iter/sec",
            "range": "stddev: 0.0003022761507904444",
            "extra": "mean: 2.580656279476019 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_memory_per_record",
            "value": 136889.62435776068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014183002655994415",
            "extra": "mean: 7.3051555564686375 usec\nrounds: 49339"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[100]",
            "value": 28.39886677919485,
            "unit": "iter/sec",
            "range": "stddev: 0.00020365623251475919",
            "extra": "mean: 35.21267266666447 msec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[1000]",
            "value": 0.8135353488293926,
            "unit": "iter/sec",
            "range": "stddev: 0.026923765468591185",
            "extra": "mean: 1.2292028876666687 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_footprint.py::bench_store_footprint[5000]",
            "value": 0.026689937109911518,
            "unit": "iter/sec",
            "range": "stddev: 2.060974027224613",
            "extra": "mean: 37.46730447066667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[50]",
            "value": 3222.6460810629374,
            "unit": "iter/sec",
            "range": "stddev: 0.00001300761334974558",
            "extra": "mean: 310.30400945243304 usec\nrounds: 2010"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[200]",
            "value": 875.1717811557047,
            "unit": "iter/sec",
            "range": "stddev: 0.000015250732928544302",
            "extra": "mean: 1.142632819672789 msec\nrounds: 671"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_resonance_build[500]",
            "value": 354.6648558334884,
            "unit": "iter/sec",
            "range": "stddev: 0.00008428784136475703",
            "extra": "mean: 2.819563268116675 msec\nrounds: 276"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_encode_with_large_resonance_graph",
            "value": 493.4105310964748,
            "unit": "iter/sec",
            "range": "stddev: 0.000521306790223933",
            "extra": "mean: 2.026709883507682 msec\nrounds: 764"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-1]",
            "value": 10338.777959744604,
            "unit": "iter/sec",
            "range": "stddev: 0.000012279992271768472",
            "extra": "mean: 96.7232301432173 usec\nrounds: 6018"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[100-2]",
            "value": 10551.918767951776,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067906487275285935",
            "extra": "mean: 94.76949377559596 usec\nrounds: 8354"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-1]",
            "value": 1824.2958423259038,
            "unit": "iter/sec",
            "range": "stddev: 0.000010264782655921165",
            "extra": "mean: 548.1567061650704 usec\nrounds: 1038"
          },
          {
            "name": "benchmarks/perf/bench_resonance.py::bench_spreading_activation[500-2]",
            "value": 1806.4833371545078,
            "unit": "iter/sec",
            "range": "stddev: 0.0001029134626644825",
            "extra": "mean: 553.5617071205072 usec\nrounds: 1615"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[100]",
            "value": 1007.9644626601065,
            "unit": "iter/sec",
            "range": "stddev: 0.00009776509000194609",
            "extra": "mean: 992.0984687901719 usec\nrounds: 785"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[1000]",
            "value": 372.12951115013726,
            "unit": "iter/sec",
            "range": "stddev: 0.0001875947405423162",
            "extra": "mean: 2.6872364863224885 msec\nrounds: 329"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_top5[10000]",
            "value": 21.132311941906966,
            "unit": "iter/sec",
            "range": "stddev: 0.001161735225050028",
            "extra": "mean: 47.32089904545298 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[1]",
            "value": 472.6639403994937,
            "unit": "iter/sec",
            "range": "stddev: 0.00005214316598041101",
            "extra": "mean: 2.115668056155932 msec\nrounds: 463"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[5]",
            "value": 372.6965598126749,
            "unit": "iter/sec",
            "range": "stddev: 0.00019039939571745696",
            "extra": "mean: 2.6831479220055616 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[10]",
            "value": 309.6063314465392,
            "unit": "iter/sec",
            "range": "stddev: 0.00009880119243388391",
            "extra": "mean: 3.229908107265802 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_varying_topk[25]",
            "value": 198.4182305433595,
            "unit": "iter/sec",
            "range": "stddev: 0.000048626156655326086",
            "extra": "mean: 5.039859478947799 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/perf/bench_retrieve.py::bench_retrieve_with_reconsolidation",
            "value": 864.6004373366848,
            "unit": "iter/sec",
            "range": "stddev: 0.000024347524282298127",
            "extra": "mean: 1.1566036249997746 msec\nrounds: 824"
          }
        ]
      }
    ]
  }
}