# LoCoMo Benchmark Results

Dataset: locomo10 (10 conversations, 1986 QA pairs)

## Aggregate Scores

| System | N | F1 | BLEU-1 | Judge Acc |
|---|---:|---:|---:|---:|
| `aft` | 1540 | 0.168 | 0.169 | 0.279 |
| `naive_rag` | 1540 | 0.271 | 0.263 | 0.441 |

## By Category

### multi_hop

| System | N | F1 | BLEU-1 | Judge Acc |
|---|---:|---:|---:|---:|
| `aft` | 282 | 0.172 | 0.200 | 0.262 |
| `naive_rag` | 282 | 0.227 | 0.231 | 0.351 |

### open_domain

| System | N | F1 | BLEU-1 | Judge Acc |
|---|---:|---:|---:|---:|
| `aft` | 96 | 0.092 | 0.095 | 0.188 |
| `naive_rag` | 96 | 0.102 | 0.105 | 0.198 |

### single_hop

| System | N | F1 | BLEU-1 | Judge Acc |
|---|---:|---:|---:|---:|
| `aft` | 841 | 0.221 | 0.208 | 0.371 |
| `naive_rag` | 841 | 0.379 | 0.350 | 0.599 |

### temporal

| System | N | F1 | BLEU-1 | Judge Acc |
|---|---:|---:|---:|---:|
| `aft` | 321 | 0.049 | 0.064 | 0.081 |
| `naive_rag` | 321 | 0.077 | 0.108 | 0.178 |

## Hypothesis Tests (pre-registration S1)

Gate 1: **FAIL**  
(n_bootstrap=2000, Holm-Bonferroni correction across H1+H2)

| Hypothesis | Metric | AFT | Baseline | Δ [95%CI] | p_one | p_adj | Result |
|---|---|---:|---:|---|---:|---:|---:|
| **H1** | f1 | 0.169 | 0.270 | -0.101 [-0.118, -0.086] | 1.0000 | 0.0000 | **✗ FAIL** |
| **H2** | judge_accuracy | 0.281 | 0.440 | -0.159 [-0.183, -0.136] | 1.0000 | 0.0000 | **✗ FAIL** |
