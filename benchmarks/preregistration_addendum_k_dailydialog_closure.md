# Addendum K — DailyDialog Affect-Conditioned Retrieval: Closure

**Date:** 2026-05-07
**Pre-registration:** `benchmarks/preregistration_addendum_k_dailydialog.md` (commit `59917e2`)
**Status:** CLOSED — **Branch B (FAIL)**

---

## 1. Executive Summary

The Hk1 hypothesis — that AFT `top1_accuracy` > naive cosine on DailyDialog affect-conditioned
retrieval at N=120 personas (396 queries, `multilingual-e5-small`, top_k=2) — **is not supported**.

| Metric | AFT | naive_cosine | Δ | 95 % CI | p_holm | d | Verdict |
|---|---|---|---|---|---|---|---|
| aggregate (top1) | 0.212 | 0.220 | −0.008 | [−0.056, +0.043] | 1.000 | −0.015 | **FAIL** |

Decision rule (pre-reg): `p_holm < 0.05 AND Δ > 0 AND CI not crossing 0 AND ≥2/3 directional types PASS` — **none** of the four conditions hold at the aggregate level (Δ < 0). Branch B is confirmed.

---

## 2. Full Results by Query Type

| Query type | N | AFT top1 | Cosine top1 | Δ | CI | p_holm | d | Verdict |
|---|---|---|---|---|---|---|---|---|
| emotion_state_recall | 120 | 0.217 | 0.225 | −0.008 | [−0.092, +0.075] | 1.000 | −0.017 | FAIL |
| affect_conditioned_content | 120 | 0.175 | 0.217 | −0.042 | [−0.125, +0.042] | 1.000 | −0.088 | FAIL |
| affective_trajectory | 39 | 0.385 | 0.282 | **+0.103** | [−0.077, +0.282] | 0.734 | +0.186 | FAIL (N=39) |
| cross_session_control | 117 | 0.188 | 0.197 | −0.009 | [−0.094, +0.077] | 1.000 | −0.017 | FAIL |

**Bootstrap:** paired, n=10 000, seed=0. **McNemar:** aggregate p=0.839. **Holm:** m=4 (aggregate + 3 directional types). One-tailed (directional, positive Δ).

The only type with a positive point estimate is `affective_trajectory` (Δ=+0.103, d=0.186), but with N=39 the CI crosses zero and p_holm=0.734. This is a weak signal consistent with the theoretical prediction (trajectory needs momentum encoding to distinguish) but falls well short of the pre-declared threshold.

---

## 3. Interpretation

**Why does AFT not win on DailyDialog?**

1. **Regime mismatch.** The realistic_recall_v2 benchmark was constructed to maximise AFT's theoretical advantage: scenarios explicitly designed around affective arcs, momentum shifts, and valence trajectories with hard semantic distractors. DailyDialog turns are short (2–3 sentences on average), lexically diverse, and the emotion labels are sparse (≈17 % of turns carry a non-zero emotion label after the ≥30 % density filter). There is insufficient affective signal in short turns for AFT's 6-signal scorer to outperform cosine similarity on raw semantic content.

2. **PAD injection at session level is coarse.** The AFT adapter injects the dominant-emotion PAD via `set_affect()` before encoding each session. On DailyDialog, the "dominant emotion" is often `happiness` (most common non-zero label) regardless of actual conversational dynamics, making the PAD signal weakly discriminative across sessions.

3. **Cosine baseline is not weak.** At top_k=2 with 4–5 sessions per persona, naive cosine already achieves top1≈0.22. The retrieval task over 4–5 sessions is relatively easy for a semantic baseline; any marginal AFT advantage over cosine may require larger session pools or richer affective content.

4. **`affective_trajectory` positive trend is coherent.** The one type where AFT shows a positive trend (Δ=+0.103, d=0.186) is precisely the type most aligned with the momentum-based retrieval signal: it requires identifying sessions with a valence direction shift. The underpowered N=39 is a consequence of the strict valence-sign-change constraint.

---

## 4. What This Changes

| Component | Before | After |
|---|---|---|
| Cross-domain claim | "AFT generalises to affect-laden ecological dialogue" (untested) | **Not established at N=120 DailyDialog** |
| Paper §6 | Absent (DailyDialog was pending) | Add §6.X with Branch B result |
| Paper §8 Limitations | "regime limited to curated benchmarks" | Strengthened: DailyDialog FAIL confirms regime specificity |
| `claim_validation_matrix.json` | `cross_domain_affect_replication: pending` | → `partial_negative` (FAIL on DailyDialog; EN curated benchmark PASS) |
| `09_current_evidence.md` | No Hk1 row | Add Hk1 row with FAIL |

---

## 5. What This Does NOT Change

- **EN headline (realistic_recall_v2, sbert, N=200):** Δ=+0.21, d=0.49, p<0.001 — unaffected.
- **Cross-embedder EN (e5-small-v2):** Δ=+0.16 — unaffected.
- **SOTA comparison (affect_reference_v1):** unaffected.
- **Hi3 resonance amplification (N=500):** unaffected.
- **Hg1 LLM-free FAIL:** unaffected.
- **Hd2 N=120 Branch C (cross-language FAIL):** already closed; consistent with Hk1.

---

## 6. `affective_trajectory` Follow-up Signal (Out-of-Scope)

The `affective_trajectory` type (Δ=+0.103, d=0.186, N=39) is noted as an exploratory positive signal. It is **not** reported as a confirmatory result (pre-reg required p_holm<0.05 and N was too small for power). Future work could pre-register a focused hypothesis on trajectory-type queries with a larger sample. This is recorded as an open direction, not a partial PASS.

---

## 7. Dataset and Artefact Integrity

- Pre-registration committed before any code or data: `59917e2` (2026-05-07).
- Persona dataset built with `seed=0` from `roskoN/dailydialog` (train+validation splits, 12118 dialogs, 2580 after ≥30 % emotion-density filter).
- No LLM in query generation; all queries programmatic from emotion labels.
- Benchmark run with `seed=0`; determinism verified (two runs produce identical numbers).
- Results files: `benchmarks/dailydialog/results.{json,md,protocol.json}`.

---

## 8. Paper / Docs Action Items

1. **`paper/main.tex`** — new §6.X: "Cross-domain ecological replication (DailyDialog)"; §8 strengthened; abstract unchanged (EN headline unaffected).
2. **`docs/research/claim_validation_matrix.json`** — `cross_domain_affect_replication: partial_negative`.
3. **`docs/research/09_current_evidence.md`** — Hk1 row.
4. **`docs/research/13_dailydialog_followup.md`** (new) — extended writeup.
5. **`README.md`, `ROADMAP.md`, `CHANGELOG.md`** — Hk1 Branch B status.
