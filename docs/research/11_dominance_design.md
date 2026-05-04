# Dominance Dimension — Design Note (G7)

**Status:** IMPLEMENTED in v0.8.2 — commit `8b9ddbe` (2026-05-04). Option A shipped.
**References:** Mehrabian & Russell (1974); `benchmarks/fidelity/test_pad_dominance.py`;
`benchmarks/fidelity/test_dominance_retrieval_gap.py`.

---

## 1. State before v0.8.2 (historical)

`CoreAffect` was a **2-dimensional** (valence × arousal) value object.
Dominance lived only in `MoodField` as a derived scalar via the PAD heuristic:

```
dominance_signal = 0.5 + 0.5 * valence * arousal
```

`MoodField.dominance` drove no retrieval signal. Two memories encoded at
identical (valence, arousal) but with semantically opposite dominance —
e.g., *"I seized control of the argument"* (high dominance) vs *"I felt helpless"*
(low dominance) — received **identical** `CoreAffect` fingerprints and were
indistinguishable by any of AFT's 6 retrieval signals.

This gap was operationalised by `test_dominance_retrieval_gap.py` (xfail until
v0.8.2).

---

## 2. The gap

The gap is operationalised in `benchmarks/fidelity/test_dominance_retrieval_gap.py`
(currently `xfail`): a high-dominance query cannot preferentially retrieve
high-dominance memories when both memories share the same valence and arousal.

This matters for:
- **Assertiveness vs helplessness** — semantically opposite but mood-matched
- **Authority vs submission** contexts in conversation memory
- **Approach/avoidance** patterns that PAD theory attributes to dominance, not to
  valence or arousal alone

---

## 3. Design options

### Option A: Elevate dominance to a `CoreAffect` primary dimension

Add `dominance: float` (clamped to [0, 1]) to `CoreAffect` alongside valence
and arousal.

Implications:
- `CoreAffect.distance()` / `CoreAffect.similarity()` become 3D — weighted
  Euclidean in (valence, arousal, dominance) space, e.g., weights (1.0, 1.0, 0.5)
  reflecting PAD's empirical finding that dominance has lower cross-situational
  variance than valence.
- `AppraisalVector → CoreAffect` must produce a dominance estimate; the
  Scherer CPM prompt must be extended.
- `KeywordAppraisalEngine` needs dominance rules (authority/control vs
  helplessness/submission vocabulary).
- `EmotionalTag` and all serialisation paths gain a `dominance` field.
- Migration: existing memories decoded without dominance default to 0.5
  (neutral).

### Option B: Keep 2D CoreAffect; add dominance as a separate retrieval signal

Keep `CoreAffect` 2D; add `dominance: float` to `EmotionalTag` and
`AffectiveState`, and add a 7th retrieval signal "dominance proximity" that
computes distance between query dominance and stored memory dominance.

Implications:
- Lower migration cost (CoreAffect stays stable).
- Dominance proximity requires a dominance estimate at retrieval time
  (from query appraisal or user-set affect).
- Signal can be toggled off (adds `enable_dominance: bool` to
  `RetrievalConfig`), consistent with the ablation framework.

---

## 4. Decision

**Preferred: Option A for v0.8.0**, with these guards:

1. The existing `test_pad_dominance.py` suite must continue passing
   (regression guard on `MoodField` behaviour).
2. `test_dominance_retrieval_gap.py` becomes the acceptance test: it is
   currently `xfail(strict=True)` and must PASS after implementation.
3. A migration script or auto-defaulting must preserve existing
   serialised memories (dominance=0.5 for legacy records).
4. Appraisal prompt extension is behind the `LLMAppraisalEngine` path;
   `KeywordAppraisalEngine` gets rule-based dominance heuristics (authority
   and control vocabulary).
5. The `AppraisalVector.dominance` field already exists — it only needs to
   be wired into `CoreAffect` production in `engine.encode()`.

---

## 5. Evidence needed before implementation

- At least one fidelity scenario where dominance-differential memories share
  valence/arousal (provided by `test_dominance_retrieval_gap.py`).
- Smoke test confirming that existing round-trip serialisation (encode →
  export → import → retrieve) is dominance-neutral at default 0.5.
- No regression on any of the 126 existing fidelity cases.

---

## 6. Interaction with G4/G5 (realistic_recall_v2)

`realistic_recall_v2.json` includes the `momentum_alignment` challenge type.
Dominance is NOT required for `momentum_alignment`; that challenge relies on
`AffectiveMomentum` (velocity + acceleration over valence/arousal history).
The v2 dataset should therefore NOT require dominance-specific scenarios — it
remains compatible with the current 2D `CoreAffect`.

Adding a dominance-specific challenge type would be a natural extension for
`realistic_recall_v3` (post v0.8.0).

---

## 7. Closure — v0.8.2 implementation (2026-05-04)

Option A was shipped in commit `8b9ddbe`. Implementation summary:

- `CoreAffect` is now **3D** (valence × arousal × dominance). `dominance: float = 0.5`
  (neutral default, clamped [0, 1] by Pydantic validator).
- `_MAX_AFFECT_DIST` in `retrieval.py` updated √5 → √6 (Euclidean normalizer for 3D PAD).
- `AppraisalVector.coping_potential` maps directly to `CoreAffect.dominance` (no signed
  conversion needed — same [0, 1] scale).
- `MoodField` reads `core_affect.dominance` directly in the EMA update (no heuristic).
- `AffectiveMomentum` gains `d_dominance` and `dd_dominance` fields.
- `AffectiveState._history` tuples extended to 4-element `(ts, v, a, d)`.
- `test_dominance_retrieval_gap.py` xfail removed — test passes.
- `test_pad_dominance.py` rewritten to test EMA-direct behaviour (not the old heuristic).

The backward-compat migration path (Option A §3 above) was intentionally skipped:
existing serialised states with 3-element history tuples are rejected on restore
(`len(entry) >= 4` guard). Fresh encode → export → import round-trips use the 4-element
format natively.
