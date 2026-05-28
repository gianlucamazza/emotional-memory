---
name: "Issue #32: Add 'When NOT to Use' section to README"
about: Document AFT boundaries and design limitations for potential adopters
labels: ["documentation", "enhancement"]
---

# Add "When NOT to Use" Section to README

**Priority**: HIGH  
**Audience**: Potential adopters making architecture decisions  
**Tagging**: @copilot

## Problem Statement

The README thoroughly documents AFT strengths (oracle affect scenarios, multilingual slices, fidelity tests) but does **not clearly surface the boundaries where AFT does not provide advantages**. Potential adopters may not discover these limitations until after implementing.

### Key Limitations Not Surfaced in README Hero

1. **LoCoMo (Factual QA) — FAIL**: AFT F1=0.168 vs naive_rag=0.271 (−10.3 pp). Affective weighting does not improve factual open-domain question-answering.
2. **LLM Appraisal End-to-End — FAIL (Hg1)**: With automatic LLM appraisal, AFT top1=0.315 vs naive_cosine=0.325 (Δ=−0.010). Synchronous appraisal is actively harmful (−18.5 pp).
3. **Short-Turn Dialogue — FAIL (Hk1)**: On naturalistic DailyDialog (120 personas, 396 queries), AFT shows no advantage over cosine (Δ=−0.008).
4. **Query-Type Routing — FAIL (Addendum L)**: Heuristic routing does not improve aggregate F1 over fixed weights.

## Current State

- ✅ Individual failures **are documented** in "Current validation status" section
- ❌ But the narrative flow emphasizes successes; failure modes require scrolling and cross-referencing
- ❌ No single "When NOT to use" or "Design boundaries" section

## Scope

Add a new section in README after or adjacent to "How it Compares" comparative table:

### Proposed Structure

```markdown
## When NOT to Use emotional_memory

AFT is designed for **affective recall** where emotional congruence matters.
It is **not** a general-purpose memory system. Consider alternatives if:

- **You need factual QA performance**: LoCoMo benchmark shows F1 0.168 vs naive_rag 0.271 (Hg1).
- **You cannot provide affect signals**: AFT advantage requires oracle affect or accurate appraisal.
  End-to-end LLM appraisal does not transfer the gain (Hg1 FAIL).
- **Your domain is short-turn dialogue**: DailyDialog (Hk1) shows no advantage over cosine on naturalistic data.
- **You need query-type adaptive routing**: Query-type routing does not close the gap vs naive_rag (Addendum L FAIL).

**Recommended for:**
- Multi-session, episodic memory with emotional context
- Domains where mood-congruent retrieval matters (journaling, conversational agents with long-term state)
- Oracle affect or high-quality appraisal available at encode time

**For comparisons with other systems**, see [Current validation status](#current-validation-status) and
[docs/research/09_current_evidence.md](docs/research/09_current_evidence.md).
```

## Acceptance Criteria

- [ ] New "When NOT to use" section added to README after "How it Compares"
- [ ] All four FAIL conditions (LoCoMo, Hg1, Hk1, Addendum L) explicitly mentioned with brief explanation
- [ ] Clear recommendation for "Recommended for:" scenarios
- [ ] Links to full evidence in `docs/research/09_current_evidence.md` and `claim_validation_matrix.json`
- [ ] No content removed; only additive clarity
- [ ] Section reviewed for tone: informative, not defensive

## Related Issues

- #31: v0.7.x arXiv submission (related: post-publication communication strategy)
- ROADMAP.md v0.10.x, v0.11.0 (reference for design decisions)

## References

- `docs/research/09_current_evidence.md` (study ladder)
- `docs/research/claim_validation_matrix.json` (canonical claim status)
- `benchmarks/comparative/protocol.md` (interpretation rules)
- README § "Current validation status" (existing documentation)
