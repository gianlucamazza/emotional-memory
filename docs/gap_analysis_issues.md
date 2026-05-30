# Gap Analysis — Issues to Create

> **Status: planning snapshot (not a live tracker).** This is a point-in-time gap
> analysis kept for historical reference; it is intentionally not linked from the
> docs navigation. Items are triaged into GitHub issues as they are picked up
> (e.g. arXiv submission → #31). Do not treat unchecked items here as the
> authoritative backlog — the issue tracker is the source of truth.

This document captures the gap analysis performed on the emotional-memory project.
Issues are created to track each item as it is prioritized.

## Priority: HIGH (Visibility & Communication)

### 1. Add "When NOT to Use" Section to README
- **Scope**: Clarify boundaries of AFT advantage (oracle affect requirement, LoCoMo FAIL, LLM appraisal limitations)
- **Audience**: Potential adopters making architecture decisions
- **Location**: README, after "How it Compares" comparative table

### 2. Configuration Decision Tree Guide
- **Scope**: Tutorial or guide on selecting EmotionalMemoryConfig values for common scenarios
- **Problem**: 8 nested config classes + 80+ parameters; users need heuristics
- **Deliverable**: `docs/tutorials/configuration_guide.md` with decision tree

### 3. Complete arXiv Submission (Issue #31)
- **Blocker**: `release.toml: arxiv_id` still empty
- **Action**: Execute upload to arXiv; unblock post-submission metadata sync (badge, CITATION.cff sync)

### 4. Link Limitations from README Hero Section
- **Current**: `docs/research/08_limitations.md` exists but is not discoverable from README
- **Action**: Add explicit link after Current Validation Status section

## Priority: MEDIUM (Documentation Completeness)

### 5. Troubleshooting Guide for Retrieval Failures
- **Scope**: Common failure modes and debugging patterns
- **Topics**: Why am I not getting the memories I expect? How to inspect retrieval signals?
- **Deliverable**: `docs/troubleshooting.md` with diagnostic checklist

### 6. LLM Appraisal Edge-Case Mitigation Strategy
- **Problem**: Hg1 FAIL shows end-to-end LLM appraisal underperforms (Δ=−0.010, sync mode harmful at −18.5 pp)
- **Scope**: When/why to fall back to KeywordAppraisalEngine; how to detect degradation
- **Deliverable**: Tutorial + updated `docs/research/09_current_evidence.md` callout

### 7. Multilingual Deployment Best Practices
- **Scope**: Evidence exists (IT, ES, FR slices) but no guide
- **Topics**: Language selection, embedder choice (SBERT vs me5), expected performance deltas
- **Deliverable**: `docs/guides/multilingual.md`

### 8. Performance Scaling Guide (InMemory → SQLite → Qdrant)
- **Scope**: Decision tree: when to use each store; latency/throughput tradeoffs
- **Deliverable**: `docs/guides/performance_scaling.md` + README link

### 9. Verify All Tutorial Links in README Exist
- **Problem**: Some linked docs may not exist in tree (e.g., `docs/tutorials/mem0.md`, `docs/tutorials/langchain.md`)
- **Action**: Audit README links; create missing tutorials or remove dead links

### 10. Clarify mem0 Integration Pattern
- **Current**: EmotionalMemoryMem0Backend exists but pattern is vague ("chain a real mem0.Memory instance")
- **Scope**: Working example with pattern explanation
- **Deliverable**: Enhanced `docs/tutorials/mem0.md` with runnable pattern + example

## Priority: MEDIUM (Internal Quality)

### 11. Add Performance Regression Testing to CI
- **Scope**: Establish baseline for key operations (encode, retrieve, retrieval+reconsolidation)
- **Action**: Alert if >10% regression on subsequent runs
- **Rationale**: Ensure performance guarantees remain valid across versions

### 12. Gate 2 (Human Evaluation) Disposition Decision
- **Status**: Kit shipped in v0.7, zero raters recruited, deferred to v1.0
- **Action**: Confirm timeline/budget; if deprioritized, close issue or move to v1.0 milestone with explicit rationale
- **Blocker**: Ambiguity on when/if human validation will happen

### 13. Expand Comparison Table: Codebase Size Breakdown
- **Current**: README shows ~4.8k LOC src vs 50k+ for competing systems
- **Action**: Add breakdown: src/ vs tests/ vs benchmarks/; highlight test/code ratio (quality signal)

## Priority: LOW (Roadmap & Futures)

### 14. v0.x → v1.0 Migration Guide (Early Draft)
- **Scope**: Preview of what will change at v1.0.0 (public-API freeze)
- **Audience**: Early adopters planning production deployments
- **Timeline**: Can be deferred; placeholder for v1.0 roadmap

### 15. Configuration Preset Packs
- **Idea**: Pre-composed `SimpleConfig`, `ResearchConfig`, `ProductionConfig` to reduce cognitive load
- **Status**: Nice-to-have; low priority unless adoption feedback requests it

---

## Notes for Issue Creation

- Use professional English
- Tag @copilot in each issue
- Include clear acceptance criteria
- Link to existing docs where relevant
- Specify if issue is blocker, dependency, or independent
- Use labels: `documentation`, `enhancement`, `blocker` (if applicable)
