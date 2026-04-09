# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-09

### Added

- **Affective Field Theory (AFT)** — original 5-layer emotional model for LLM memory systems
- `CoreAffect` — continuous valence/arousal circumplex (Barrett/Russell)
- `AffectiveMomentum` — velocity and acceleration of affect transitions (Spinoza)
- `StimmungField` — slow-moving global mood with inertia, updated via EMA (Heidegger/PAD)
- `AppraisalVector` — emotion derived from cognitive evaluation (Scherer/Lazarus/Stoics)
- `ResonanceLink` — associative memory graph (semantic, emotional, temporal, causal, contrastive)
- `EmotionalTag` — snapshot of all 5 layers at encoding time + consolidation metadata
- `EmotionalMemory` — main facade with `encode()`, `retrieve()`, `get_state()`, `set_affect()`
- `InMemoryStore` — dict-backed `MemoryStore` with brute-force cosine search
- `Embedder` and `MemoryStore` — `typing.Protocol` interfaces for dependency injection
- Power-law memory decay (ACT-R style), arousal-modulated, with high-arousal floor
- Mood-congruent retrieval via 6-signal weighted scoring with Stimmung-adaptive weights
- Reconsolidation: retrieval updates memory tag when Affective Prediction Error exceeds threshold
- Resonance link builder: composite semantic + emotional + temporal scoring
- 135 unit and integration tests
