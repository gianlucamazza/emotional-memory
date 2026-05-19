# Figure Inventory

Source of truth: [`figure_inventory.json`](figure_inventory.json). This document is a human-readable view — edit the JSON, not this table.

Three distinct categories of figures are produced by three separate generators. Each generator writes to its own directory; they do not cross-populate each other's directories by default.

## Category roles

| Category | Generator | Output directory | Format | Purpose |
|----------|-----------|-----------------|--------|---------|
| `didactic` | `scripts/generate_docs_images.py` | `docs/images/` | PNG | API docs, README, docs site |
| `evidence` | `scripts/generate_research_figures.py` | `docs/images/research/` | PNG + PDF | Research docs, benchmark evidence |
| `paper` | `scripts/generate_paper_figures.py` | `paper/figures/` | PDF | `paper/main.tex` only |

To place evidence figures inside `paper/figures/`, invoke the generator explicitly with `--pdf-dir paper/figures`. This is not done automatically.

## Regeneration commands

```bash
make docs-images       # didactic figures only
make research-figures  # evidence figures only
make paper-figures     # paper figures only
make figures           # all three
```

## Didactic figures (`docs/images/`)

| Stem | Plot kind | Description | Referenced in |
|------|-----------|-------------|---------------|
| `circumplex` | scatter | Russell (1980) circumplex with sample memory positions | README.md |
| `decay_curves` | line | ACT-R power-law decay at three arousal levels | README.md |
| `yerkes_dodson` | line | Yerkes-Dodson arousal-performance curve | README.md |
| `retrieval_radar` | radar | Six-signal retrieval score breakdown | README.md |
| `mood_evolution` | line\_multipanel | PAD mood-field trajectory over 30 minutes | README.md |
| `adaptive_weights_heatmap` | heatmap | Adaptive weight distribution across valence/arousal space | README.md |
| `resonance_network` | network\_graph | Resonance link graph with five link types | README.md |
| `appraisal_radar` | radar | Scherer CPM five-dimensional appraisal vector | README.md |

All didactic figures use synthetic/hardcoded inputs — no benchmark JSON.

## Evidence figures (`docs/images/research/`)

| Stem | Plot kind | Source benchmark(s) | Claim | Referenced in |
|------|-----------|---------------------|-------|---------------|
| `research_realistic_v2_overview` | bar\_with\_ci | `benchmarks/realistic/results.v2.sbert.json` | `replayable_multi_session_help` | README.md, `09_current_evidence.md` |
| `research_realistic_v2_challenges` | grouped\_bar | `benchmarks/realistic/results.v2.sbert.json` | `replayable_multi_session_help` | README.md, `09_current_evidence.md` |
| `research_ablation_s3` | bar\_forest | `benchmarks/ablation/results.v2.sbert.json` | `theory_faithful_operationalization` | README.md, `09_current_evidence.md` |
| `research_multilingual` | grouped\_bar\_with\_ci | `results.v2_{it,es}.{sbert,me5}.json` (N=80) + `results.v2_fr.me5.json` (N=120 Hm1) | `retrieval_affect_aware`, `cross_domain_affect_replication` | README.md, `09_current_evidence.md` |
| `research_locomo_negative` | grouped\_bar\_with\_ci | `benchmarks/locomo/results.json` | `locomo_external_qa_negative` | README.md, `09_current_evidence.md` |

Claim IDs map to `docs/research/claim_validation_matrix.json`.

## Paper figures (`paper/figures/`)

| Stem | Plot kind | Description | Referenced in |
|------|-----------|-------------|---------------|
| `fig1_circumplex` | scatter | Russell circumplex — 12 seeded memories | `paper/main.tex` |
| `fig2_decay_curves` | line | ACT-R decay modulated by arousal | `paper/main.tex` |
| `fig3_mood_evolution` | line\_multipanel | PAD trajectory over simulated conversation | `paper/main.tex` |
| `fig4_resonance_network` | network\_graph | Resonance link graph snapshot | `paper/main.tex` |

Paper figures are schematic/synthetic (seeded random data); they do not reference benchmark JSON.

## Planned evidence figures (Fase 2)

The following benchmark JSON artefacts exist but are not yet visualised:

| Planned stem | Source JSON | Claim |
|---|---|---|
| `research_sota_realistic_v2` | `benchmarks/comparative/results.sota.v2.sbert.json` | `realistic_replay_vs_sota` |
| `research_dailydialog_negative` | `benchmarks/dailydialog/results.json` | `cross_domain_affect_replication` |
| `research_hi3_resonance` | `benchmarks/ablation/results.hi3.json` | `resonance_amplification_e5` |
| `research_hd2_multilingual_power` | `benchmarks/appraisal_confound/results.hd2_{it,es}.me5{,.v120}.json` | `retrieval_affect_aware` |
| `research_locomo_pareto` | `benchmarks/locomo/pareto_results.json` | `locomo_external_qa_negative` |
