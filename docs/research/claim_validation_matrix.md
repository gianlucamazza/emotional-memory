# Claim Validation Matrix

The claim validation matrix is the authoritative record of all empirical claims made
by the Affective Field Theory (AFT) library, their current evidence status, and the
references to benchmark artefacts that support or qualify each claim.

Each entry carries:

- `claim_text` — the exact public-facing claim
- `evidence_refs` / `benchmark_refs` / `protocol_refs` / `limitations_refs` — file
  paths within the repository that serve as evidence (all paths validated in CI via
  `tools/audit_claim_refs.py`)
- `current_evidence` — evidence classification (e.g. `strong_intra_theory`,
  `strong_cross_task`, `not_yet_shown`)
- `not_yet_shown` — explicit scope limitations attached to each claim
- `allowed_public_wording` — the precise wording permitted in communications
- `next_study` — reference to any planned follow-up pre-registration

[Download raw JSON](claim_validation_matrix.json){ .md-button }

## Validation tool

```bash
uv run python tools/audit_claim_refs.py
```

Exits 0 if all referenced paths exist on disk; exits 1 with a summary of missing
paths. Integrated into `make meta-check` and therefore into `make check`.
