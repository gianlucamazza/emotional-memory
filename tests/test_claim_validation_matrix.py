from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_matrix() -> dict[str, object]:
    matrix_path = _repo_root() / "docs" / "research" / "claim_validation_matrix.json"
    return json.loads(matrix_path.read_text(encoding="utf-8"))


def test_claim_validation_matrix_has_expected_shape() -> None:
    payload = _load_matrix()

    assert payload["matrix_version"] == "1.0"
    assert "status_legend" in payload
    assert "evidence_level_legend" in payload
    claims = payload["claims"]
    assert isinstance(claims, list)
    assert claims


def test_claim_validation_matrix_uses_controlled_enums() -> None:
    payload = _load_matrix()
    claims = payload["claims"]

    valid_statuses = {
        "implemented",
        "controlled_evidence",
        "strong_intra_theory_evidence",
        "early_controlled_evidence",
        "not_established",
    }
    valid_evidence_levels = {
        "1_theory_fidelity",
        "2_controlled_retrieval",
        "3_appraisal_quality",
        "4_realistic_tasks",
        "5_human_ecological",
    }

    for claim in claims:
        assert claim["status"] in valid_statuses
        assert claim["evidence_level"] in valid_evidence_levels


def test_claim_validation_matrix_has_unique_ids_and_required_fields() -> None:
    payload = _load_matrix()
    claims = payload["claims"]
    seen_ids: set[str] = set()
    required_fields = {
        "claim_id",
        "claim_text",
        "claim_area",
        "status",
        "evidence_level",
        "allowed_public_wording",
        "current_evidence",
        "not_yet_shown",
        "next_study",
        "evidence_refs",
        "benchmark_refs",
        "protocol_refs",
        "limitations_refs",
    }

    for claim in claims:
        assert required_fields <= set(claim)
        claim_id = claim["claim_id"]
        assert claim_id not in seen_ids
        seen_ids.add(claim_id)
        assert claim["allowed_public_wording"]


def test_claim_validation_matrix_references_existing_repo_paths() -> None:
    root = _repo_root()
    payload = _load_matrix()
    claims = payload["claims"]

    for claim in claims:
        for ref_group in (
            "evidence_refs",
            "benchmark_refs",
            "protocol_refs",
            "limitations_refs",
        ):
            refs = claim[ref_group]
            assert isinstance(refs, list)
            for ref in refs:
                assert (root / ref).exists(), f"Missing reference path in matrix: {ref}"


def test_current_evidence_page_mentions_canonical_matrix_and_allowed_wording() -> None:
    payload = _load_matrix()
    evidence_page = (_repo_root() / "docs" / "research" / "09_current_evidence.md").read_text(
        encoding="utf-8"
    )

    assert "claim_validation_matrix.json" in evidence_page
    assert "## Claim matrix" in evidence_page

    for claim in payload["claims"]:
        assert claim["claim_id"] in evidence_page
        assert claim["allowed_public_wording"] in evidence_page


def test_readme_points_to_canonical_claim_matrix() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "claim_validation_matrix.json" in readme
    assert "current claim-to-evidence matrix" in readme
