"""Guards for the Dependabot auto-merge workflow hardening."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "dependabot-auto-merge.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_dependabot_auto_merge_pins_fetch_metadata_to_sha() -> None:
    text = _workflow_text()
    assert re.search(
        r"dependabot/fetch-metadata@[0-9a-f]{40}",
        text,
    ), "dependabot/fetch-metadata must be SHA-pinned"


def test_dependabot_auto_merge_does_not_trust_github_actor() -> None:
    code = "\n".join(
        line for line in _workflow_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert "github.actor" not in code
    assert "github.event.pull_request.user.login == 'dependabot[bot]'" in code
    assert "github.event.pull_request.head.repo.full_name" in code


def test_dependabot_auto_merge_scopes_write_permissions_to_job() -> None:
    text = _workflow_text()
    # Workflow-level write on pull_request is the high zizmor finding.
    head, _, rest = text.partition("jobs:")
    assert "permissions: {}" in head
    assert "contents: write" not in head
    assert "contents: write" in rest
