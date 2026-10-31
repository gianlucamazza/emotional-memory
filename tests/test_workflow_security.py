"""Guards for workflow hardening (Dependabot auto-merge + Docs + receipts)."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "dependabot-auto-merge.yml"
DOCS_WORKFLOW = WORKFLOWS / "docs.yml"
RECEIPTS_WORKFLOW = WORKFLOWS / "shared-history-receipts.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _docs_text() -> str:
    return DOCS_WORKFLOW.read_text(encoding="utf-8")


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


def test_docs_workflow_keeps_post_ci_workflow_run() -> None:
    text = _docs_text()
    assert "workflow_run:" in text
    assert 'workflows: ["CI"]' in text
    assert "zizmor: ignore[dangerous-triggers]" in text


def test_docs_workflow_does_not_checkout_workflow_run_head() -> None:
    text = _docs_text()
    assert "ref: ${{ github.event.workflow_run.head_sha }}" not in text
    assert "ref: ${{ github.event.workflow_run.head_branch }}" not in text
    assert "github.event.pull_request.head" not in text


def test_docs_workflow_write_job_does_not_checkout() -> None:
    text = _docs_text()
    head, _, jobs = text.partition("jobs:")
    assert "permissions: {}" in head
    deploy = jobs.split("deploy:", 1)[1]
    assert "actions/checkout@" not in deploy
    assert "pages: write" in deploy
    assert "id-token: write" in deploy
    build = jobs.split("build:", 1)[1].split("deploy:", 1)[0]
    assert "contents: read" in build
    assert "pages: write" not in build


def test_docs_workflow_requires_successful_same_repo_main_push() -> None:
    text = _docs_text()
    assert "github.event.workflow_run.event == 'push'" in text
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in text
    assert "github.event.workflow_run.head_branch == 'main'" in text
    assert "github.event.workflow_run.conclusion == 'success'" in text


def _receipts_text() -> str:
    return RECEIPTS_WORKFLOW.read_text(encoding="utf-8")


def _receipts_code() -> str:
    return "\n".join(
        line for line in _receipts_text().splitlines() if not line.lstrip().startswith("#")
    )


def test_receipts_workflow_does_not_checkout_pr_head() -> None:
    text = _receipts_text()
    code = _receipts_code()
    assert "github.event.pull_request.head" not in text
    assert "github.head_ref" not in code
    assert "persist-credentials: false" in code
    assert code.count("persist-credentials: false") == 1
    assert "name: Checkout trusted event SHA" in code
    after_checkout = code.split("uses: actions/checkout", 1)[1]
    checkout_block = after_checkout.split("\n      - ", 1)[0]
    assert "ref:" not in checkout_block


def test_receipts_workflow_keeps_unprivileged_receipt_permissions() -> None:
    text = _receipts_text()
    head, _, jobs = text.partition("jobs:")
    assert "permissions: {}" in head
    assert "contents: write" not in text
    assert "contents: read" in jobs
    assert "secrets." not in _receipts_code()
    assert "pull_request:" in head
    assert "workflow_dispatch:" in head


def test_receipts_workflow_does_not_checkout_dispatch_input_ref() -> None:
    """Reproduction uses `gh workflow run --ref`, not a user-supplied checkout ref."""
    text = _receipts_text()
    code = _receipts_code()
    assert "inputs.ref" not in text
    assert "ref: ${{ inputs." not in code
    assert code.count("uses: actions/checkout") == 1
    dispatch = code.split("workflow_dispatch:", 1)[1].split("push:", 1)[0]
    assert "inputs:" not in dispatch
