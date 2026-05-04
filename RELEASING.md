# Release Runbook

Step-by-step procedure for cutting a new release of `emotional_memory`.
The pipeline (`scripts/release.py`) is fully automated — this document
covers the manual prerequisites and recovery paths.

## Prerequisites (one-time setup)

- `.env` with `ZENODO_TOKEN`, `PYPI_TOKEN`, optional `HF_TOKEN`
- `gh` CLI authenticated (`gh auth status`)
- `latexmk` + TeX Live for paper PDF rebuild

## Pre-flight checklist (before each release)

1. **Webhook off** — verify https://zenodo.org/account/settings/github/
   shows `emotional-memory` toggle **OFF**.  Preflight gate G14 catches this
   automatically, but the toggle must be off *before* tag push or a shadow
   deposit will appear under the canonical concept.
2. Bump `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md` to target version
3. `make sync-metadata` — propagates SSOT from `release.toml` to all consumers
4. `make check` green locally (CI billing-blocked; run before every release)
5. Working tree clean, on `main`, up-to-date with `origin/main`
6. `uv run python scripts/preflight.py VERSION` — all gates green

## Run

```bash
make release VERSION=X.Y.Z
```

The orchestrator (`scripts/release.py`) runs 9 phases and persists state to
`.release_state.json` (gitignored).

| Phase | Name | Key action |
|---|---|---|
| 0 | preflight | G1–G14 gates |
| 1 | zenodo_reserve | create draft + prereserve DOI + concept/shadow guards |
| 2 | doi_sync | write DOI to `release.toml`, `make sync-metadata`, rebuild PDF |
| 3 | commit_tag | `git commit` SSOT files + annotated tag |
| 4 | zenodo_publish | upload PDF + source + arXiv tarball, publish deposit |
| 5 | pypi | `uv publish dist/*` |
| 6 | github_release | `gh release create` + assets |
| 7 | hf_space | push demo to Hugging Face Space |
| 8 | swh | trigger Software Heritage save |
| 9 | report | print summary table |

## Recovery

- **Phase fails before Phase 4** — fix root cause, then
  `make release-resume VERSION=X.Y.Z`.  Nothing is published yet; the Zenodo
  draft can be discarded via UI if needed.
- **Phase 4+ fails** — the Zenodo deposit is already published (irreversible).
  Continue with `make release-resume`; partial publish is acceptable.
- **G14 fails** (active webhook detected) — disable the toggle at
  https://zenodo.org/account/settings/github/, then re-run preflight.
- **Phase 1 anti-shadow fails** (duplicate version under concept) — a webhook
  shadow deposit is already published.  Delete it from Zenodo UI
  (record → Edit → Delete, reason: duplicate), confirm the toggle is OFF,
  then delete `.release_state.json` and retry from scratch.

## Post-release

```bash
git push origin main vX.Y.Z   # orchestrator pushes, but verify
```

After arXiv announces the paper:

```bash
# set arxiv_id in release.toml, then:
make sync-metadata
git add release.toml <ssot-files>
git commit -m "docs: add arXiv ID XXXX.XXXXX to release.toml"
git push origin main
```

## Zenodo concept-DOI provenance

The project has three concept umbrellas from early webhook proliferation
(April–May 2026).  Only the canonical one should receive new versions:

| Concept DOI | Status | Records |
|---|---|---|
| `10.5281/zenodo.19635748` | archived — v0.5.1 only | 1 |
| `10.5281/zenodo.19972284` | archived — v0.7.0 only | 1 |
| `10.5281/zenodo.19972258` | **canonical** — v0.6.1 → present | active |

`release.toml` `concept_doi` is pinned to `19972258`.  Never create a new
concept — always use `actions/newversion` against the latest record under the
canonical umbrella.  The Phase 1 guard in `scripts/release.py` enforces this.
