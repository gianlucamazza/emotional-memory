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

## Run — hybrid flow

PyPI and the GitHub release are published by the **on-tag GitHub Actions workflow**
(`.github/workflows/release.yml`) via Trusted Publishing (OIDC, no long-lived
token). `make release` therefore handles only what must happen *before the tag*
or that the workflow cannot do: Zenodo DOI reservation + deposit, the paper PDF
rebuild, HF Space, and Software Heritage.

```bash
# 1) Local: reserve DOI, sync metadata, rebuild PDF, commit + tag, deposit Zenodo, HF, SWH.
#    (Skips PyPI + GitHub release by default — see RELEASE_FLAGS in the Makefile.)
make release VERSION=X.Y.Z

# 2) Push: the tag triggers PyPI (OIDC) + GitHub release.
git push origin main vX.Y.Z
```

`scripts/release.py` runs these phases (state persisted to `.release_state.json`,
gitignored; each phase idempotent, `--resume` skips completed ones):

| Phase | Name | Key action | Hybrid default |
|---|---|---|---|
| 0 | preflight | G1–G14 gates | run |
| 1 | zenodo_reserve | create draft + prereserve DOI + concept/shadow guards | run |
| 2 | doi_sync | write DOI to `release.toml`, `make sync-metadata`, rebuild PDF | run |
| 3 | commit_tag | `git commit` SSOT files + annotated tag | run |
| 4 | zenodo_publish | upload PDF + source + arXiv tarball, publish deposit | run |
| 5 | pypi | `uv publish dist/*` | **skipped** (on-tag OIDC) |
| 6 | github_release | `gh release create` + assets | **skipped** (on-tag) |
| 7 | hf_space | push demo to Hugging Face Space | run |
| 8 | swh | trigger Software Heritage save | run |
| 9 | report | print summary table | run |

Phases 5–6 are skipped via `RELEASE_FLAGS ?= --skip-pypi --skip-github-release`
(Makefile). To run the **full** legacy pipeline (PyPI via `uv publish` + token),
override: `RELEASE_FLAGS="" make release VERSION=X.Y.Z`.

> **Do not** ship a release with just `make bump` + `git push` of a tag: that
> skips the Zenodo DOI reservation, so the tag would carry the *previous*
> release's `version_doi`. The on-tag workflow's **DOI-freshness gate**
> (`scripts/check_doi_freshness.py`) blocks exactly this — it fails if the
> tag's `release.toml` `version_doi` equals the previous tag's. v0.11.2 and
> v0.11.3 slipped through before this gate existed.

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
