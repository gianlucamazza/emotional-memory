# arXiv Submission Checklist — emotional_memory

Use this checklist before submitting to arXiv. Update the ✓/✗ column in place.

---

## 1. Source bundle

| Item | Status | Notes |
|---|---|---|
| `arxiv-submission.tar.gz` exists and is up-to-date | ✗ | Regenerate with `make arxiv-bundle` or equivalent |
| Bundle compiles to PDF without errors (`pdflatex` or `latexmk`) | ✗ | Check `.log` for errors |
| No compilation warnings about missing figures | ✗ | |
| All figures in `figures/` are referenced in `main.tex` | ✗ | |
| All figures are in acceptable format (PDF, PNG, EPS) | ✗ | |
| `refs.bib` is included and all citations resolve | ✗ | Run `biber` / `bibtex`; no `?` in references |
| No `\usepackage{minted}` or other packages requiring `-shell-escape` | ✗ | arXiv does not support `-shell-escape` |

---

## 2. Content

| Item | Status | Notes |
|---|---|---|
| Title matches repo/Zenodo metadata | ✗ | Check `release.toml` and `CITATION.cff` |
| Authors and affiliations complete | ✗ | |
| Abstract ≤ 1920 characters (arXiv limit) | ✗ | Count at submit time |
| No placeholders (`XXXX`, `TODO`, `???`) in text | ✗ | `grep -n "TODO\|XXXX\|???"`  in `main.tex` |
| Acknowledgements section present | ✗ | |
| All claims in §Results match committed JSON artifacts | ✗ | Run `make reproduce-paper` and diff |
| §Limitations is present and complete | ✓ | Updated in v0.8.3 (oracle-affect, sign-reversal, dataset scope) |
| Negative results (LoCoMo Gate 1 FAIL) are disclosed | ✓ | §Limitations §External-benchmark scope |

---

## 3. Metadata (arXiv submission form)

| Item | Status | Notes |
|---|---|---|
| Primary category: `cs.AI` | ✗ | Check arXiv taxonomy |
| Cross-list categories: `cs.CL`, `cs.LG` | ✗ | Optional but recommended |
| MSC classification (if required): not required for cs | — | |
| License: `CC BY 4.0` or `CC BY-NC 4.0` | ✗ | Must match `LICENSE` file |
| DOI (Zenodo concept DOI): `10.5281/zenodo.19972258` | ✗ | Paste into "Related DOI" field |
| arXiv ID: update `release.toml: arxiv_id` after submission | ✗ | Required for Zenodo version DOI link |

---

## 4. Anonymization

| Item | Status | Notes |
|---|---|---|
| No anonymization required (not submitted to blind-review venue) | ✓ | Author names in paper |
| Repository URL in paper is public | ✓ | `https://github.com/gianlucamazza/emotional-memory` |

---

## 5. Post-submission

| Item | Status | Notes |
|---|---|---|
| Update `release.toml: arxiv_id` with assigned arXiv ID | ✗ | |
| Update `CITATION.cff: identifiers` with arXiv URL | ✗ | |
| Update `docs/research/claim_validation_matrix.json` references | ✗ | If any claim cites the arXiv paper itself |
| Create a Zenodo version snapshot pointing to the arXiv ID | ✗ | Zenodo supports arXiv DOI linking |
| Announce in repository `CHANGELOG.md` under `[Unreleased]` | ✗ | |

---

## 6. Reproducibility gate (run before submitting)

```bash
make check                   # lint + typecheck + test + bench-fidelity
make reproduce-paper         # regenerates paper/tables/; diff vs committed
git diff --stat              # must be clean before bundle generation
```

If any of the above fail, do not submit until resolved.

---

*Last updated: 2026-05-05. Mark ✗ → ✓ when each item is confirmed.*
