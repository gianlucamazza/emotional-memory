# arXiv Submission Checklist — emotional_memory

Use this checklist before submitting to arXiv. Update the ✓/✗ column in place.

---

## 1. Source bundle

| Item                                                                 | Status | Notes                                                                                 |
| -------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `arxiv-submission.tar.gz` exists and is up-to-date                   | ✓      | `make check-arxiv-bundle` enforces freshness; `make paper-arxiv` regenerates          |
| Bundle compiles to PDF without errors (`pdflatex` or `latexmk`)      | ✓      | 20pp, ~540KB (addenda through Y/Z) — only benign hyperref warnings |
| No compilation warnings about missing figures                        | ✓      | Confirmed in last `latexmk` run                                                       |
| All figures in `figures/` are referenced in `main.tex`               | ✓      | `make paper-arxiv` now does selective copy of only referenced figures                 |
| All figures are in acceptable format (PDF, PNG, EPS)                 | ✓      | All 4 figures are PDF                                                                 |
| `refs.bib` is included and all citations resolve                     | ✓      | 44 unique `\cite` keys, 44 entries — all resolve                                      |
| No `\usepackage{minted}` or other packages requiring `-shell-escape` | ✓      | Confirmed: no `minted` in `main.tex`                                                  |

---

## 2. Content

| Item                                                  | Status | Notes                                                                                                                            |
| ----------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Title matches repo/Zenodo metadata                    | ✗      | Check `release.toml` and `CITATION.cff`                                                                                          |
| Authors and affiliations complete                     | ✗      |                                                                                                                                  |
| Abstract ≤ 1920 characters (arXiv limit)              | ✓      | reframed (V+T lead); ≈1890 rendered est. after the X2 both-corpora rewording — re-verify with arXiv's own counter at submit time |
| No placeholders (`XXXX`, `TODO`, `???`) in text       | ✓      | `grep -n "TODO\|XXXX\|???"` in `main.tex` — clean (Acknowledgements TODO resolved 2026-07-02)                                    |
| Acknowledgements section present                      | ✓      | Final text: no external funding, independent work; thanks to OSS maintainers and corpus authors                                  |
| All claims in §Results match committed JSON artifacts | ✓      | `make reproduce-paper-check` passes — zero diff                                                                                  |
| §Limitations is present and complete                  | ✓      | Includes X/X2 third-party FAIL, Y query-affect gate (safe wrapper), Z learned-profile FAIL                           |
| Negative results (LoCoMo Gate 1 FAIL) are disclosed   | ✓      | §Limitations §External-benchmark scope                                                                                           |

---

## 3. Metadata (arXiv submission form)

| Item                                                       | Status | Notes                                |
| ---------------------------------------------------------- | ------ | ------------------------------------ |
| Primary category: `cs.LG` (blocked on endorsement)         | ✗      | 2026-01-21 policy; issue #31 stays open. Cross-list `cs.AI`, `cs.CL` optional |
| Cross-list categories: `cs.AI`, `cs.CL`                    | ✗      | Optional but recommended             |
| MSC classification (if required): not required for cs      | —      |                                      |
| License: `CC BY 4.0` or `CC BY-NC 4.0`                     | ✗      | Must match `LICENSE` file            |
| DOI (Zenodo concept DOI): `10.5281/zenodo.19972258`        | ✓      | In `release.toml`; confirmed correct |
| arXiv ID: update `release.toml: arxiv_id` after submission | ✗      | Required for Zenodo version DOI link |

---

## 4. Anonymization

| Item                                                            | Status | Notes                                               |
| --------------------------------------------------------------- | ------ | --------------------------------------------------- |
| No anonymization required (not submitted to blind-review venue) | ✓      | Author names in paper                               |
| Repository URL in paper is public                               | ✓      | `https://github.com/gianlucamazza/emotional-memory` |

---

## 5. Post-submission

| Item                                                           | Status | Notes                                     |
| -------------------------------------------------------------- | ------ | ----------------------------------------- |
| Update `release.toml: arxiv_id` with assigned arXiv ID         | ✗      |                                           |
| Update `CITATION.cff: identifiers` with arXiv URL              | ✗      |                                           |
| Update `docs/research/claim_validation_matrix.json` references | ✗      | If any claim cites the arXiv paper itself |
| Create a Zenodo version snapshot pointing to the arXiv ID      | ✗      | Zenodo supports arXiv DOI linking         |
| Announce in repository `CHANGELOG.md` under `[Unreleased]`     | ✗      |                                           |

---

## 6. Reproducibility gate (run before submitting)

```bash
make check                   # lint + typecheck + test + bench-fidelity
make reproduce-paper         # regenerates paper/tables/; diff vs committed
git diff --stat              # must be clean before bundle generation
```

If any of the above fail, do not submit until resolved.

---

_Last updated: 2026-09-11 (endorsement policy 2026-01-21 recorded; issue #31
stays open). Prior 2026-08-10: software snapshot **v0.18.0**, version DOI
`10.5281/zenodo.21870707`, concept DOI `10.5281/zenodo.19972258`; bundle/DOI
synced via `make release`). Prior 2026-07-17: software snapshot v0.17.0, version
DOI `10.5281/zenodo.21402228`. Prior 2026-07-07b: abstract boundary claim sharpened
after Addendum X2 ("affect improves retrieval exactly when relevance is itself
affect-conditioned"); addenda range A–X2; earlier software snapshot was v0.15.0.
Prior 2026-07-02: related-work refresh, 44/44 cite keys, Acknowledgements TODO
resolved, primary category cs.LG. Remaining ✗ items (arXiv upload / arxiv_id in
`release.toml`) are blocked on endorsement (arXiv policy 2026-01-21) — issue #31
stays open. The 2026-07-02 note that `cs.LG` needed no endorsement is obsolete._
