# arXiv Submission Checklist

Build the submission bundle first:

```bash
make bench-comparative-sbert  # regenerate SBERT comparative results (Table 3 source)
make reproduce-paper          # regenerate paper/tables/*.tex from committed results
make paper                    # rebuild PDF (checks LaTeX + bibliography)
make paper-arxiv              # produces paper/arxiv-submission.tar.gz
tar -tzf paper/arxiv-submission.tar.gz   # verify contents
```

Bundle must contain: `main.tex`, `main.bbl`, `refs.bib`, `figures/*.pdf`,
`tables/*.tex`. No `.aux`, `.log`, or `.pdf` at the root level (arXiv compiles
from source).

---

## Pre-submission checks

### Content

- [ ] Abstract ≤ 1920 chars (arXiv hard limit)
- [ ] Author name and affiliation in `\author{}` / `\affil{}`
- [ ] Contact email visible (or use arXiv author contact field)
- [ ] All figures referenced in text (`\ref{fig:X}`) and present in bundle
- [ ] All tables referenced in text and present in bundle
- [ ] DOI for software artifact correct: `10.5281/zenodo.21870707`
- [ ] PyPI version pinned: `emotional-memory==0.18.1`
- [ ] No `\todo{}` or `\note{}` macros remaining

### Bibliography

- [ ] `main.bbl` generated from latest `refs.bib` (run `make paper` then `make paper-arxiv`)
- [ ] All `\cite{}` keys resolve without warnings in `main.blg`

### arXiv metadata (fill during submission form)

| Field | Value |
|---|---|
| Title | Emotional Memory for LLMs: Affective Field Theory |
| Authors | Gianluca Mazza |
| Affiliation | Independent Researcher |
| Primary category | `cs.LG` once endorsed (see below) |
| Secondary categories | `cs.AI`, `cs.CL` (optional cross-list) |
| MSC class | 68T07, 68T50 (optional) |
| ACM class | I.2.7 (optional) |
| Comments | 20 pages, 4 figures, 3 tables. Software: emotional-memory v0.18.1 |
| License | CC BY 4.0 (recommended) |
| DOI | 10.5281/zenodo.21870707 (Zenodo software record for v0.18.1) |

---

## Category and endorsement

**Blocked (2026-01-21 policy).** Issue #31 stays open until endorsement is
obtained. The submission bundle is ready; the software is already citable via
Zenodo `10.5281/zenodo.21870707`.

As of [21 January 2026](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy),
new submitters in **all categories** (including `cs.LG`) need one of:

1. **Automatic path:** an institutional academic/research email **and** prior
   authorship on an accepted arXiv paper in the same endorsement domain
   ([paper ownership](https://info.arxiv.org/help/authority.html)).
2. **Personal path:** endorsement from an established arXiv author in that
   domain ([who can endorse](https://info.arxiv.org/help/endorsement.html#who-can-endorse)).

An institutional email alone is no longer sufficient. arXiv staff cannot waive
the requirement or personally endorse. Independent researchers without a prior
paper in the domain must use path 2.

Intended category once endorsed: primary `cs.LG`, optional cross-list `cs.AI`
and `cs.CL`. Request endorsement at https://arxiv.org/auth/endorse.

**Not arXiv (optional, no endorsement):** [OSF Preprints](https://osf.io/preprints/)
gives a DOI and Google Scholar indexing. It is not a substitute for arXiv.

---

## Submission steps (arXiv)

1. Go to https://arxiv.org/submit
2. Click **Start new submission**
3. Select primary category `cs.LG` (only after endorsement; see above)
4. Upload `paper/arxiv-submission.tar.gz`
5. Wait for auto-compilation preview — fix any LaTeX errors
6. Fill metadata form (title, authors, abstract, comments, DOI)
7. Agree to license (CC BY 4.0 recommended)
8. Submit → paper enters moderation queue (~1 business day)
9. Once published, update `CITATION.cff` and `README.md` with the arXiv ID

---

## Post-acceptance

- [ ] Update `CITATION.cff`: add `url: https://arxiv.org/abs/XXXX.XXXXX`
- [ ] Update `README.md` badges with arXiv shield
- [ ] Update `paper/main.tex` with arXiv reference (author note or footnote)
- [ ] Announce on GitHub Discussions / social
