# arXiv Submission Checklist

Build the submission bundle first:

```bash
make paper-arxiv          # produces paper/arxiv-submission.tar.gz
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
- [ ] DOI for software artifact correct: `10.5281/zenodo.19695146`
- [ ] PyPI version pinned: `emotional-memory==0.6.3`
- [ ] No `\todo{}` or `\note{}` macros remaining

### Bibliography

- [ ] `main.bbl` generated from latest `refs.bib` (run `make paper` then `make paper-arxiv`)
- [ ] All `\cite{}` keys resolve without warnings in `main.blg`

### arXiv metadata (fill during submission form)

| Field | Value |
|---|---|
| Title | Emotional Memory for LLMs: Affective Field Theory |
| Authors | Gianluca Mazza |
| Affiliation | (your affiliation or leave blank) |
| Primary category | see below |
| Secondary categories | cs.LG, stat.ML (optional) |
| MSC class | 68T07, 68T50 (optional) |
| ACM class | I.2.7 (optional) |
| Comments | 10 pages, 4 figures, 3 tables. Software: emotional-memory v0.6.3 |
| License | CC BY 4.0 (recommended) |
| DOI | 10.5281/zenodo.19695146 (Zenodo concept record for emotional-memory) |

---

## Category selection

### Option A — cs.AI (requires endorsement)

**cs.AI** (Artificial Intelligence) is the most natural fit.  Requires an
endorser who has previously submitted to cs.AI.  Ask on the arXiv mailing
list or find a collaborator.

Steps:
1. Request endorsement at https://arxiv.org/auth/endorse
2. Endorser enters code at https://arxiv.org/auth/endorse?x={code}
3. Submit once endorsed

### Option B — cs.LG (no endorsement required)

**cs.LG** (Machine Learning) does not require endorsement for new submitters.
Lower barrier; still visible to the AI/ML community.  Add cs.AI as a
cross-list category.

### Option C — OSF Preprints (no gate)

https://osf.io/preprints/ — submit without review or endorsement.  Provides
a DOI and is indexed by Google Scholar.  Less prestigious than arXiv but
zero friction.

---

## Submission steps (arXiv)

1. Go to https://arxiv.org/submit
2. Click **Start new submission**
3. Select primary category (cs.LG or cs.AI if endorsed)
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
