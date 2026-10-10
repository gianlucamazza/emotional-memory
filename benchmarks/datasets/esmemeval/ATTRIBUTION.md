# ATTRIBUTION — ES-MemEval / EvoEmo

This folder vendors third-party data. It is **not** part of the owner-only
Hugging Face dataset listed in `../PUBLISHABLE.txt`.

The vendored JSON is excluded from byte-mutating pre-commit hooks so the
sha256 pin stays intact (`.pre-commit-config.yaml`).

## Source

- **Upstream:** slptongji/ES-MemEval
  (<https://github.com/slptongji/ES-MemEval>); Zenodo DOI
  [10.5281/zenodo.18338564](https://doi.org/10.5281/zenodo.18338564)
- **Release snapshot:** ES-MemEval v1.0.0, GitHub tree `6926242`
- **Vendored file (byte-identical; sha256-pinned by the loader):**
  `evo_emo.json`
- **Pin and refusal-on-mismatch:**
  `benchmarks/preregistration_addendum_x2_esmemeval_third_party.md`
- **Loader:** `benchmarks/esmemeval/dataset.py`
  (`EVO_EMO_SHA256`, `N_SEEKERS=18`, `N_SESSIONS=401`, `N_QUESTIONS=1427`,
  `N_IN_FAMILY=1133`)

The released artifact contains 1,427 QA questions while the paper reports
1,209; the hash-pinned artifact is treated as ground truth
(`benchmarks/datasets/README.md`).

## License

**CC-BY-4.0** (per the Zenodo record). Applies to the vendored file.
Redistribution of this file or derivatives must preserve attribution per
the CC-BY-4.0 terms. Recorded in `benchmarks/datasets/README.md`.

## Citation

Chen, T., Lu, J., Shen, Y., & Zhang, L. (2026). *ES-MemEval: Benchmarking
Conversational Agents on Personalized Long-Term Emotional Support.* The Web
Conference (WWW) 2026, doi:10.1145/3774904.3792143; arXiv:2602.01885.
