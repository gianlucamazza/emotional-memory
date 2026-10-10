# ATTRIBUTION — dailydialog_personas_v1

`dailydialog_personas_v1.json` is **derived from DailyDialog** and is **not**
part of the owner-only Hugging Face dataset listed in `PUBLISHABLE.txt`.

The raw DailyDialog corpus is not redistributed in this repository. Personas
are built by `benchmarks/dailydialog/persona_builder.py` from DailyDialog
train+validation (see `benchmarks/dailydialog/README.md` and
`benchmarks/dailydialog/dataset.py`).

## Source

- **Upstream:** DailyDialog (Li et al. 2017, IJCNLP / COLING)
  <https://aclanthology.org/I17-1099/>
- **This file:** 120 synthetic personas, 396 affect-conditioned queries,
  `seed=0`, `emotion_density_threshold=0.3` (JSON header;
  `benchmarks/preregistration_addendum_k_dailydialog.md`)
- **Query generation:** programmatic from Ekman labels; no LLM
  (`benchmarks/dailydialog/README.md`)

## License

DailyDialog is released under **CC BY-NC-SA 4.0**. The pre-built personas
JSON is derived from that corpus and is included for reproducibility under
the same CC BY-NC-SA 4.0 terms (`benchmarks/dailydialog/README.md`).

Because the texts are third-party-derived, this set **cannot** be released
under the owner Hub licence and must not appear in `PUBLISHABLE.txt`.

## Citation

Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). *DailyDialog:
A Manually Labelled Multi-turn Dialogue Dataset.* IJCNLP 2017.
