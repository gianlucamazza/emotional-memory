# ATTRIBUTION — emobank_v1

`emobank_v1.json` is a third-party human-annotated VAD subset. It is **not**
part of the owner-only Hugging Face dataset listed in `PUBLISHABLE.txt`.

## Source

- **Upstream:** EmoBank — JULIELab/EmoBank
  (<https://github.com/JULIELab/EmoBank>); texts from the Manually Annotated
  Sub-Corpus (MASC)
- **This file:** 300 rows sampled (`seed=42`, filtered to ≥4 words) from
  EmoBank, as recorded in the JSON `source` / `license` fields and in
  `benchmarks/datasets/README.md`
- **Used by:** Addendum S (`benchmarks/human_gold_appraisal/`), Addendum V,
  Addendum W

## License

**CC-BY-SA 4.0** (share-alike). Applies to `emobank_v1.json` only; it does
**not** change the MIT license of the project's code. Redistribution of this
file or derivatives must preserve attribution and the CC-BY-SA 4.0 terms.

## Citation

Buechel, S., & Hahn, U. (2017). *EmoBank: Studying the Impact of Annotation
Perspective and Representation Format on Dimensional Emotion Analysis.*
EACL 2017, 578–585.
