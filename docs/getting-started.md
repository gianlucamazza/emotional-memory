# Getting Started

Install `emotional_memory`, encode your first memories, and run an
emotionally-weighted retrieval. The install recipes and quickstart below are
sourced from the top-level `README.md` (SSOT) so the PyPI page and this guide
never drift.

<!--
Installation + Quickstart are sourced from README.md between the
`ssot:getting-started-start` / `ssot:getting-started-end` markers. To edit,
change the README; the docs site picks it up on next build.
-->
{%
  include-markdown "../README.md"
  start="<!-- ssot:getting-started-start -->"
  end="<!-- ssot:getting-started-end -->"
  rewrite-relative-urls=true
%}

## Next steps

- [Mental Model](mental_model.md) — a 5-minute walkthrough of the AFT pipeline
- [Module Overview](architecture/module-overview.md) — how the modules compose
- [Tutorials](tutorials/async.md) — async, persistence, LangChain, query routing, custom appraisal schemas
- [API Reference](api/engine.md) — full symbol-level documentation
- [Research](research/index.md) — the theory and current evidence behind each layer
