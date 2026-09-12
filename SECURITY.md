# Security Policy

## Supported versions

| Version         | Supported               |
| --------------- | ----------------------- |
| 0.16.x (latest) | ✓ — full support        |
| 0.15.x          | ✓ — security fixes only |
| < 0.15          | ✗                       |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **info@gianlucamazza.it** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if available)
- Affected versions

You will receive an acknowledgement within **72 hours**. A fix or mitigation
will be targeted within **30 days** of confirmed impact.

Once a fix is released, the disclosure will be coordinated with the reporter
before any public announcement.

## LLM integration threat model

User-supplied `content`, `query`, and `metadata` values are embedded in appraisal
and query-classifier prompts without sanitization. Treat all inputs as untrusted:
bound payload size with `EmotionalMemoryConfig.max_content_length` when deploying
against external users, and never execute model output as code.

## Scope

This policy covers the `emotional-memory` library code published on PyPI.
It does **not** cover:

- Development dependencies (`dev`, `bench`, `docs` extras)
- Third-party optional dependencies (`sentence-transformers`, `sqlite-vec`,
  `langchain-core`, etc.) — report those to the respective upstream projects
- Demonstration code in `examples/` or `demo/`

## Resolved optional-dependency advisories

These affected **optional or development dependencies only** — the published
runtime wheel does not import them and was never exposed.

The optional `[chroma]` extra installs the HTTP-only `chromadb-client`, not the
vulnerable embedded `chromadb` server package. Since 2026-09-12, `ChromaStore`
requires `host=` and rejects local ephemeral or persistent mode. Operate the
remote server outside this library, require authentication, and use only a
deployment verified not to be affected by the listed advisories. This removes CVE-2026-45830,
CVE-2026-45831, and CVE-2026-45833 from this project's dependency graph while
upstream has no fixed `chromadb` release.

`torch` CVE-2025-3000 is **resolved**: a patched `torch` 2.12.1 has shipped and
`uv.lock` is pinned to it. A `uv.lock` refresh (2026-06-27) cleared every other
advisory by pulling patched transitive versions of optional/dev dependencies —
`cryptography` 49.0.0, `langsmith` 0.9.3, `starlette` 1.3.1, `pydantic-settings`
2.14.2, and `gradio` 6.19.0 (PYSEC-2026-211).
