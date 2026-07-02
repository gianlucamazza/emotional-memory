# Security Policy

## Supported versions

| Version         | Supported               |
| --------------- | ----------------------- |
| 0.14.x (latest) | ✓ — full support        |
| 0.13.x          | ✓ — security fixes only |
| < 0.13          | ✗                       |

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

## Scope

This policy covers the `emotional-memory` library code published on PyPI.
It does **not** cover:

- Development dependencies (`dev`, `bench`, `docs` extras)
- Third-party optional dependencies (`sentence-transformers`, `sqlite-vec`,
  `langchain-core`, etc.) — report those to the respective upstream projects
- Demonstration code in `examples/` or `demo/`

## Known advisories (optional / dev dependencies)

These affect **optional or development dependencies only** — the published
runtime wheel does not import them and is unaffected. They have no upstream patch
at time of writing; we monitor PyPI and bump `uv.lock` as soon as one ships.

| Dependency           | Advisory       | Severity | Exposure                                                                   |
| -------------------- | -------------- | -------- | -------------------------------------------------------------------------- |
| `chromadb` (≤ 1.5.9) | CVE-2026-45829 | Critical | Optional `chroma` extra + dev/test installs only; not in the runtime wheel |

`torch` CVE-2025-3000 is **resolved**: a patched `torch` 2.12.1 has shipped and
`uv.lock` is pinned to it. A `uv.lock` refresh (2026-06-27) cleared every other
advisory by pulling patched transitive versions of optional/dev dependencies —
`cryptography` 49.0.0, `langsmith` 0.9.3, `starlette` 1.3.1, `pydantic-settings`
2.14.2, and `gradio` 6.19.0 (PYSEC-2026-211). A full-lock `pip-audit`
(`uv export --all-extras`) now reports **only** `chromadb` ≤ 1.5.9, which remains
unpatched on PyPI; we monitor and will bump `uv.lock` as soon as a fixed release
ships.
