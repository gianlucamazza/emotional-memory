# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.9.x (latest minor) | ✓ — full support |
| 0.8.x   | ✓ — security fixes only |
| < 0.8   | ✗         |

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
