# Production Readiness

`emotional-memory` is production-hardened for teams that need supply-chain assurances:

| Signal | Status |
|---|---|
| **PyPI releases** | Trusted Publishing (OIDC, no long-lived tokens) |
| **SLSA provenance** | Level 3 — build-provenance attestation on every release |
| **SBOM** | CycloneDX JSON generated and attested per release (`dist/sbom.cdx.json`) |
| **PEP 740 attestations** | Signed attestations verifiable via `gh attestation verify` |
| **SAST** | CodeQL workflow on every push/PR to `main` |
| **Workflow security** | All third-party GitHub Actions SHA-pinned; zizmor static analysis in CI |
| **Dependency audit** | pip-audit in CI on every push; no known CVEs |
| **Coverage** | ≥80% branch coverage enforced; informational target 90% |
| **Type safety** | mypy strict + basedpyright (secondary) on every PR |
| **Conventional commits** | PR title enforced (amannn/action-semantic-pull-request) |

```bash
# Verify provenance of a released wheel locally (substitute the version you installed):
gh attestation verify emotional_memory-<version>-py3-none-any.whl \
  --repo gianlucamazza/emotional-memory
```

## See also

- [Getting Started](getting-started.md) — install the package
- [Contributing](contributing/ssot-policy.md) — release and metadata discipline
