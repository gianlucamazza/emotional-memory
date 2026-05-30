# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-demo install-sqlite install-redis install-sentence-transformers install-langchain install-mem0 install-bench install-llm-test install-viz install-docs install-release install-all lint format test cov typecheck meta-check meta-check-local check check-all check-arxiv-bundle bench-perf bench-fidelity bench bench-appraisal bench-comparative bench-comparative-sbert bench-comparative-sota bench-realistic bench-realistic-hash bench-realistic-v2-sbert bench-realistic-v2-e5 bench-realistic-it-sbert bench-realistic-it-e5 bench-realistic-it-me5 bench-realistic-es-sbert bench-realistic-es-me5 bench-realistic-fr-me5 bench-ablation bench-ablation-sbert bench-ablation-hash bench-hi3-sbert bench-hi3-e5 bench-hi3-analyze bench-appraisal-confound bench-appraisal-confound-hash bench-addendum-g bench-addendum-g-hash bench-appraisal-diagnostics bench-appraisal-diagnostics-dry bench-dailydialog bench-dailydialog-dry build-dailydialog-personas build-dailydialog-personas-dry bench-locomo bench-locomo-routing bench-locomo-dry bench-locomo-pareto bench-locomo-pareto-dry human-eval-packets human-eval-summary reproduce-paper paper test-llm llm-config llm-config-strict demo-check demo-run docs-images research-figures paper-figures figures docs docs-serve dist bump publish publish-pypi-manual verify-pypi-release sync-release-metadata zenodo-draft zenodo-publish release-check release-space clean help

install:
	uv pip install -e ".[dev]"

install-demo:
	uv pip install -e ".[dev,demo,viz,sentence-transformers,llm-test]"

install-viz:
	uv pip install -e ".[dev,viz]"

install-docs:
	uv pip install -e ".[docs]"

install-release:
	uv pip install -e ".[dev,release,llm-test,bench]"

install-bench:
	uv pip install -e ".[dev,bench]"

install-llm-test:
	uv pip install -e ".[dev,llm-test]"

install-dotenv:
	uv pip install -e ".[dev,dotenv]"

install-sqlite:
	uv pip install -e ".[dev,sqlite]"

install-redis:
	uv pip install -e ".[dev,redis]"

install-sentence-transformers:
	uv pip install -e ".[dev,sentence-transformers]"

install-langchain:
	uv pip install -e ".[dev,langchain]"

install-mem0:
	uv pip install -e ".[dev,mem0]"

install-all:
	uv pip install -e ".[dev,demo,viz,docs,bench,llm-test,dotenv,sqlite,sentence-transformers,langchain,release]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run python -m pytest

test-all:
	uv run python -m pytest --override-ini="addopts=-q --tb=short -m 'not llm and not appraisal_quality'"

cov:
	uv run python -m pytest --cov --cov-report=term-missing

typecheck:
	uv run mypy src/emotional_memory/

sync-metadata:
	uv run python scripts/sync_release_metadata.py --from-toml

sync-metadata-dry:
	uv run python scripts/sync_release_metadata.py --from-toml --dry-run

bump:
	@test -n "$(VERSION)" || (echo "Usage: make bump VERSION=X.Y.Z [DATE=YYYY-MM-DD]"; exit 1)
	uv run python scripts/bump_version.py $(VERSION) $(if $(DATE),--date $(DATE),)

meta-check:
	uv run python scripts/check_release_metadata.py
	uv run python tools/audit_claim_refs.py

meta-check-local:
	uv run python scripts/check_release_metadata.py --require-local-doi

check: lint typecheck meta-check test bench-fidelity

check-all: check install-docs
	uv run python -m mkdocs build --strict --quiet
	uv run python scripts/preflight.py --fast || true  # G6 always fails on feature branches; hard gate is in make release
	$(MAKE) reproduce-paper-check
	$(MAKE) check-arxiv-bundle

check-arxiv-bundle:
	@diff <(tar -xzf paper/arxiv-submission.tar.gz -O ./main.tex) paper/main.tex > /dev/null || \
		(echo "ERROR: paper/arxiv-submission.tar.gz is stale — run 'make paper-arxiv' and commit the bundle"; exit 1)
	@echo "OK: arxiv bundle main.tex matches paper/main.tex"

bench-fidelity:
	uv run python -m pytest benchmarks/fidelity/ -v -m fidelity

bench-perf:
	uv run python -m pytest benchmarks/perf/ --benchmark-only --benchmark-sort=mean

bench: bench-fidelity bench-perf

bench-comparative:
	uv run python -m benchmarks.comparative.runner

bench-comparative-sbert:
	uv run python -m benchmarks.comparative.runner --embedder sbert --out benchmarks/comparative/results.sbert.csv

bench-comparative-sota: llm-config-strict
	uv run python -m benchmarks.comparative.runner \
		--systems aft,naive_cosine,recency,mem0,langmem \
		--embedder sbert \
		--out benchmarks/comparative/results.sota.sbert.csv

bench-realistic:
	uv run python -m benchmarks.realistic.runner --embedder sbert-bge

bench-realistic-hash:
	uv run python -m benchmarks.realistic.runner --embedder hash

bench-realistic-v2-sbert:
	uv run python -m benchmarks.realistic.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v2.json \
		--out-json benchmarks/realistic/results.v2.sbert.json \
		--out-md benchmarks/realistic/results.v2.sbert.md \
		--out-protocol benchmarks/realistic/results.protocol.v2.sbert.json

bench-realistic-v2-e5:
	uv run python -m benchmarks.realistic.runner --embedder e5-small-v2 \
		--dataset benchmarks/datasets/realistic_recall_v2.json \
		--out-json benchmarks/realistic/results.v2.e5.json \
		--out-md benchmarks/realistic/results.v2.e5.md \
		--out-protocol benchmarks/realistic/results.protocol.v2.e5.json

bench-realistic-it-sbert:
	uv run python -m benchmarks.realistic.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v2_it.json \
		--out-json benchmarks/realistic/results.v2_it.sbert.json \
		--out-md benchmarks/realistic/results.v2_it.sbert.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_it.sbert.json

bench-realistic-it-e5:
	uv run python -m benchmarks.realistic.runner --embedder e5-small-v2 \
		--dataset benchmarks/datasets/realistic_recall_v2_it.json \
		--out-json benchmarks/realistic/results.v2_it.e5.json \
		--out-md benchmarks/realistic/results.v2_it.e5.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_it.e5.json

bench-realistic-it-me5:
	uv run python -m benchmarks.realistic.runner --embedder multilingual-e5-small \
		--dataset benchmarks/datasets/realistic_recall_v2_it.json \
		--out-json benchmarks/realistic/results.v2_it.me5.json \
		--out-md benchmarks/realistic/results.v2_it.me5.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_it.me5.json

bench-realistic-es-sbert:
	uv run python -m benchmarks.realistic.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v2_es.json \
		--out-json benchmarks/realistic/results.v2_es.sbert.json \
		--out-md benchmarks/realistic/results.v2_es.sbert.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_es.sbert.json

bench-realistic-es-me5:
	uv run python -m benchmarks.realistic.runner --embedder multilingual-e5-small \
		--dataset benchmarks/datasets/realistic_recall_v2_es.json \
		--out-json benchmarks/realistic/results.v2_es.me5.json \
		--out-md benchmarks/realistic/results.v2_es.me5.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_es.me5.json

bench-realistic-fr-me5:
	uv run python -m benchmarks.realistic.runner --embedder multilingual-e5-small \
		--dataset benchmarks/datasets/realistic_recall_v2_fr.json \
		--out-json benchmarks/realistic/results.v2_fr.me5.json \
		--out-md benchmarks/realistic/results.v2_fr.me5.md \
		--out-protocol benchmarks/realistic/results.protocol.v2_fr.me5.json

bench-ablation:
	uv run python -m benchmarks.ablation.runner --embedder hash

bench-ablation-sbert:
	uv run python -m benchmarks.ablation.runner --embedder sbert-bge \
		--out-json benchmarks/ablation/results.sbert.json \
		--out-md benchmarks/ablation/results.sbert.md \
		--out-protocol benchmarks/ablation/results.sbert.protocol.json

bench-ablation-hash:
	uv run python -m benchmarks.ablation.runner --embedder hash

bench-appraisal-confound:
	uv run python -m benchmarks.appraisal_confound.runner --embedder sbert-bge

bench-appraisal-confound-hash:
	uv run python -m benchmarks.appraisal_confound.runner --embedder hash

# Addendum G — Hg1: dual-path LLM appraisal on affect-free dataset (requires API key)
bench-addendum-g: llm-config-strict
	uv run python -m benchmarks.appraisal_confound.runner_hg1 --embedder sbert-bge

bench-addendum-g-hash: llm-config-strict
	uv run python -m benchmarks.appraisal_confound.runner_hg1 --embedder hash

# WP-1a — Appraisal diagnostics: residuals of LLM appraisal vs oracle affect (requires API key)
bench-appraisal-diagnostics: llm-config-strict
	uv run python -m benchmarks.appraisal_diagnostics.runner --seed 42

# Smoke test with a fixed appraisal vector — no LLM key required
bench-appraisal-diagnostics-dry:
	uv run python -m benchmarks.appraisal_diagnostics.runner --dry-run --verbose

# Hi3 confirmatory ablation @ N=500 (seed=1 frozen — runs on realistic_recall_v3)
bench-hi3-sbert:
	uv run python -m benchmarks.ablation.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v3.json \
		--seed 1 \
		--per-query-records \
		--out-json benchmarks/ablation/results.v3.sbert.json \
		--out-md benchmarks/ablation/results.v3.sbert.md \
		--out-protocol benchmarks/ablation/results.v3.sbert.protocol.json

bench-hi3-e5:
	uv run python -m benchmarks.ablation.runner --embedder e5-small-v2 \
		--dataset benchmarks/datasets/realistic_recall_v3.json \
		--seed 1 \
		--per-query-records \
		--out-json benchmarks/ablation/results.v3.e5.json \
		--out-md benchmarks/ablation/results.v3.e5.md \
		--out-protocol benchmarks/ablation/results.v3.e5.protocol.json

# Hi3 Holm-family confirmatory analysis (requires bench-hi3-sbert + bench-hi3-e5 first)
bench-hi3-analyze:
	uv run python -m benchmarks.ablation.runner_hi3 \
		--results-sbert benchmarks/ablation/results.v3.sbert.json \
		--results-e5 benchmarks/ablation/results.v3.e5.json \
		--out-json benchmarks/ablation/results.hi3.json \
		--out-md benchmarks/ablation/results.hi3.md \
		--out-protocol benchmarks/ablation/results.hi3.protocol.json

# S3 ablation @ N=200 (pre-registered Study S3, powered — runs on realistic_recall_v2)
bench-s3-sbert:
	uv run python -m benchmarks.ablation.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v2.json \
		--out-json benchmarks/ablation/results.v2.sbert.json \
		--out-md benchmarks/ablation/results.v2.sbert.md \
		--out-protocol benchmarks/ablation/results.v2.sbert.protocol.json

bench-s3-e5:
	uv run python -m benchmarks.ablation.runner --embedder e5-small-v2 \
		--dataset benchmarks/datasets/realistic_recall_v2.json \
		--out-json benchmarks/ablation/results.v2.e5.json \
		--out-md benchmarks/ablation/results.v2.e5.md \
		--out-protocol benchmarks/ablation/results.v2.e5.protocol.json

# Hd2 generalization (Addendum D, v2 EN + cross-language IT slice)
bench-hd2-sbert:
	uv run python -m benchmarks.appraisal_confound.runner --embedder sbert-bge \
		--dataset benchmarks/datasets/realistic_recall_v2.json \
		--out-json benchmarks/appraisal_confound/results.hd2.sbert.json \
		--out-md benchmarks/appraisal_confound/results.hd2.sbert.md \
		--out-protocol benchmarks/appraisal_confound/results.hd2.sbert.protocol.json

bench-hd2-it-me5:
	uv run python -m benchmarks.appraisal_confound.runner --embedder multilingual-e5-small \
		--dataset benchmarks/datasets/realistic_recall_v2_it.json \
		--out-json benchmarks/appraisal_confound/results.hd2_it.me5.json \
		--out-md benchmarks/appraisal_confound/results.hd2_it.me5.md \
		--out-protocol benchmarks/appraisal_confound/results.hd2_it.me5.protocol.json

bench-dailydialog:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.dailydialog.runner

bench-dailydialog-dry:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.dailydialog.runner --dry-run

build-dailydialog-personas:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.dailydialog.persona_builder \
	    --n 120 --seed 0

build-dailydialog-personas-dry:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.dailydialog.persona_builder \
	    --n 5 --seed 0 --dry-run

bench-locomo: llm-config-strict
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.runner

bench-locomo-routing: llm-config-strict
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.routing_runner

bench-locomo-dry:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.runner --limit-conversations 2 --limit-qa 5 --no-judge

bench-locomo-pareto: llm-config-strict
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.pareto_runner

bench-locomo-pareto-dry:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.pareto_runner \
	    --dry-run --limit-configs 2 --no-judge

human-eval-packets:
	uv run python -m benchmarks.human_eval.pipeline packets

human-eval-summary:
	uv run python -m benchmarks.human_eval.pipeline summary

reproduce-paper:
	uv run python scripts/reproduce_paper.py

reproduce-paper-check:
	uv run python scripts/reproduce_paper.py
	git diff --exit-code paper/tables/ || (echo "ERROR: paper/tables/ is stale — run 'make reproduce-paper' and commit"; exit 1)

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

paper-arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
	mkdir -p paper/arxiv-build/figures paper/arxiv-build/tables
	cp paper/main.tex paper/arxiv-build/
	cp paper/main.bbl paper/arxiv-build/
	cp paper/refs.bib paper/arxiv-build/
	cp paper/figures/*.pdf paper/arxiv-build/figures/
	cp paper/tables/*.tex paper/arxiv-build/tables/
	tar -czf paper/arxiv-submission.tar.gz -C paper/arxiv-build .
	sha256sum paper/arxiv-submission.tar.gz | awk '{print $$1}' > paper/arxiv-submission.sha256
	rm -rf paper/arxiv-build
	@echo "arXiv bundle ready: paper/arxiv-submission.tar.gz"
	@tar -tzf paper/arxiv-submission.tar.gz

bench-appraisal: llm-config-strict
	uv run python -m pytest benchmarks/appraisal_quality/ -v -m appraisal_quality

test-llm: llm-config-strict
	uv run python -m pytest tests/test_llm_integration.py -v -m llm

llm-config:
	uv run python scripts/check_llm_config.py

llm-config-strict:
	uv run python scripts/check_llm_config.py --strict --require-key

demo-check:
	uv run python -m pytest tests/test_demo_ui_config.py -q
	uv run python -m pytest tests/test_demo_app.py -q

demo-run:
	uv run python demo/app.py

# Didactic figures (docs site only, synthetic input)
docs-images:
	uv run python scripts/generate_docs_images.py

# Evidence figures from committed benchmark JSONs (PNG+PDF → docs/images/research/)
research-figures:
	uv run python scripts/generate_research_figures.py \
	  --png-dir docs/images/research \
	  --pdf-dir docs/images/research

# Schematic/synthetic figures used inside paper/main.tex (→ paper/figures/)
paper-figures:
	uv run python scripts/generate_paper_figures.py

figures: docs-images research-figures paper-figures

docs: install-docs
	uv run python -m mkdocs build --strict

docs-serve: install-docs
	uv run python -m mkdocs serve

dist:
	uv build

preflight:
	uv run python scripts/preflight.py

preflight-fast:
	uv run python scripts/preflight.py --fast

publish: preflight dist
	uv publish

publish-pypi-manual: dist
	@test -n "$$PYPI_TOKEN" || (echo "PYPI_TOKEN not set"; exit 1)
	uv run python -m twine upload dist/* -u __token__ -p "$$PYPI_TOKEN"

verify-pypi-release:
	@test -n "$(VERSION)" || (echo "Usage: make verify-pypi-release VERSION=0.6.3"; exit 1)
	uv run python scripts/verify_pypi_release.py "$(VERSION)"

sync-release-metadata:
	uv run python scripts/sync_release_metadata.py

zenodo-draft:
	@test -n "$$ZENODO_TOKEN" || (echo "ZENODO_TOKEN not set"; exit 1)
	uv run python scripts/zenodo_deposit.py --draft-only

zenodo-reserve:
	@test -n "$$ZENODO_TOKEN" || (echo "ZENODO_TOKEN not set"; exit 1)
	uv run python scripts/zenodo_deposit.py --reserve-only

zenodo-upload-publish:
	@test -n "$$ZENODO_TOKEN" || (echo "ZENODO_TOKEN not set"; exit 1)
	uv run python scripts/zenodo_deposit.py --upload-from-state

zenodo-publish:
	@test -n "$$ZENODO_TOKEN" || (echo "ZENODO_TOKEN not set"; exit 1)
	@test -n "$(DEPOSIT_ID)" || (echo "Usage: make zenodo-publish DEPOSIT_ID=123"; exit 1)
	uv run python scripts/zenodo_deposit.py --publish-id "$(DEPOSIT_ID)"

release-swh:
	uv run python scripts/swh_client.py

## Hybrid release: PyPI + GitHub release come from the on-tag workflow (Trusted
## Publishing OIDC), so the local pipeline skips them by default and handles
## reserve-DOI + Zenodo + HF + SWH. Override to run the full legacy pipeline:
##   RELEASE_FLAGS="" make release VERSION=0.9.0   # PyPI via uv publish + token
## Resume a failed run: make release-resume VERSION=0.9.0
RELEASE_FLAGS ?= --skip-pypi --skip-github-release

## Full automated release (API-driven, no GitHub webhook needed)
## Usage: make release VERSION=0.9.0
release:
	@test -n "$(VERSION)" || (echo "Usage: make release VERSION=0.9.0"; exit 1)
	@export $$(grep -v '^#' .env | xargs) 2>/dev/null; \
		uv run python scripts/release.py "$(VERSION)" $(RELEASE_FLAGS)

release-resume:
	@test -n "$(VERSION)" || (echo "Usage: make release-resume VERSION=0.9.0"; exit 1)
	@export $$(grep -v '^#' .env | xargs) 2>/dev/null; \
		uv run python scripts/release.py "$(VERSION)" --resume $(RELEASE_FLAGS)

release-sandbox:
	@test -n "$(VERSION)" || (echo "Usage: make release-sandbox VERSION=0.9.0"; exit 1)
	@export $$(grep -v '^#' .env | xargs) 2>/dev/null; \
		uv run python scripts/release.py "$(VERSION)" --sandbox $(RELEASE_FLAGS)

release-check:
	@test -n "$(VERSION)" || (echo "Usage: make release-check VERSION=0.6.3"; exit 1)
	$(MAKE) check
	$(MAKE) test-llm
	$(MAKE) bench-appraisal
	uv run python scripts/preflight.py "$(VERSION)"

release-space:
	@git config --get remote.space.url >/dev/null || (echo "remote 'space' not configured"; exit 1)
	@sha=$$(git subtree split --prefix=demo HEAD); \
		echo "Pushing demo subtree $$sha to space/main"; \
		git push space $$sha:main --force

clean:
	rm -rf dist/ build/ site/ htmlcov/ .coverage coverage.xml benchmark-results.json
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "emotional-memory — available make targets"
	@echo ""
	@echo "Setup:"
	@echo "  install                    Install package with dev dependencies"
	@echo "  install-sqlite             + sqlite-vec (SQLiteStore)"
	@echo "  install-redis              + redis (RedisAffectiveStateStore)"
	@echo "  install-sentence-transformers  + sentence-transformers (real embeddings)"
	@echo "  install-langchain              + langchain-core (LangChain adapter)"
	@echo "  install-viz                + matplotlib (visualization)"
	@echo "  install-demo               + Gradio demo runtime (local canonical demo setup)"
	@echo "  install-bench              + pytest-benchmark (performance benchmarks)"
	@echo "  install-llm-test           + httpx (real-LLM tests)"
	@echo "  install-docs               + mkdocs (documentation)"
	@echo "  install-release            Maintainer release toolchain + release gates"
	@echo "  install-all                All extras"
	@echo ""
	@echo "Quality:"
	@echo "  check                      lint + typecheck + test (run before commit)"
	@echo "  lint                       ruff check"
	@echo "  format                     ruff format (in-place)"
	@echo "  typecheck                  mypy strict"
	@echo "  meta-check                 release metadata consistency"
	@echo "  meta-check-local           release metadata + local .zenodo_doi consistency"
	@echo "  test                       pytest (unit + integration, excludes slow)"
	@echo "  test-all                   pytest (all tests including slow)"
	@echo "  cov                        pytest with branch coverage"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench-fidelity             127 psychological invariant tests"
	@echo "  bench-perf                 latency / throughput benchmarks"
	@echo "  bench                      fidelity + performance"
	@echo "  bench-appraisal            LLM appraisal quality (requires API key)"
	@echo "  bench-comparative          Cross-system comparison (hash embedder, quick)"
	@echo "  bench-comparative-sbert    Cross-system comparison (SBERT embedder, paper-canonical)"
	@echo "  bench-comparative-sota     Cross-system comparison incl. Mem0+LangMem (requires API key + install-mem0 install-langmem)"
	@echo "  bench-appraisal-confound   Appraisal confound study (SBERT, no LLM key)"
	@echo "  bench-appraisal-confound-hash  Appraisal confound study (hash embedder)"
	@echo "  bench-appraisal-diagnostics  WP-1a — LLM appraisal residuals vs oracle (requires API key)"
	@echo "  bench-appraisal-diagnostics-dry  WP-1a smoke test (fixed vector, no key)"
	@echo "  bench-addendum-g           Add. G — Hg1: LLM appraisal affect-free (requires API key)"
	@echo "  bench-addendum-g-hash      Add. G fast smoke test with hash embedder"
	@echo "  bench-realistic            Replayable multi-session benchmark with persisted state"
	@echo "  bench-dailydialog          DailyDialog affect-conditioned retrieval (Hk1, no API key)"
	@echo "  bench-dailydialog-dry      DailyDialog dry run (5 personas)"
	@echo "  build-dailydialog-personas Build synthetic-persona JSON (requires: pip install datasets)"
	@echo "  bench-locomo               LoCoMo benchmark (requires EMOTIONAL_MEMORY_LLM_API_KEY)"
	@echo "  bench-locomo-dry           LoCoMo dry run: 2 conversations, 5 QA each, no judge"
	@echo "  bench-locomo-pareto        Add. J Pareto sweep (10 weight configs × 200 QA, requires API key)"
	@echo "  bench-locomo-pareto-dry    Add. J Pareto dry run: 2 configs, 4 QA/cat, no judge"
	@echo "  human-eval-packets         Build human-eval packet + ratings template from replay results"
	@echo "  human-eval-summary         Summarize filled human-eval ratings"
	@echo "  reproduce-paper            Regenerate all paper tables"
	@echo ""
	@echo "LLM tests:"
	@echo "  test-llm                   Integration tests (requires EMOTIONAL_MEMORY_LLM_API_KEY)"
	@echo "  llm-config                 Print resolved LLM config without secrets"
	@echo "  llm-config-strict          Fail fast on missing/unsupported LLM config"
	@echo "  demo-check                 Demo wiring + runtime regression tests"
	@echo "  demo-run                   Launch demo/app.py using current exported env"
	@echo ""
	@echo "Docs:"
	@echo "  docs                       Build static site"
	@echo "  docs-serve                 Live-reload local server"
	@echo "  docs-images                Regenerate docs/images/ PNGs"
	@echo "  research-figures           Regenerate evidence figures from benchmark JSON"
	@echo "  figures                    Regenerate docs + research figures"
	@echo ""
	@echo "Release (automated, API-driven):"
	@echo "  release VERSION=x.y.z        Full release: preflight→DOI reserve→sync→commit→tag→Zenodo→PyPI→GH→HF→SWH"
	@echo "  release-resume VERSION=x.y.z Resume from last successful phase"
	@echo "  release-sandbox VERSION=x.y.z Test against Zenodo sandbox"
	@echo "  release-check VERSION=x.y.z  Pre-release gate (tests + LLM checks)"
	@echo "  zenodo-reserve               Phase 1: prereserve Zenodo DOI (before tagging)"
	@echo "  zenodo-upload-publish        Phase 2: upload files + publish reserved draft"
	@echo "  release-swh                  Trigger Software Heritage save"
	@echo "  release-space                Push demo subtree to Hugging Face Space"
	@echo "  publish-pypi-manual          Manual PyPI upload via twine + PYPI_TOKEN"
	@echo "  verify-pypi-release VERSION=x.y.z  Poll PyPI until the release is visible"
	@echo "  sync-release-metadata        Sync all files from release.toml SSOT"
	@echo "  zenodo-draft                 Create Zenodo draft (legacy, fresh deposit)"
	@echo "  zenodo-publish DEPOSIT_ID=N  Publish an existing Zenodo draft"
	@echo "  dist                         Build wheel + sdist"
	@echo "  publish                      Build and publish to PyPI"
	@echo "  clean                        Remove build artefacts"
