# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-demo install-sqlite install-redis install-sentence-transformers install-langchain install-mem0 install-langmem install-bench install-llm-test install-viz install-docs install-release install-all lint format test cov typecheck meta-check meta-check-local check bench-perf bench-fidelity bench bench-appraisal bench-comparative bench-realistic bench-realistic-hash bench-ablation bench-ablation-hash human-eval-packets human-eval-summary reproduce-paper paper test-llm llm-config llm-config-strict demo-check demo-run docs-images docs docs-serve dist publish publish-pypi-manual verify-pypi-release sync-release-metadata zenodo-draft zenodo-publish release-check release-space clean help

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

install-langmem:
	uv pip install -e ".[dev,langmem]"

install-all:
	uv pip install -e ".[dev,demo,viz,docs,bench,llm-test,dotenv,sqlite,sentence-transformers,langchain,release]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term-missing

typecheck:
	uv run mypy src/emotional_memory/

meta-check:
	uv run python scripts/check_release_metadata.py

meta-check-local:
	uv run python scripts/check_release_metadata.py --require-local-doi

check: lint typecheck meta-check test

bench-fidelity:
	uv run pytest benchmarks/fidelity/ -v -m fidelity

bench-perf:
	uv run pytest benchmarks/perf/ --benchmark-only --benchmark-sort=mean

bench: bench-fidelity bench-perf

bench-comparative:
	uv run python -m benchmarks.comparative.runner

bench-realistic:
	uv run python -m benchmarks.realistic.runner --embedder sbert-bge

bench-realistic-hash:
	uv run python -m benchmarks.realistic.runner --embedder hash

bench-ablation:
	uv run python -m benchmarks.ablation.runner --embedder sbert-bge

bench-ablation-hash:
	uv run python -m benchmarks.ablation.runner --embedder hash

human-eval-packets:
	uv run python -m benchmarks.human_eval.pipeline packets

human-eval-summary:
	uv run python -m benchmarks.human_eval.pipeline summary

reproduce-paper:
	uv run python scripts/reproduce_paper.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

paper-arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
	mkdir -p paper/arxiv-build
	cp paper/main.tex paper/arxiv-build/
	cp paper/main.bbl paper/arxiv-build/
	cp paper/refs.bib paper/arxiv-build/
	cp -r paper/figures paper/arxiv-build/
	cp -r paper/tables paper/arxiv-build/
	tar -czf paper/arxiv-submission.tar.gz -C paper/arxiv-build .
	rm -rf paper/arxiv-build
	@echo "arXiv bundle ready: paper/arxiv-submission.tar.gz"
	@tar -tzf paper/arxiv-submission.tar.gz

bench-appraisal: llm-config-strict
	uv run pytest benchmarks/appraisal_quality/ -v -m appraisal_quality

test-llm: llm-config-strict
	uv run pytest tests/test_llm_integration.py -v -m llm

llm-config:
	uv run python scripts/check_llm_config.py

llm-config-strict:
	uv run python scripts/check_llm_config.py --strict --require-key

demo-check:
	uv run pytest tests/test_demo_ui_config.py -q
	uv run pytest tests/test_demo_app.py -q

demo-run:
	uv run python demo/app.py

docs-images:
	uv run python scripts/generate_docs_images.py

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

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

zenodo-publish:
	@test -n "$$ZENODO_TOKEN" || (echo "ZENODO_TOKEN not set"; exit 1)
	@test -n "$(DEPOSIT_ID)" || (echo "Usage: make zenodo-publish DEPOSIT_ID=123"; exit 1)
	uv run python scripts/zenodo_deposit.py --publish-id "$(DEPOSIT_ID)"

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
	@echo "  test                       pytest (unit + integration)"
	@echo "  cov                        pytest with branch coverage"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench-fidelity             126 psychological invariant tests"
	@echo "  bench-perf                 latency / throughput benchmarks"
	@echo "  bench                      fidelity + performance"
	@echo "  bench-appraisal            LLM appraisal quality (requires API key)"
	@echo "  bench-comparative          Cross-system comparison (v0.6)"
	@echo "  bench-realistic            Replayable multi-session benchmark with persisted state"
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
	@echo ""
	@echo "Release:"
	@echo "  release-check VERSION=x.y.z  Full release gate incl. real-LLM checks"
	@echo "  publish-pypi-manual       Manual PyPI upload via twine + PYPI_TOKEN"
	@echo "  verify-pypi-release VERSION=x.y.z  Poll PyPI until the release is visible"
	@echo "  sync-release-metadata     Sync Zenodo DOI metadata from .zenodo_doi"
	@echo "  zenodo-draft              Create/upload a Zenodo draft deposition"
	@echo "  zenodo-publish            Publish an existing Zenodo draft by DEPOSIT_ID"
	@echo "  release-space             Push demo subtree to the Hugging Face Space remote"
	@echo "  dist                       Build wheel + sdist"
	@echo "  publish                    Build and publish to PyPI"
	@echo "  clean                      Remove build artefacts"
