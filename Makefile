# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-demo install-sqlite install-redis install-sentence-transformers install-langchain install-mem0 install-langmem install-bench install-llm-test install-viz install-docs install-release install-all lint format test cov typecheck meta-check meta-check-local check check-all bench-perf bench-fidelity bench bench-appraisal bench-comparative bench-comparative-sbert bench-realistic bench-realistic-hash bench-realistic-v2-sbert bench-realistic-v2-e5 bench-realistic-it-sbert bench-realistic-it-e5 bench-ablation bench-ablation-sbert bench-ablation-hash bench-appraisal-confound bench-appraisal-confound-hash bench-locomo bench-locomo-dry human-eval-packets human-eval-summary reproduce-paper paper test-llm llm-config llm-config-strict demo-check demo-run docs-images research-figures figures docs docs-serve dist bump publish publish-pypi-manual verify-pypi-release sync-release-metadata zenodo-draft zenodo-publish release-check release-space clean help

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

check-all: check
	uv run mkdocs build --strict --quiet
	uv run python scripts/preflight.py --fast
	$(MAKE) reproduce-paper-check

bench-fidelity:
	uv run pytest benchmarks/fidelity/ -v -m fidelity

bench-perf:
	uv run pytest benchmarks/perf/ --benchmark-only --benchmark-sort=mean

bench: bench-fidelity bench-perf

bench-comparative:
	uv run python -m benchmarks.comparative.runner

bench-comparative-sbert:
	uv run python -m benchmarks.comparative.runner --embedder sbert --out benchmarks/comparative/results.sbert.csv

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

bench-locomo:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.runner

bench-locomo-dry:
	PYTHONUNBUFFERED=1 uv run python -m benchmarks.locomo.runner --limit-conversations 2 --limit-qa 5 --no-judge

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

research-figures:
	uv run python scripts/generate_research_figures.py

figures: docs-images research-figures

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

## Full automated release (API-driven, no GitHub webhook needed)
## Usage: make release VERSION=0.9.0
## Resume a failed run: make release VERSION=0.9.0 RELEASE_FLAGS=--resume
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
	@echo "  test                       pytest (unit + integration)"
	@echo "  cov                        pytest with branch coverage"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench-fidelity             126 psychological invariant tests"
	@echo "  bench-perf                 latency / throughput benchmarks"
	@echo "  bench                      fidelity + performance"
	@echo "  bench-appraisal            LLM appraisal quality (requires API key)"
	@echo "  bench-comparative          Cross-system comparison (hash embedder, quick)"
	@echo "  bench-comparative-sbert    Cross-system comparison (SBERT embedder, paper-canonical)"
	@echo "  bench-appraisal-confound   Appraisal confound study (SBERT, no LLM key)"
	@echo "  bench-appraisal-confound-hash  Appraisal confound study (hash embedder)"
	@echo "  bench-realistic            Replayable multi-session benchmark with persisted state"
	@echo "  bench-locomo               LoCoMo benchmark (requires EMOTIONAL_MEMORY_LLM_API_KEY)"
	@echo "  bench-locomo-dry           LoCoMo dry run: 2 conversations, 5 QA each, no judge"
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
