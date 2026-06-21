.PHONY: install install-runtime install-llamacpp test test-fast lint format type-check grade figures clean help

PYTHON := uv run python

help:
	@echo "Available commands:"
	@echo "  make install         Install core + dev deps (keyless Tier-1; what CI uses)"
	@echo "  make install-runtime Install the AirLLM model-runtime extra (GPU box / Tier-2)"
	@echo "  make install-llamacpp Add the llama.cpp competitor extra (Phase 5; build-fragile)"
	@echo "  make test            Run tests with coverage (excludes hardware-marked tests)"
	@echo "  make test-fast       Run tests without coverage (faster)"
	@echo "  make lint            Run ruff linter + format check (no auto-fix)"
	@echo "  make format          Run ruff formatter and auto-fix"
	@echo "  make type-check      Run mypy --strict on src/"
	@echo "  make grade           Full quality gate (lint + type-check + test + scanners)"
	@echo "  make figures         Regenerate report figures from results/ (keyless, offline)"
	@echo "  make clean           Remove caches and build artifacts"

install:
	uv sync

install-runtime:
	uv sync --extra runtime

install-llamacpp:
	uv sync --extra runtime --extra llamacpp

test:
	$(PYTHON) -m pytest -m "not hardware" --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=90

test-fast:
	$(PYTHON) -m pytest -x --no-cov

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

type-check:
	$(PYTHON) -m mypy src/

grade: lint type-check test
	$(PYTHON) scripts/check_file_sizes.py
	$(PYTHON) scripts/check_anti_patterns.py
	$(PYTHON) scripts/check_no_hardcoded.py
	$(PYTHON) scripts/check_raw_data.py
	$(PYTHON) scripts/check_report.py
	$(PYTHON) scripts/self_grade.py
	@echo ""
	@echo "All quality checks passed."

figures:
	$(PYTHON) -m airllm_bench.cli figures

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage coverage.xml htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
