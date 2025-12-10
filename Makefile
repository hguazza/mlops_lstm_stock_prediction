.PHONY: help install run test lint format check clean pre-commit-install pre-commit-run

.DEFAULT_GOAL := help

# Variables for consistency
PYTHON := uv run

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv sync

run: ## Run the main application
	$(PYTHON) src/mlops_stock_prediction/main.py

test: ## Run all tests
	$(PYTHON) pytest

test-unit: ## Run unit tests only
	$(PYTHON) pytest tests/unit -v

test-fast: ## Run tests excluding slow ones
	$(PYTHON) pytest -m "not slow" -v

test-coverage: ## Run tests with coverage report
	$(PYTHON) pytest --cov=src --cov-report=term-missing --cov-report=html

check: ## Run linter checks only
	$(PYTHON) ruff check .

format: ## Format code with ruff
	$(PYTHON) ruff format .

lint: check format ## Run all linting (check + format)

clean: ## Remove cache and virtual environment
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

pre-commit-install: ## Install pre-commit hooks
	$(PYTHON) pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	$(PYTHON) pre-commit run --all-files

run-api: ## Run FastAPI development server
	$(PYTHON) uvicorn src.presentation.main:app --reload --host 0.0.0.0 --port 8000

run-api-prod: ## Run FastAPI production server
	$(PYTHON) uvicorn src.presentation.main:app --host 0.0.0.0 --port 8000 --workers 4
