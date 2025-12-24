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

docker-build: ## Build Docker image
	docker build -t stock-prediction-api:latest .

docker-run: ## Run with docker-compose
	docker-compose up -d

docker-dev: ## Run with docker-compose in development mode
	docker-compose -f docker-compose.dev.yml up -d

docker-stop: ## Stop docker-compose services
	docker-compose down

docker-logs: ## View docker-compose logs
	docker-compose logs -f

docker-clean: ## Remove containers, volumes, and images
	docker-compose down -v
	docker rmi stock-prediction-api:latest || true

docker-ps: ## Show running containers
	docker-compose ps

# ====================
# Production Commands
# ====================

prod-build: ## Build Docker image for production
	docker-compose -f docker-compose.prod.yml build

prod-up: ## Start production services (PostgreSQL + MLflow + API)
	docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

prod-down: ## Stop production services
	docker-compose -f docker-compose.prod.yml down

prod-restart: ## Restart production services
	docker-compose -f docker-compose.prod.yml restart

prod-logs: ## View production logs
	docker-compose -f docker-compose.prod.yml logs -f

prod-logs-api: ## View API logs only
	docker-compose -f docker-compose.prod.yml logs -f api

prod-logs-mlflow: ## View MLflow logs only
	docker-compose -f docker-compose.prod.yml logs -f mlflow-server

prod-status: ## Check status of production services
	docker-compose -f docker-compose.prod.yml ps

prod-clean: ## Clean production volumes (WARNING: deletes data!)
	docker-compose -f docker-compose.prod.yml down -v

prod-backup-db: ## Backup PostgreSQL database
	docker exec mlflow-postgres pg_dump -U mlflow mlflow > backup_$(shell date +%Y%m%d_%H%M%S).sql

prod-restore-db: ## Restore PostgreSQL database (usage: make prod-restore-db BACKUP_FILE=backup.sql)
	docker exec -i mlflow-postgres psql -U mlflow mlflow < $(BACKUP_FILE)

prod-health: ## Check health of all production services
	@echo "Checking API health..."
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool || echo "❌ API is down"
	@echo "\nChecking MLflow health..."
	@curl -s http://localhost:5000/health | python -m json.tool || echo "❌ MLflow is down"
	@echo "\nChecking PostgreSQL health..."
	@docker exec mlflow-postgres pg_isready -U mlflow && echo "✅ PostgreSQL is healthy" || echo "❌ PostgreSQL is down"

prod-test-train: ## Test training in production environment
	curl -X POST http://localhost:8000/api/v1/train \
		-H "Content-Type: application/json" \
		-d '{"symbol": "AAPL", "period": "1y", "config": {"epochs": 50}}'

prod-test-predict: ## Test prediction in production environment
	curl -X POST http://localhost:8000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"symbol": "AAPL", "model_version": "latest"}'

prod-list-models: ## List all registered models
	curl -s http://localhost:8000/api/v1/models | python -m json.tool

# ====================
# Monitoring & Debugging
# ====================

monitor-resources: ## Monitor Docker resource usage
	docker stats

db-shell: ## Open PostgreSQL shell
	docker exec -it mlflow-postgres psql -U mlflow mlflow

api-shell: ## Open API container shell
	docker exec -it stock-prediction-api /bin/bash
