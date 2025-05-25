.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean dev api dashboard cli

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests with coverage"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e      Run end-to-end tests only"
	@echo "  lint          Run all linting tools"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run mypy type checking"
	@echo "  clean         Clean up cache and build files"
	@echo "  dev           Start development environment"
	@echo "  api           Start FastAPI server"
	@echo "  dashboard     Start Streamlit dashboard"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

test-watch:
	pytest --cov=src --cov-report=term-missing -f

# Code Quality
lint: format type-check
	@echo "Running all linting tools..."

format:
	@echo "Formatting code with black..."
	black src/ tests/ scripts/
	@echo "Sorting imports with isort..."
	isort src/ tests/ scripts/

type-check:
	@echo "Running mypy type checking..."
	mypy src/

# Alternative linting with ruff
ruff-check:
	ruff check src/ tests/

ruff-fix:
	ruff check --fix src/ tests/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

# Development servers
dev:
	@echo "Starting development environment..."
	@echo "API will be available at http://localhost:8000"
	@echo "Dashboard will be available at http://localhost:8501"

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/main.py --server.port 8501

# Data and model commands
data-collect:
	python scripts/download_data.py

train-models:
	python scripts/train_models.py

backtest:
	python scripts/run_backtest.py

# CLI
cli:
	tspmo --help
