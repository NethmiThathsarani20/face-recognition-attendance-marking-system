.PHONY: help install format lint test clean run setup-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

setup-dev: ## Setup development environment with pre-commit hooks
	pip install pre-commit
	pre-commit install
	@echo "Development environment setup complete!"

format: ## Format code with Ruff
	ruff format src/ tests/

lint: ## Lint code with Ruff
	ruff check src/ tests/ --fix

lint-check: ## Check code with Ruff (no fixes)
	ruff check src/ tests/

test: ## Run tests
	python tests/run_tests.py

test-coverage: ## Run tests with coverage
	coverage run tests/run_tests.py
	coverage report
	coverage html

clean: ## Clean up temporary files and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .ruff_cache/

run: ## Run the face recognition attendance system
	python run.py

demo: ## Run the demo script
	python demo.py

security: ## Run security checks
	bandit -r src/ -f json

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports --no-strict-optional

all-checks: lint-check type-check security test ## Run all code quality checks

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

build-docs: ## Build documentation (if you add docs later)
	@echo "Documentation building not yet configured"

docker-build: ## Build Docker image (if you add Docker later)
	@echo "Docker configuration not yet added"

install-hooks: setup-dev ## Alias for setup-dev

check: all-checks ## Alias for all-checks
