# ShapleyIQ Makefile

.PHONY: help install install-dev test lint format clean build upload docs

# Default target
help:
	@echo "ShapleyIQ Development Commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies" 
	@echo "  test         Run tests with pytest"
	@echo "  lint         Run linting with flake8 and mypy"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI (requires credentials)"
	@echo "  docs         Build documentation"
	@echo "  demo         Run the demo script"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing targets  
test:
	pytest tests/ -v --cov=src/shapleyiq --cov-report=html --cov-report=term

test-quick:
	pytest tests/ -v

# Code quality targets
lint:
	flake8 src/shapleyiq tests/
	mypy src/shapleyiq

format:
	black src/shapleyiq tests/
	isort src/shapleyiq tests/

format-check:
	black --check src/shapleyiq tests/
	isort --check-only src/shapleyiq tests/

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Documentation targets
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Demo target
demo:
	python -m shapleyiq.example_usage

# Development workflow
dev-setup: install-dev
	pre-commit install

check-all: format-check lint test

# CI targets
ci-test: install-dev test lint

# Package verification
verify-install:
	python -c "import shapleyiq; print('ShapleyIQ installed successfully')"
	python -c "from shapleyiq import ShapleyValueRCA; print('Main algorithm imported successfully')"
