.PHONY: help install install-dev test lint format clean setup-dirs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linting
	flake8 src/ tests/
	black --check src/ tests/

format:  ## Format code
	black src/ tests/

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

setup-dirs:  ## Create necessary directories
	mkdir -p datasets logs models preprocessed notebooks tests
	mkdir -p src/deepfake_detector/data
	mkdir -p src/deepfake_detector/models  
	mkdir -p src/deepfake_detector/utils

setup: setup-dirs install  ## Full project setup

# Development commands
run-notebooks:  ## Start Jupyter lab
	jupyter lab

download-sample:  ## Download sample data (placeholder)
	@echo "Sample data download will be implemented in Phase 1"

preprocess:  ## Run preprocessing pipeline
	python -m deepfake_detector.data.preprocessing

train:  ## Train models (Phase 2)
	@echo "Training will be implemented in Phase 2"
