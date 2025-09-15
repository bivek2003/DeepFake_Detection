.PHONY: help install install-dev test test-quick test-unit test-integration lint format clean setup-dirs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pip install pytest pytest-cov pytest-mock

# Testing commands
test:  ## Run all tests
	pytest tests/ -v --cov=src/deepfake_detector --cov-report=term-missing

test-quick:  ## Run quick Phase 1 test (no external datasets needed)
	python test_phase1_quick.py

test-unit:  ## Run unit tests only
	pytest tests/ -v -m "not integration"

test-integration:  ## Run integration tests only  
	pytest tests/ -v -m "integration"

test-coverage:  ## Run tests with detailed coverage report
	pytest tests/ --cov=src/deepfake_detector --cov-report=html --cov-report=term

test-dataset:  ## Test dataset management
	pytest tests/test_dataset_manager.py -v

test-video:  ## Test video processing
	pytest tests/test_video_processor.py -v

test-audio:  ## Test audio processing
	pytest tests/test_audio_processor.py -v

# Code quality
lint:  ## Run linting
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	black --check src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

setup-dirs:  ## Create necessary directories
	mkdir -p datasets logs models preprocessed notebooks tests
	mkdir -p src/deepfake_detector/{data,models,utils}
	mkdir -p tests/{unit,integration}

setup: setup-dirs install  ## Full project setup

# Development commands
demo:  ## Run Phase 1 demo
	python examples/phase1_demo.py

test-demo:  ## Run quick testing demo
	python test_phase1_quick.py

validate:  ## Validate Phase 1 implementation
	@echo "🔍 Validating Phase 1 implementation..."
	@python -c "from deepfake_detector.data import DatasetManager; print('✅ Dataset management OK')"
	@python -c "from deepfake_detector.data import VideoProcessor; print('✅ Video processing OK')"
	@python -c "from deepfake_detector.data import AudioProcessor; print('✅ Audio processing OK')"
	@python -c "from deepfake_detector.utils import ConfigManager; print('✅ Configuration OK')"
	@echo "✅ All core components validated!"

# Benchmarking
benchmark:  ## Run performance benchmarks
	python -m pytest tests/ -v --benchmark-only

# Documentation
docs:  ## Generate documentation
	@echo "📚 Documentation available in README.md"
	@echo "🧪 Testing guide available in test documentation"

# CI/CD simulation
ci:  ## Simulate CI/CD pipeline
	make clean
	make install-dev
	make lint
	make test-quick
	make test-unit
	@echo "🎉 CI/CD simulation complete!"
