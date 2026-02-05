.PHONY: help up down build logs test lint format migrate clean shell backend-shell frontend-shell db-shell redis-cli train train-shell download-data extract-faces tensorboard

# Default target
help:
	@echo "Deepfake Detection Platform - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make build       - Build Docker images"
	@echo "  make logs        - View all service logs"
	@echo "  make logs-f      - Follow all service logs"
	@echo ""
	@echo "Training (GPU):"
	@echo "  make train-build     - Build training Docker image"
	@echo "  make train-shell     - Start training container with GPU"
	@echo "  make train           - Run training (Celeb-DF + FaceForensics, >90% target)"
	@echo "  make train-all       - Extract faces, prepare splits, then train (full pipeline)"
	@echo "  make train-production - Run PRODUCTION training (ensemble, 96%+ accuracy)"
	@echo "  make train-quick     - Run quick training (single model)"
	@echo "  make download-data   - Download datasets (Celeb-DF, DFDC preview)"
	@echo "  make download-faceforensics - Download FaceForensics++ (~40GB, use EU2 server)"
	@echo "  make extract-faces   - Extract faces from Celeb-DF and FaceForensics"
	@echo "  make prepare-ff-splits - Create FaceForensics split files"
	@echo "  make tensorboard     - Start TensorBoard monitoring"
	@echo "  make evaluate        - Evaluate trained model"
	@echo "  make gpu-check       - Verify GPU is accessible"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-backend  - Run backend tests"
	@echo "  make test-frontend - Run frontend tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make typecheck   - Run type checking"
	@echo ""
	@echo "Database:"
	@echo "  make migrate     - Run database migrations"
	@echo "  make migrate-new - Create new migration"
	@echo ""
	@echo "Utilities:"
	@echo "  make shell       - Shell into backend container"
	@echo "  make clean       - Clean up containers and volumes"

# =============================================================================
# Docker Compose Commands
# =============================================================================

up:
	@echo "Starting all services..."
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

down:
	@echo "Stopping all services..."
	docker compose down

build:
	@echo "Building Docker images..."
	docker compose build

logs:
	docker compose logs

logs-f:
	docker compose logs -f

# =============================================================================
# Training Commands (GPU)
# =============================================================================

train-build:
	@echo "Building training Docker image with CUDA..."
	docker compose -f docker-compose.train.yml build trainer

train-shell:
	@echo "Starting training container with GPU access..."
	@mkdir -p datasets checkpoints logs
	docker compose -f docker-compose.train.yml run --rm trainer bash

train:
	@echo "Starting training pipeline (uses Celeb-DF + FaceForensics when available)..."
	@mkdir -p datasets checkpoints logs
	@echo "Tip: Run 'make extract-faces' first to include FaceForensics in training."
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/train.py --config configs/train_config.yaml

train-production:
	@echo "Starting PRODUCTION training (ensemble, max accuracy)..."
	@mkdir -p datasets checkpoints logs
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/train_production.py \
		--config configs/production_config.yaml \
		--data-root /app/datasets \
		--model ensemble \
		--epochs 100 \
		--batch-size 16

download-faceforensics:
	@echo "Downloading FaceForensics++ (EU2 server, ~40GB, may take hours)..."
	@mkdir -p datasets
	docker compose -f docker-compose.train.yml run --rm -e NVIDIA_VISIBLE_DEVICES="" trainer \
		python /app/datasets/ff_download.py /app/datasets/FaceForensics \
		--datasets original Deepfakes Face2Face FaceSwap -c c23 --auto-accept --server EU2

train-quick:
	@echo "Starting quick training (EfficientNet-B4 only)..."
	@mkdir -p datasets checkpoints logs
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/train_production.py \
		--config configs/production_config.yaml \
		--model efficientnet_b4 \
		--epochs 30 \
		--batch-size 32

download-data:
	@echo "Downloading datasets..."
	@mkdir -p datasets
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/download_datasets.py

extract-faces:
	@echo "Extracting faces from Celeb-DF and FaceForensics videos..."
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/extract_faces.py --dataset all

prepare-ff-splits:
	@echo "Creating FaceForensics split files (ff_train.txt, ff_val.txt, ff_test.txt)..."
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/prepare_ff_splits.py

train-all: extract-faces prepare-ff-splits train
	@echo "Full pipeline complete: extract faces -> prepare splits -> train"

tensorboard:
	@echo "Starting TensorBoard..."
	docker compose -f docker-compose.train.yml up -d tensorboard
	@echo "TensorBoard: http://localhost:6006"

evaluate:
	@echo "Evaluating model..."
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/evaluate.py

export-model:
	@echo "Exporting model for production..."
	docker compose -f docker-compose.train.yml run --rm trainer python scripts/export_model.py

# =============================================================================
# Testing
# =============================================================================

test: test-backend test-frontend
	@echo "All tests completed!"

test-backend:
	@echo "Running backend tests..."
	docker compose exec -T backend pytest -v --tb=short

test-frontend:
	@echo "Running frontend tests..."
	docker compose exec -T frontend npm run test

test-e2e:
	@echo "Running E2E tests..."
	docker compose exec -T frontend npm run test:e2e

# =============================================================================
# Code Quality
# =============================================================================

lint: lint-backend lint-frontend
	@echo "Linting completed!"

lint-backend:
	@echo "Linting backend..."
	docker compose exec -T backend ruff check app tests

lint-frontend:
	@echo "Linting frontend..."
	docker compose exec -T frontend npm run lint

format: format-backend format-frontend
	@echo "Formatting completed!"

format-backend:
	@echo "Formatting backend..."
	docker compose exec -T backend ruff check --fix app tests

format-frontend:
	@echo "Formatting frontend..."
	docker compose exec -T frontend npm run format

typecheck:
	@echo "Running type checks..."
	docker compose exec -T backend mypy app
	docker compose exec -T frontend npm run typecheck

# =============================================================================
# Database
# =============================================================================

migrate:
	@echo "Running database migrations..."
	docker compose exec -T backend alembic upgrade head

migrate-new:
	@read -p "Migration message: " msg; \
	docker compose exec -T backend alembic revision --autogenerate -m "$$msg"

migrate-down:
	docker compose exec -T backend alembic downgrade -1

# =============================================================================
# Shell Access
# =============================================================================

shell: backend-shell

backend-shell:
	docker compose exec backend bash

frontend-shell:
	docker compose exec frontend sh

db-shell:
	docker compose exec postgres psql -U deepfake -d deepfake_detection

redis-cli:
	docker compose exec redis redis-cli

# =============================================================================
# Development Utilities
# =============================================================================

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

dev-worker:
	cd backend && celery -A app.workers.celery_app worker --loglevel=info

# =============================================================================
# GPU Utilities
# =============================================================================

gpu-check:
	@echo "Checking GPU availability..."
	docker compose -f docker-compose.train.yml run --rm trainer python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up..."
	docker compose down -v --remove-orphans
	docker compose -f docker-compose.train.yml down -v --remove-orphans 2>/dev/null || true
	docker system prune -f
	rm -rf backend/.pytest_cache
	rm -rf backend/.mypy_cache
	rm -rf backend/.ruff_cache
	rm -rf frontend/node_modules/.cache
	@echo "Cleanup completed!"

clean-uploads:
	@echo "Cleaning upload directories..."
	rm -rf uploads/*
	rm -rf assets/*
	@echo "Upload directories cleaned!"

clean-training:
	@echo "Cleaning training artifacts..."
	rm -rf checkpoints/*
	rm -rf logs/*
	@echo "Training artifacts cleaned!"

# =============================================================================
# Production
# =============================================================================

prod-build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# =============================================================================
# Health Checks
# =============================================================================

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/api/v1/healthz | jq . || echo "Backend not responding"
	@curl -s http://localhost:8000/api/v1/readyz | jq . || echo "Backend not ready"
	@echo ""
