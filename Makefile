.PHONY: help up down build logs test lint format migrate clean shell backend-shell frontend-shell db-shell redis-cli

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
	docker-compose up -d
	@echo ""
	@echo "Services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

down:
	@echo "Stopping all services..."
	docker-compose down

build:
	@echo "Building Docker images..."
	docker-compose build

logs:
	docker-compose logs

logs-f:
	docker-compose logs -f

# =============================================================================
# Testing
# =============================================================================

test: test-backend test-frontend
	@echo "All tests completed!"

test-backend:
	@echo "Running backend tests..."
	docker-compose exec -T backend pytest -v --tb=short

test-frontend:
	@echo "Running frontend tests..."
	docker-compose exec -T frontend npm run test

test-e2e:
	@echo "Running E2E tests..."
	docker-compose exec -T frontend npm run test:e2e

# =============================================================================
# Code Quality
# =============================================================================

lint: lint-backend lint-frontend
	@echo "Linting completed!"

lint-backend:
	@echo "Linting backend..."
	docker-compose exec -T backend ruff check app tests
	docker-compose exec -T backend black --check app tests

lint-frontend:
	@echo "Linting frontend..."
	docker-compose exec -T frontend npm run lint

format: format-backend format-frontend
	@echo "Formatting completed!"

format-backend:
	@echo "Formatting backend..."
	docker-compose exec -T backend black app tests
	docker-compose exec -T backend ruff check --fix app tests

format-frontend:
	@echo "Formatting frontend..."
	docker-compose exec -T frontend npm run format

typecheck:
	@echo "Running type checks..."
	docker-compose exec -T backend mypy app
	docker-compose exec -T frontend npm run typecheck

# =============================================================================
# Database
# =============================================================================

migrate:
	@echo "Running database migrations..."
	docker-compose exec -T backend alembic upgrade head

migrate-new:
	@read -p "Migration message: " msg; \
	docker-compose exec -T backend alembic revision --autogenerate -m "$$msg"

migrate-down:
	docker-compose exec -T backend alembic downgrade -1

# =============================================================================
# Shell Access
# =============================================================================

shell: backend-shell

backend-shell:
	docker-compose exec backend bash

frontend-shell:
	docker-compose exec frontend sh

db-shell:
	docker-compose exec postgres psql -U deepfake -d deepfake_detection

redis-cli:
	docker-compose exec redis redis-cli

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
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
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

# =============================================================================
# Production
# =============================================================================

prod-build:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# =============================================================================
# Health Checks
# =============================================================================

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/api/v1/healthz | jq . || echo "Backend not responding"
	@curl -s http://localhost:8000/api/v1/readyz | jq . || echo "Backend not ready"
	@echo ""
