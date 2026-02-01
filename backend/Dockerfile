# Backend Dockerfile - Python 3.11 + FastAPI (Minimal)
# Pre-built image with common dependencies
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy and install minimal requirements first
COPY requirements-minimal.txt ./
RUN pip install --upgrade pip && pip install -r requirements-minimal.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/uploads /app/assets /app/weights

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
