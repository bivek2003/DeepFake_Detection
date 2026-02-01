# Backend Dockerfile - Python 3.11 + FastAPI
# For GPU inference, use Dockerfile.gpu or install NVIDIA Container Toolkit
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for inference, GPU training uses Dockerfile.train)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install timm for model architecture
RUN pip install timm>=0.9.12

# Copy and install other requirements
COPY requirements-minimal.txt ./
RUN pip install -r requirements-minimal.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/uploads /app/assets /app/weights

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
