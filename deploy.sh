#!/bin/bash
# Deployment script for production

set -e

echo "================================"
echo "Deepfake Detection API Deployment"
echo "================================"

# Check if model exists
if [ ! -f "models/faceforensics_improved.pth" ]; then
    echo "Error: Model file not found!"
    echo "Please ensure models/faceforensics_improved.pth exists"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker compose build

# Start services
echo "Starting services..."
docker compose up -d

# Wait for health check
echo "Waiting for API to be healthy..."
sleep 10

# Test health endpoint
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API is healthy!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "================================"
echo "Deployment Complete!"
echo "================================"
echo "API URL: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Nginx URL: http://localhost"
echo ""
echo "View logs: docker-compose logs -f"
echo "Stop services: docker-compose down"
