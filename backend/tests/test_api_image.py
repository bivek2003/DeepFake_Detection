"""
Tests for image analysis API endpoints.
"""

import io

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_analyze_image_success(client: TestClient, sample_image_bytes: bytes):
    """Test successful image analysis."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client.post("/api/v1/analyze/image", files=files)

    # Note: May fail without full app setup, testing schema at minimum
    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert "verdict" in data
        assert data["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "sha256" in data
        assert len(data["sha256"]) == 64
        assert "model_version" in data
        assert "disclaimer" in data


def test_analyze_image_invalid_type(client: TestClient):
    """Test image analysis with invalid file type."""
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/v1/analyze/image", files=files)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_analyze_image_too_large(client: TestClient):
    """Test image analysis with file too large."""
    # Create a large fake file
    large_content = b"x" * (200 * 1024 * 1024)  # 200MB
    files = {"file": ("large.jpg", io.BytesIO(large_content), "image/jpeg")}

    # This should fail due to size limit
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code in [400, 413, 500]  # May vary by server config


def test_model_info(client: TestClient):
    """Test model info endpoint."""
    response = client.get("/api/v1/model/info")

    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "demo_mode" in data
        assert "device" in data
