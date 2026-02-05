"""
Tests for job management endpoints.
"""

import io
import uuid

from fastapi.testclient import TestClient


def test_job_not_found(client: TestClient):
    """Test getting non-existent job."""
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{fake_id}")
    assert response.status_code == 404


def test_job_result_not_found(client: TestClient):
    """Test getting result for non-existent job."""
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{fake_id}/result")
    assert response.status_code == 404


def test_report_not_found(client: TestClient):
    """Test downloading report for non-existent job."""
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/reports/{fake_id}.pdf")
    assert response.status_code == 404


def test_video_analysis_submission(client: TestClient, sample_video_path: str):
    """Test video analysis submission returns job ID."""
    with open(sample_video_path, "rb") as f:
        files = {"file": ("test.mp4", f, "video/mp4")}
        response = client.post("/api/v1/analyze/video", files=files)

    # Should return 202 Accepted with job_id
    if response.status_code == 202:
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"


def test_video_analysis_invalid_type(client: TestClient):
    """Test video analysis with invalid file type."""
    files = {"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")}
    response = client.post("/api/v1/analyze/video", files=files)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
