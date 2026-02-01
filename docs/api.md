# API Documentation

This document provides examples for all API endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Authentication is disabled by default for demo mode. When enabled (`AUTH_ENABLED=true`),
include the JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

---

## Health Checks

### Liveness Check

```bash
curl http://localhost:8000/api/v1/healthz
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Readiness Check

```bash
curl http://localhost:8000/api/v1/readyz
```

Response:
```json
{
  "status": "ready",
  "database": "healthy",
  "redis": "healthy",
  "model": "healthy"
}
```

---

## Image Analysis

### Analyze Image

Synchronously analyze an image for deepfake detection.

```bash
curl -X POST http://localhost:8000/api/v1/analyze/image \
  -F "file=@/path/to/image.jpg"
```

Response (200 OK):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "verdict": "FAKE",
  "confidence": 0.87,
  "heatmap_url": "/api/v1/assets/heatmaps/550e8400_heatmap.png",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "model_version": "1.0.0-demo",
  "runtime_ms": 234,
  "device": "cpu",
  "created_at": "2024-01-15T10:30:00Z",
  "disclaimer": "This is a forensic estimate, not certainty. Results should be verified by experts."
}
```

Error Responses:
- `400 Bad Request`: Invalid file type
- `413 Request Entity Too Large`: File too large
- `500 Internal Server Error`: Analysis failed

---

## Video Analysis

### Submit Video for Analysis

Submit a video for asynchronous analysis.

```bash
curl -X POST http://localhost:8000/api/v1/analyze/video \
  -F "file=@/path/to/video.mp4"
```

Response (202 Accepted):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video submitted for processing. Use job ID to check status."
}
```

### Check Job Status

Poll for job status and progress.

```bash
curl http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```

Response (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.45,
  "message": "Video is being analyzed",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:31:00Z",
  "error": null
}
```

Status values:
- `pending`: Job is queued
- `processing`: Job is being processed
- `completed`: Job finished successfully
- `failed`: Job failed (check `error` field)

### Get Job Results

Get complete results after job completes.

```bash
curl http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result
```

Response (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "verdict": "FAKE",
  "confidence": 0.82,
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "model_version": "1.0.0-demo",
  "runtime_ms": 12500,
  "device": "cpu",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:32:30Z",
  "total_frames": 300,
  "analyzed_frames": 50,
  "frame_scores": [
    {"frame_index": 0, "timestamp": 0.0, "score": 0.65, "overlay_url": null},
    {"frame_index": 6, "timestamp": 0.2, "score": 0.72, "overlay_url": null}
  ],
  "suspicious_frames": [
    {"frame_index": 45, "timestamp": 1.5, "score": 0.92, "overlay_url": "/api/v1/assets/heatmaps/..."}
  ],
  "chart_data": {
    "timeline": [
      {"timestamp": 0.0, "score": 0.65, "frame": 0},
      {"timestamp": 0.2, "score": 0.72, "frame": 6}
    ],
    "distribution": {
      "buckets": [
        {"range": "0.0-0.2", "count": 5},
        {"range": "0.2-0.4", "count": 8}
      ],
      "mean": 0.65,
      "max": 0.92,
      "min": 0.35
    }
  },
  "report_url": "/api/v1/reports/550e8400-e29b-41d4-a716-446655440000.pdf",
  "timeline_chart_url": "/api/v1/assets/charts/550e8400_timeline.png",
  "disclaimer": "This is a forensic estimate, not certainty. Results should be verified by experts."
}
```

Error responses:
- `404 Not Found`: Job not found
- `425 Too Early`: Job still processing

---

## Reports

### Download PDF Report

```bash
curl -o report.pdf http://localhost:8000/api/v1/reports/550e8400-e29b-41d4-a716-446655440000.pdf
```

Response: PDF file download

---

## Assets

### Get Asset

Retrieve generated assets (heatmaps, charts).

```bash
curl http://localhost:8000/api/v1/assets/heatmaps/550e8400_heatmap.png > heatmap.png
```

Response: Image file

---

## Model Information

### Get Model Info

```bash
curl http://localhost:8000/api/v1/model/info
```

Response:
```json
{
  "model_name": "demo-detector",
  "model_version": "1.0.0-demo",
  "commit_hash": null,
  "calibration_method": "none",
  "demo_mode": true,
  "device": "cpu",
  "metrics": {
    "note": "Demo mode - scores are deterministic but not real predictions"
  }
}
```

---

## Metrics

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Response: Prometheus text format

---

## Rate Limiting

When rate limiting is enabled, responses include headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
```

When exceeded (429 Too Many Requests):
```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "retry_after": 60
}
```

---

## Error Format

All errors follow this format:

```json
{
  "detail": "Error message",
  "type": "ErrorType",
  "code": "ERROR_CODE"
}
```

Common HTTP status codes:
- `400`: Bad Request (validation error)
- `401`: Unauthorized (auth required)
- `404`: Not Found
- `413`: Payload Too Large
- `425`: Too Early (job not ready)
- `429`: Too Many Requests
- `500`: Internal Server Error
