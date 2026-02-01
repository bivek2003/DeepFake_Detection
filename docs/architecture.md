# Architecture Overview

This document describes the system architecture of the Deepfake Detection Platform.

## System Diagram

```
                                    ┌──────────────────────────────────────────────────────────────┐
                                    │                     NGINX (Port 3000)                        │
                                    │                     Reverse Proxy                            │
                                    └─────────────────────────┬────────────────────────────────────┘
                                                              │
                        ┌─────────────────────────────────────┼─────────────────────────────────────┐
                        │                                     │                                     │
                        ▼                                     ▼                                     │
            ┌───────────────────────┐             ┌───────────────────────┐                        │
            │   Frontend (React)    │             │   Backend (FastAPI)   │                        │
            │   - Dashboard         │             │   - REST API          │                        │
            │   - Upload Pages      │             │   - ML Inference      │                        │
            │   - Result Viewer     │             │   - File Storage      │                        │
            │   Port: 5173          │             │   Port: 8000          │                        │
            └───────────────────────┘             └───────────┬───────────┘                        │
                                                              │                                     │
                ┌─────────────────────────────────────────────┼─────────────────────────────────────┤
                │                                             │                                     │
                ▼                                             ▼                                     ▼
    ┌───────────────────────┐             ┌───────────────────────┐             ┌───────────────────────┐
    │   PostgreSQL          │             │   Redis               │             │   Celery Worker       │
    │   - Analyses          │             │   - Cache             │             │   - Video Jobs        │
    │   - Frames            │             │   - Job Queue         │             │   - Report Gen        │
    │   - Assets            │             │   - Rate Limits       │             │   - Heatmaps          │
    │   Port: 5432          │             │   Port: 6379          │             │                       │
    └───────────────────────┘             └───────────────────────┘             └───────────────────────┘
```

## Component Details

### Frontend (React + TypeScript + Vite)

The frontend is a single-page application built with:

- **React 18**: UI framework with hooks
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool with HMR
- **TailwindCSS**: Utility-first CSS
- **Recharts**: Interactive charts
- **react-dropzone**: File upload handling

Key pages:
- `Dashboard`: Overview and recent analyses
- `UploadImage`: Synchronous image analysis
- `UploadVideo`: Async video analysis with polling
- `ResultDetail`: Detailed analysis results

### Backend (FastAPI + Python 3.11)

The backend is a REST API built with:

- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation with Python type hints
- **SQLAlchemy 2.0**: Async ORM for PostgreSQL
- **Celery**: Distributed task queue
- **PyTorch**: ML inference engine

Key modules:
- `api/`: REST endpoints and request handling
- `ml/`: Machine learning inference and explainability
- `services/`: Business logic (storage, reporting, etc.)
- `workers/`: Celery async tasks
- `persistence/`: Database models and CRUD

### Database (PostgreSQL)

Tables:
```sql
analyses
├── id (UUID)
├── type (image/video)
├── status (pending/processing/completed/failed)
├── verdict (REAL/FAKE/UNCERTAIN)
├── confidence (0-1)
├── sha256 (file hash for caching)
├── model_version
├── runtime_ms
├── device (cpu/cuda)
└── timestamps

frames
├── id
├── analysis_id (FK)
├── frame_index
├── timestamp
├── score
└── overlay_path

assets
├── id
├── analysis_id (FK)
├── kind (heatmap/chart/report)
└── path
```

### Cache & Queue (Redis)

Used for:
- Celery message broker and result backend
- Rate limiting token buckets
- Analysis result caching (optional)

### Worker (Celery)

Handles async tasks:
1. Video frame sampling
2. Per-frame inference
3. Heatmap generation for suspicious frames
4. Timeline chart generation
5. Distribution chart generation
6. PDF report generation

## Data Flow

### Image Analysis (Synchronous)

```
1. User uploads image
2. Backend validates file (type, size)
3. File saved to storage
4. Image preprocessed (face detection, resize)
5. Model inference → verdict + confidence
6. Grad-CAM heatmap generated
7. Result saved to database
8. Response returned immediately
```

### Video Analysis (Asynchronous)

```
1. User uploads video
2. Backend validates file (type, size)
3. File saved to storage
4. Job created in database (status: pending)
5. Celery task queued
6. Response with job_id returned
7. Worker picks up task
8. Video frames sampled
9. Per-frame inference
10. Suspicious frames identified
11. Heatmaps generated for top frames
12. Charts generated
13. PDF report generated
14. Database updated (status: completed)
15. Frontend polls until complete
```

## ML Pipeline

### Demo Mode (Default)

When `DEMO_MODE=true`:
- Uses deterministic scoring based on image statistics
- Generates plausible heatmaps using edge detection
- No real model weights required
- Full system functionality for testing

### Real Mode

When weights are available:
- Loads pretrained detection model
- Uses Grad-CAM for explainability
- Temperature scaling for calibration
- Caches results by file hash + model version

## Security Considerations

- File type validation (MIME + extension)
- File size limits (configurable, default 100MB)
- Rate limiting (configurable requests/window)
- CORS restrictions (configurable origins)
- JWT authentication (optional, off by default)
- SQL injection prevention (SQLAlchemy ORM)
- Input sanitization (Pydantic validation)

## Monitoring

- **Prometheus metrics**: `/metrics` endpoint
- **Health checks**: `/api/v1/healthz` (liveness), `/api/v1/readyz` (readiness)
- **Structured JSON logging**: All logs in JSON format
- **Request tracing**: X-Process-Time header
