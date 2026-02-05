# Deepfake Detection Platform

A production-ready defensive media forensics platform for detecting manipulated images and videos. **No deepfake generation capabilities** - this is a detection-only system.

[![CI](https://github.com/bivek2003/DeepFake_Detection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bivek2003/DeepFake_Detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **Image Analysis**: Upload images for instant deepfake detection with confidence scores and heatmap overlays
- **Video Analysis**: Async video processing with frame-by-frame analysis, timeline charts, and suspicious frame detection
- **Explainability**: Grad-CAM heatmaps showing which regions triggered detection
- **PDF Reports**: Downloadable forensic reports with metadata, charts, and limitations disclosure
- **Demo Mode**: Works out-of-the-box without external datasets or pretrained weights

## Quick Start (Demo Mode)

```bash
# Clone the repository
git clone https://github.com/bivek2003/DeepFake_Detection.git
cd DeepFake_Detection

# Copy environment file
cp .env.example .env

# Start all services
make up

# Open browser
open http://localhost:3000
```

The platform runs in **Demo Mode** by default, providing realistic detection outputs without requiring model weights.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Nginx       │────▶│    Backend      │
│  React + Vite   │     │  Reverse Proxy  │     │    FastAPI      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                ▼
                ┌───────────────┐              ┌─────────────────┐              ┌─────────────────┐
                │   PostgreSQL  │              │      Redis      │              │  Celery Worker  │
                │   Database    │              │     Cache       │              │  Video Jobs     │
                └───────────────┘              └─────────────────┘              └─────────────────┘
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEMO_MODE` | Enable demo mode (no real weights) | `true` |
| `AUTH_ENABLED` | Enable JWT authentication | `false` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `MAX_FILE_SIZE_MB` | Maximum upload size | `100` |
| `LOG_LEVEL` | Logging level | `INFO` |

See `.env.example` for all available options.

## Development

### Prerequisites

- Docker & Docker Compose
- Make (optional but recommended)
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### Commands

```bash
# Start all services
make up

# Stop all services
make down

# Run tests
make test

# Run linting
make lint

# Format code
make format

# View logs
make logs

# Database migrations
make migrate

# Clean up
make clean
```

### Local Development (without Docker)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- API Reference: [docs/api.md](docs/api.md)

### Quick API Examples

```bash
# Analyze image
curl -X POST http://localhost:8000/api/v1/analyze/image \
  -F "file=@test_image.jpg"

# Analyze video (returns job_id)
curl -X POST http://localhost:8000/api/v1/analyze/video \
  -F "file=@test_video.mp4"

# Check job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Download PDF report
curl -O http://localhost:8000/api/v1/reports/{job_id}.pdf
```

## Project Structure

```
/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes and schemas
│   │   ├── ml/             # ML inference and explainability
│   │   ├── services/       # Business logic services
│   │   ├── workers/        # Celery tasks
│   │   └── persistence/    # Database models and CRUD
│   ├── migrations/         # Alembic migrations
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   └── src/
│       ├── pages/          # Page components
│       └── components/     # Reusable components
├── infra/                  # Infrastructure configs
│   └── nginx/              # Nginx configuration
├── docs/                   # Documentation
└── docker-compose.yml
```

## Ethical Considerations

This platform is designed for **defensive forensics only**:

- **No Generation**: Cannot create deepfakes or synthetic media
- **Disclaimer**: All outputs include "Forensic estimate, not certainty"
- **Limitations**: Model card documents known failure modes
- **Privacy**: Uploaded media is processed locally and can be configured to auto-delete

See [docs/model_card.md](docs/model_card.md) for detailed limitations and ethical guidelines.

## Troubleshooting

### Common Issues

**Services won't start**
```bash
# Check Docker is running
docker info

# Check ports aren't in use
lsof -i :3000 -i :8000 -i :5432 -i :6379

# Reset everything
make clean && make up
```

**Database connection errors**
```bash
# Wait for PostgreSQL to be ready
make migrate
```

**"Network Error" when analyzing image/video**
- Open the app via **http://localhost:3000** (nginx), not http://localhost:5173. The frontend uses same-origin API calls so nginx can proxy `/api` to the backend.
- Restart frontend after changing API config: `docker compose restart frontend`
- Ensure backend is up: `curl http://localhost:8000/api/v1/healthz`

**Frontend can't connect to backend**
```bash
# Ensure nginx is routing correctly
docker compose logs nginx
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## Acknowledgments

- Built with FastAPI, React, and PyTorch
- Heatmap visualization using Grad-CAM techniques
- PDF generation with ReportLab
