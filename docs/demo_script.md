# Demo Script (3 Minutes)

This script guides you through demonstrating the Deepfake Detection Platform to recruiters or stakeholders.

---

## Setup (30 seconds)

Before the demo, ensure the platform is running:

```bash
# Start all services
make up

# Verify services are healthy
make health
```

Open the browser to http://localhost:3000

---

## Part 1: Introduction (30 seconds)

**Say:**
> "This is a production-ready deepfake detection platform I built. It's designed for defensive media forensics - detecting manipulated content, not creating it."

**Show:**
- The dashboard with the demo mode banner
- Highlight: "You can see it's running in demo mode, which works without any external datasets or model weights"

---

## Part 2: Image Analysis (45 seconds)

**Say:**
> "Let me show you real-time image analysis."

**Do:**
1. Click "Image Analysis"
2. Drag and drop a sample image (or use the sample button)
3. Click "Analyze Image"

**Say:**
> "The system extracts faces, runs inference through our detection model, and generates a Grad-CAM heatmap showing which regions triggered the detection."

**Show:**
- The verdict card (REAL/FAKE with confidence)
- The heatmap overlay
- The analysis metadata (processing time, model version, file hash)

**Highlight:**
> "Notice the disclaimer - 'forensic estimate, not certainty.' We built ethical guardrails throughout the system."

---

## Part 3: Video Analysis (45 seconds)

**Say:**
> "For videos, we use async processing since it takes longer."

**Do:**
1. Click "Video Analysis"
2. Upload a sample video
3. Click "Start Analysis"

**Say:**
> "The job is queued to Celery workers. The frontend polls for progress automatically."

**Show:**
- The progress bar updating
- When complete, show:
  - Timeline chart of fake scores over time
  - Suspicious frames gallery
  - Score distribution

**Say:**
> "Each suspicious frame gets its own heatmap, and we generate a downloadable PDF report."

---

## Part 4: Technical Highlights (30 seconds)

**Switch to terminal or code:**

```bash
# Show running containers
docker-compose ps

# Show API docs
open http://localhost:8000/docs
```

**Say:**
> "The tech stack is production-grade:
> - FastAPI with async/await
> - PostgreSQL with Alembic migrations
> - Redis + Celery for job queue
> - React + TypeScript frontend
> - Full CI/CD with GitHub Actions
> - Prometheus metrics for monitoring"

**Highlight:**
> "Everything runs locally with one command - `make up`. No external dependencies needed for demo mode."

---

## Part 5: Code Quality (15 seconds)

**Show:**
- The test suite: `make test`
- Linting: `make lint`
- API documentation at `/docs`

**Say:**
> "The codebase has full type hints, structured JSON logging, comprehensive tests, and auto-generated API documentation."

---

## Closing (15 seconds)

**Say:**
> "This demonstrates my ability to:
> - Build production-ready full-stack applications
> - Design scalable async architectures
> - Integrate ML inference pipelines
> - Write maintainable, well-tested code
> - Consider ethics in AI applications"

**Questions?**

---

## Troubleshooting

If something goes wrong during the demo:

1. **Services won't start:**
   ```bash
   make clean && make up
   ```

2. **Frontend not loading:**
   - Check nginx: `docker-compose logs nginx`
   - Access backend directly: http://localhost:8000/docs

3. **Analysis hangs:**
   - Check worker: `docker-compose logs celery-worker`
   - Worker might be initializing the model

4. **Database errors:**
   - Run migrations: `make migrate`
