"""
Prometheus metrics for monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info("deepfake_detection_app", "Application information")

# Request metrics
REQUESTS_TOTAL = Counter(
    "deepfake_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "deepfake_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Analysis metrics
ANALYSES_TOTAL = Counter(
    "deepfake_analyses_total",
    "Total number of analyses",
    ["type", "verdict"],
)

ANALYSIS_DURATION = Histogram(
    "deepfake_analysis_duration_seconds",
    "Analysis duration in seconds",
    ["type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Job metrics
JOBS_IN_PROGRESS = Gauge(
    "deepfake_jobs_in_progress",
    "Number of jobs currently in progress",
)

JOBS_TOTAL = Counter(
    "deepfake_jobs_total",
    "Total number of jobs",
    ["status"],
)

# Model metrics
MODEL_INFERENCE_DURATION = Histogram(
    "deepfake_model_inference_seconds",
    "Model inference duration in seconds",
    ["model_name"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Storage metrics
STORAGE_OPERATIONS = Counter(
    "deepfake_storage_operations_total",
    "Total storage operations",
    ["operation", "status"],
)

# Error metrics
ERRORS_TOTAL = Counter(
    "deepfake_errors_total",
    "Total number of errors",
    ["type", "endpoint"],
)


def init_metrics(version: str, commit: str) -> None:
    """Initialize application metrics."""
    APP_INFO.info({
        "version": version,
        "commit": commit,
        "service": "deepfake-detection",
    })
