"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Type of analysis."""
    IMAGE = "image"
    VIDEO = "video"


class AnalysisStatus(str, Enum):
    """Status of an analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, Enum):
    """Detection verdict."""
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"


# =============================================================================
# Analysis Schemas
# =============================================================================

class ImageAnalysisResponse(BaseModel):
    """Response for image analysis."""
    id: str = Field(..., description="Analysis ID")
    verdict: Verdict = Field(..., description="Detection verdict: REAL, FAKE, or UNCERTAIN")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    heatmap_url: str | None = Field(None, description="URL to heatmap overlay image")
    sha256: str = Field(..., description="SHA256 hash of the uploaded file")
    model_version: str = Field(..., description="Model version used for analysis")
    runtime_ms: int = Field(..., description="Processing time in milliseconds")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    created_at: datetime = Field(..., description="Analysis timestamp")
    disclaimer: str = Field(
        default="This is a forensic estimate, not certainty. Results should be verified by experts.",
        description="Legal disclaimer",
    )

    model_config = {"from_attributes": True}


class VideoAnalysisResponse(BaseModel):
    """Response for video analysis (returns job ID for async processing)."""
    job_id: str = Field(..., description="Job ID for tracking async processing")
    status: AnalysisStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")


# =============================================================================
# Job Schemas
# =============================================================================

class JobStatus(BaseModel):
    """Job status response."""
    job_id: str = Field(..., description="Job ID")
    status: AnalysisStatus = Field(..., description="Current status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress (0-1)")
    message: str | None = Field(None, description="Status message")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    error: str | None = Field(None, description="Error message if failed")


class FrameScore(BaseModel):
    """Score for a single video frame."""
    frame_index: int = Field(..., description="Frame index in video")
    timestamp: float = Field(..., description="Timestamp in seconds")
    score: float = Field(..., ge=0.0, le=1.0, description="Fake probability score")
    overlay_url: str | None = Field(None, description="URL to heatmap overlay")


class ChartData(BaseModel):
    """Chart data for visualization."""
    timeline: list[dict[str, Any]] = Field(..., description="Timeline data points")
    distribution: dict[str, Any] = Field(..., description="Score distribution data")


class AnalysisListItem(BaseModel):
    """Single analysis for list/dashboard."""
    id: str = Field(..., description="Analysis/Job ID")
    type: AnalysisType = Field(..., description="image or video")
    status: AnalysisStatus = Field(..., description="Status")
    verdict: Verdict | None = Field(None, description="Verdict when completed")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence when completed")
    created_at: datetime = Field(..., description="Creation timestamp")


class JobResult(BaseModel):
    """Complete job result with analysis details."""
    job_id: str = Field(..., description="Job ID")
    status: AnalysisStatus = Field(..., description="Analysis status")
    verdict: Verdict | None = Field(None, description="Overall verdict")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    sha256: str = Field(..., description="SHA256 hash of uploaded file")
    model_version: str = Field(..., description="Model version used")
    runtime_ms: int | None = Field(None, description="Total processing time in ms")
    device: str = Field(..., description="Device used for inference")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    
    # Video-specific fields
    total_frames: int | None = Field(None, description="Total frames in video")
    analyzed_frames: int | None = Field(None, description="Number of analyzed frames")
    frame_scores: list[FrameScore] = Field(default_factory=list, description="Per-frame scores")
    suspicious_frames: list[FrameScore] = Field(
        default_factory=list, description="Top suspicious frames"
    )
    chart_data: ChartData | None = Field(None, description="Visualization data")
    
    # Asset URLs
    heatmap_url: str | None = Field(None, description="URL to heatmap overlay (image analysis)")
    report_url: str | None = Field(None, description="URL to PDF report")
    timeline_chart_url: str | None = Field(None, description="URL to timeline chart image")
    
    disclaimer: str = Field(
        default="This is a forensic estimate, not certainty. Results should be verified by experts.",
        description="Legal disclaimer",
    )

    model_config = {"from_attributes": True}


# =============================================================================
# Model Info Schemas
# =============================================================================

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    commit_hash: str | None = Field(None, description="Git commit hash")
    calibration_method: str | None = Field(None, description="Calibration method used")
    demo_mode: bool = Field(..., description="Whether running in demo mode")
    device: str = Field(..., description="Device being used")
    metrics: dict[str, Any] | None = Field(None, description="Performance metrics if available")


# =============================================================================
# Health Check Schemas
# =============================================================================

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")


class ReadinessCheck(BaseModel):
    """Readiness check response."""
    status: str = Field(..., description="Readiness status")
    database: str = Field(..., description="Database connection status")
    redis: str = Field(..., description="Redis connection status")
    model: str = Field(..., description="Model status")


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., description="Error message")
    type: str | None = Field(None, description="Error type")
    code: str | None = Field(None, description="Error code")
