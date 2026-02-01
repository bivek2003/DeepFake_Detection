"""
Job status and results endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession
from app.api.schemas import (
    AnalysisStatus,
    ChartData,
    FrameScore,
    JobResult,
    JobStatus,
)
from app.logging_config import get_logger
from app.persistence import crud

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Get job status",
    description="Check the status and progress of a video analysis job.",
)
async def get_job_status(
    job_id: str,
    db: DBSession,
) -> JobStatus:
    """
    Get the current status of a video analysis job.
    
    Use this endpoint to poll for job completion.
    """
    analysis = await crud.get_analysis(db, job_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    # Calculate progress based on status
    progress = 0.0
    if analysis.status == AnalysisStatus.PENDING:
        progress = 0.0
    elif analysis.status == AnalysisStatus.PROCESSING:
        # Estimate progress based on analyzed frames
        if analysis.total_frames and analysis.total_frames > 0:
            frames = await crud.get_frames_for_analysis(db, job_id)
            progress = min(0.95, len(frames) / analysis.total_frames)
        else:
            progress = 0.5
    elif analysis.status == AnalysisStatus.COMPLETED:
        progress = 1.0
    elif analysis.status == AnalysisStatus.FAILED:
        progress = 0.0
    
    return JobStatus(
        job_id=job_id,
        status=analysis.status,
        progress=progress,
        message=_get_status_message(analysis.status, analysis.error),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        error=analysis.error,
    )


@router.get(
    "/jobs/{job_id}/result",
    response_model=JobResult,
    summary="Get job results",
    description="Get the complete results of a finished video analysis job.",
)
async def get_job_result(
    job_id: str,
    db: DBSession,
) -> JobResult:
    """
    Get the complete results of a video analysis job.
    
    Only available after job is completed.
    """
    analysis = await crud.get_analysis(db, job_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    
    if analysis.status == AnalysisStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Job is still pending. Please wait for processing to start.",
        )
    
    if analysis.status == AnalysisStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Job is still processing. Please poll /jobs/{job_id} for status.",
        )
    
    if analysis.status == AnalysisStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {analysis.error or 'Unknown error'}",
        )
    
    # Get frame scores
    frames = await crud.get_frames_for_analysis(db, job_id)
    frame_scores = [
        FrameScore(
            frame_index=f.frame_index,
            timestamp=f.timestamp,
            score=f.score,
            overlay_url=f"/api/v1/assets/{f.overlay_path}" if f.overlay_path else None,
        )
        for f in frames
    ]
    
    # Get suspicious frames (top 5 by score)
    suspicious_frames = sorted(frame_scores, key=lambda x: x.score, reverse=True)[:5]
    
    # Get assets
    assets = await crud.get_assets_for_analysis(db, job_id)
    
    report_url = None
    timeline_chart_url = None
    
    for asset in assets:
        if asset.kind == "report":
            report_url = f"/api/v1/reports/{job_id}.pdf"
        elif asset.kind == "timeline_chart":
            timeline_chart_url = f"/api/v1/assets/{asset.path}"
    
    # Build chart data
    chart_data = None
    if frame_scores:
        timeline_data = [
            {"timestamp": f.timestamp, "score": f.score, "frame": f.frame_index}
            for f in frame_scores
        ]
        
        # Calculate distribution buckets
        scores = [f.score for f in frame_scores]
        distribution = {
            "buckets": _calculate_distribution(scores),
            "mean": sum(scores) / len(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "min": min(scores) if scores else 0,
        }
        
        chart_data = ChartData(
            timeline=timeline_data,
            distribution=distribution,
        )
    
    return JobResult(
        job_id=job_id,
        status=analysis.status,
        verdict=analysis.verdict,
        confidence=analysis.confidence,
        sha256=analysis.sha256,
        model_version=analysis.model_version,
        runtime_ms=analysis.runtime_ms,
        device=analysis.device,
        created_at=analysis.created_at,
        completed_at=analysis.updated_at,
        total_frames=analysis.total_frames,
        analyzed_frames=len(frame_scores),
        frame_scores=frame_scores,
        suspicious_frames=suspicious_frames,
        chart_data=chart_data,
        report_url=report_url,
        timeline_chart_url=timeline_chart_url,
    )


def _get_status_message(status: AnalysisStatus, error: str | None) -> str:
    """Get human-readable status message."""
    messages = {
        AnalysisStatus.PENDING: "Job is queued for processing",
        AnalysisStatus.PROCESSING: "Video is being analyzed",
        AnalysisStatus.COMPLETED: "Analysis completed successfully",
        AnalysisStatus.FAILED: f"Analysis failed: {error or 'Unknown error'}",
    }
    return messages.get(status, "Unknown status")


def _calculate_distribution(scores: list[float]) -> list[dict]:
    """Calculate score distribution in buckets."""
    buckets = [
        {"range": "0.0-0.2", "min": 0.0, "max": 0.2, "count": 0},
        {"range": "0.2-0.4", "min": 0.2, "max": 0.4, "count": 0},
        {"range": "0.4-0.6", "min": 0.4, "max": 0.6, "count": 0},
        {"range": "0.6-0.8", "min": 0.6, "max": 0.8, "count": 0},
        {"range": "0.8-1.0", "min": 0.8, "max": 1.0, "count": 0},
    ]
    
    for score in scores:
        for bucket in buckets:
            if bucket["min"] <= score < bucket["max"] or (bucket["max"] == 1.0 and score == 1.0):
                bucket["count"] += 1
                break
    
    return buckets
