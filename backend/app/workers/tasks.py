"""
Celery tasks for async video processing.
Uses a single event loop per task to avoid "Future attached to a different loop" errors
with shared DB engine and async code.
"""

import asyncio
from pathlib import Path

from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.schemas import AnalysisStatus, Verdict
from app.logging_config import get_logger
from app.metrics import JOBS_IN_PROGRESS, JOBS_TOTAL
from app.ml.inference_video import analyze_video, generate_frame_heatmap
from app.ml.model_registry import ModelRegistry
from app.persistence.db import get_session_maker, create_engine
from app.persistence import crud
from app.services.plotting import (
    generate_distribution_chart,
    generate_suspicious_frames_montage,
    generate_timeline_chart,
)
from app.services.reporting import generate_pdf_report
from app.services.storage import StorageService
from app.settings import get_settings
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


def run_async(coro, loop: asyncio.AbstractEventLoop):
    """Run async function in the given event loop (sync context)."""
    return loop.run_until_complete(coro)


@celery_app.task(bind=True)
def process_video_task(
    self,
    job_id: str,
    video_path: str,
    file_hash: str,
    model_filename: str | None = None,
):
    """
    Process video for deepfake detection.
    
    Uses a single event loop for the whole task so DB engine and async code
    share the same loop (avoids "Future attached to a different loop").
    """
    JOBS_IN_PROGRESS.inc()
    
    # Single event loop for entire task (avoids loop mismatch with DB engine)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create a task-specific DB engine to ensure it's bound to this loop
    # Do NOT use the global cached engine from get_session_maker/get_async_engine
    engine = create_engine()
    
    session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    
    try:
        logger.info(f"Starting video processing for job {job_id} with model {model_filename}")
        
        # Get settings and services
        settings = get_settings()
        storage = StorageService(
            upload_dir=settings.upload_dir,
            assets_dir=settings.assets_dir,
        )
        
        # Initialize model registry (in this loop)
        registry = ModelRegistry()
        run_async(registry.initialize(), loop)
        
        # Switch to requested model if specified
        if model_filename:
             run_async(registry.switch_model(model_filename), loop)
        
        # Update status to processing
        # Note: We use the local session_maker, not the global one
        async def update_status():
            async with session_maker() as db:
                await crud.update_analysis_status(db, job_id, AnalysisStatus.PROCESSING)
        run_async(update_status(), loop)
        
        # Run video analysis
        upload_path = storage.get_upload_path(video_path)
        result = run_async(analyze_video(upload_path, registry, settings), loop)
        
        logger.info(f"Video analysis complete: {result.verdict.value}")
        
        # Save frame results and generate heatmaps for suspicious frames
        suspicious_set = {id(f) for f in result.suspicious_frames}
        
        async def save_frames():
            async with session_maker() as db:
                for frame_result in result.frame_results:
                    overlay_path = None
                    if id(frame_result) in suspicious_set and frame_result.frame_image is not None:
                        heatmap_bytes = generate_frame_heatmap(frame_result.frame_image)
                        overlay_filename = f"{job_id}_frame_{frame_result.frame_index}_heatmap.png"
                        overlay_path = await storage.save_asset(
                            heatmap_bytes,
                            overlay_filename,
                            subfolder="heatmaps",
                        )
                    await crud.create_frame(
                        db,
                        analysis_id=job_id,
                        frame_index=frame_result.frame_index,
                        timestamp=frame_result.timestamp,
                        score=frame_result.score,
                        overlay_path=overlay_path,
                    )
        run_async(save_frames(), loop)
        
        # Generate charts
        timestamps = [r.timestamp for r in result.frame_results]
        scores = [r.score for r in result.frame_results]
        
        timeline_chart = generate_timeline_chart(timestamps, scores)
        distribution_chart = generate_distribution_chart(scores)
        
        # Generate suspicious frames montage
        suspicious_images = [r.frame_image for r in result.suspicious_frames if r.frame_image is not None]
        suspicious_scores = [r.score for r in result.suspicious_frames if r.frame_image is not None]
        suspicious_timestamps = [r.timestamp for r in result.suspicious_frames if r.frame_image is not None]
        
        montage = None
        if suspicious_images:
            montage = generate_suspicious_frames_montage(
                suspicious_images,
                suspicious_scores,
                suspicious_timestamps,
            )
        
        # Save charts as assets
        async def save_assets():
            async with session_maker() as db:
                timeline_path = await storage.save_asset(
                    timeline_chart,
                    f"{job_id}_timeline.png",
                    subfolder="charts",
                )
                await crud.create_asset(db, job_id, "timeline_chart", timeline_path)
                dist_path = await storage.save_asset(
                    distribution_chart,
                    f"{job_id}_distribution.png",
                    subfolder="charts",
                )
                await crud.create_asset(db, job_id, "distribution_chart", dist_path)
                return timeline_path
        
        run_async(save_assets(), loop)
        
        # Get analysis record for report
        async def get_analysis():
            async with session_maker() as db:
                return await crud.get_analysis(db, job_id)
        
        analysis = run_async(get_analysis(), loop)
        
        # Generate PDF report
        pdf_report = generate_pdf_report(
            job_id=job_id,
            verdict=result.verdict.value,
            confidence=result.confidence,
            sha256=file_hash,
            model_version=registry.model_version,
            runtime_ms=result.runtime_ms,
            device=registry.device,
            created_at=analysis.created_at,
            total_frames=result.total_frames,
            analyzed_frames=result.analyzed_frames,
            timeline_chart=timeline_chart,
            distribution_chart=distribution_chart,
            suspicious_frames_montage=montage,
        )
        
        # Save PDF report
        async def save_report():
            async with session_maker() as db:
                report_path = await storage.save_asset(
                    pdf_report,
                    f"{job_id}_report.pdf",
                    subfolder="reports",
                )
                await crud.create_asset(db, job_id, "report", report_path)
        
        run_async(save_report(), loop)
        
        # Update analysis with results
        async def complete_analysis():
            async with session_maker() as db:
                await crud.update_analysis_result(
                    db,
                    job_id,
                    verdict=result.verdict,
                    confidence=result.confidence,
                    runtime_ms=result.runtime_ms,
                    total_frames=result.total_frames,
                )
        run_async(complete_analysis(), loop)
        
        JOBS_TOTAL.labels(status="completed").inc()
        logger.info(f"Video processing completed for job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "verdict": result.verdict.value,
            "confidence": result.confidence,
        }
        
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {e}", exc_info=True)
        
        async def fail_analysis():
            # Use the local session_maker
            async with session_maker() as db:
                await crud.update_analysis_status(
                    db,
                    job_id,
                    AnalysisStatus.FAILED,
                    error=str(e),
                )
        try:
            run_async(fail_analysis(), loop)
        except Exception:
            pass
        
        JOBS_TOTAL.labels(status="failed").inc()
        raise
        
    finally:
        JOBS_IN_PROGRESS.dec()
        # Dispose engine (closes connections properly for this loop)
        run_async(engine.dispose(), loop)
        loop.close()