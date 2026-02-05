"""
Analysis endpoints for image and video deepfake detection.
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import AppSettings, DBSession, Registry, Storage
from app.api.schemas import (
    AnalysisStatus,
    AnalysisType,
    ImageAnalysisResponse,
    VideoAnalysisResponse,
)
from app.logging_config import get_logger
from app.metrics import ANALYSES_TOTAL, ANALYSIS_DURATION
from app.ml.inference_image import analyze_image
from app.persistence import crud
from app.persistence.models import Analysis
from app.workers.tasks import process_video_task

logger = get_logger(__name__)

router = APIRouter()


def compute_sha256(content: bytes) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


@router.post(
    "/analyze/image",
    response_model=ImageAnalysisResponse,
    summary="Analyze image for deepfake",
    description="Upload an image for real-time deepfake detection analysis.",
)
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="Image file to analyze"),
    db: DBSession = None,
    registry: Registry = None,
    storage: Storage = None,
    settings: AppSettings = None,
) -> ImageAnalysisResponse:
    """
    Analyze a single image for deepfake manipulation.
    
    Returns verdict, confidence score, and optional heatmap overlay.
    """
    # Validate content type
    if file.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Allowed: {settings.allowed_image_types_list}",
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )
    
    # Compute hash
    file_hash = compute_sha256(content)
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    start_time = time.time()
    
    try:
        # Save uploaded file
        upload_path = await storage.save_upload(
            content=content,
            filename=f"{analysis_id}_{file.filename}",
            subfolder="images",
        )
        
        # Run analysis
        result = await analyze_image(
            content=content,
            registry=registry,
            settings=settings,
        )
        
        runtime_ms = int((time.time() - start_time) * 1000)
        
        # Save heatmap if generated and link as asset for job result
        heatmap_url = None
        heatmap_path = None
        if result.heatmap is not None:
            heatmap_path = await storage.save_asset(
                content=result.heatmap,
                filename=f"{analysis_id}_heatmap.png",
                subfolder="heatmaps",
            )
            heatmap_url = f"/api/v1/assets/{heatmap_path}"
        
        # Create database record FIRST (before creating assets that reference it)
        analysis = Analysis(
            id=analysis_id,
            type=AnalysisType.IMAGE,
            status=AnalysisStatus.COMPLETED,
            verdict=result.verdict,
            confidence=result.confidence,
            sha256=file_hash,
            model_version=registry.model_version,
            runtime_ms=runtime_ms,
            device=registry.device,
            upload_path=upload_path,
        )
        
        db.add(analysis)
        await db.commit()
        
        # Now create asset record (after analysis exists in DB)
        if heatmap_path is not None:
            await crud.create_asset(db, analysis_id, "heatmap", heatmap_path)
            await db.commit()
        
        # Record metrics
        ANALYSES_TOTAL.labels(type="image", verdict=result.verdict.value).inc()
        ANALYSIS_DURATION.labels(type="image").observe(runtime_ms / 1000)
        
        logger.info(
            "Image analysis completed",
            extra={
                "analysis_id": analysis_id,
                "verdict": result.verdict.value,
                "confidence": result.confidence,
                "runtime_ms": runtime_ms,
            },
        )
        
        return ImageAnalysisResponse(
            id=analysis_id,
            verdict=result.verdict,
            confidence=result.confidence,
            heatmap_url=heatmap_url,
            sha256=file_hash,
            model_version=registry.model_version,
            runtime_ms=runtime_ms,
            device=registry.device,
            created_at=datetime.now(timezone.utc),
        )
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        
        # Save failed analysis record
        analysis = Analysis(
            id=analysis_id,
            type=AnalysisType.IMAGE,
            status=AnalysisStatus.FAILED,
            sha256=file_hash,
            model_version=registry.model_version if registry else "unknown",
            device="unknown",
            error=str(e),
        )
        db.add(analysis)
        await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/analyze/video",
    response_model=VideoAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze video for deepfake",
    description="Upload a video for asynchronous deepfake detection analysis.",
)
async def analyze_video_endpoint(
    file: UploadFile = File(..., description="Video file to analyze"),
    db: DBSession = None,
    storage: Storage = None,
    settings: AppSettings = None,
    registry: Registry = None,
) -> VideoAnalysisResponse:
    """
    Submit a video for asynchronous deepfake analysis.
    
    Returns a job ID that can be used to poll for status and results.
    """
    # Validate content type
    if file.content_type not in settings.allowed_video_types_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Allowed: {settings.allowed_video_types_list}",
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )
    
    # Compute hash
    file_hash = compute_sha256(content)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        upload_path = await storage.save_upload(
            content=content,
            filename=f"{job_id}_{file.filename}",
            subfolder="videos",
        )
        
        # Create pending analysis record
        analysis = Analysis(
            id=job_id,
            type=AnalysisType.VIDEO,
            status=AnalysisStatus.PENDING,
            sha256=file_hash,
            model_version=registry.model_version,
            device=registry.device,
            upload_path=upload_path,
        )
        
        db.add(analysis)
        await db.commit()
        
        # Get active model filename for worker consistency
        model_filename = registry.active_model_filename if hasattr(registry, "active_model_filename") else None

        # Submit Celery task
        process_video_task.delay(
            job_id=job_id,
            video_path=upload_path,
            file_hash=file_hash,
            model_filename=model_filename,
        )
        
        logger.info(
            "Video analysis job submitted",
            extra={
                "job_id": job_id,
                "file_size": len(content),
            },
        )
        
        return VideoAnalysisResponse(
            job_id=job_id,
            status=AnalysisStatus.PENDING,
            message="Video submitted for processing. Use job ID to check status.",
        )
        
    except Exception as e:
        logger.error(f"Video submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit video: {str(e)}",
        )
