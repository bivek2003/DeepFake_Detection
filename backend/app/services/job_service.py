"""
Job management service.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AnalysisStatus, Verdict
from app.logging_config import get_logger
from app.persistence import crud
from app.persistence.models import Analysis

logger = get_logger(__name__)


class JobService:
    """Service for managing analysis jobs."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_job(self, job_id: str) -> Analysis | None:
        """Get job by ID."""
        return await crud.get_analysis(self.db, job_id)
    
    async def update_job_status(
        self,
        job_id: str,
        status: AnalysisStatus,
        error: str | None = None,
    ) -> None:
        """Update job status."""
        await crud.update_analysis_status(self.db, job_id, status, error)
        logger.info(f"Job {job_id} status updated to {status.value}")
    
    async def complete_job(
        self,
        job_id: str,
        verdict: Verdict,
        confidence: float,
        runtime_ms: int,
        total_frames: int | None = None,
    ) -> None:
        """Mark job as completed with results."""
        await crud.update_analysis_result(
            self.db,
            job_id,
            verdict=verdict,
            confidence=confidence,
            runtime_ms=runtime_ms,
            total_frames=total_frames,
        )
        logger.info(
            f"Job {job_id} completed",
            extra={
                "verdict": verdict.value,
                "confidence": confidence,
                "runtime_ms": runtime_ms,
            },
        )
    
    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        await crud.update_analysis_status(
            self.db,
            job_id,
            status=AnalysisStatus.FAILED,
            error=error,
        )
        logger.error(f"Job {job_id} failed: {error}")
    
    async def add_frame_result(
        self,
        job_id: str,
        frame_index: int,
        timestamp: float,
        score: float,
        overlay_path: str | None = None,
    ) -> None:
        """Add frame analysis result."""
        await crud.create_frame(
            self.db,
            analysis_id=job_id,
            frame_index=frame_index,
            timestamp=timestamp,
            score=score,
            overlay_path=overlay_path,
        )
    
    async def add_asset(
        self,
        job_id: str,
        kind: str,
        path: str,
    ) -> None:
        """Add asset to job."""
        await crud.create_asset(
            self.db,
            analysis_id=job_id,
            kind=kind,
            path=path,
        )
