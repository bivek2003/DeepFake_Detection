"""
CRUD operations for database models.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AnalysisStatus, Verdict
from app.persistence.models import Analysis, Asset, Frame


async def get_analysis(db: AsyncSession, analysis_id: str) -> Analysis | None:
    """Get analysis by ID."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    return result.scalar_one_or_none()


async def get_analysis_by_hash(
    db: AsyncSession,
    sha256: str,
    model_version: str,
) -> Analysis | None:
    """Get analysis by file hash and model version (for caching)."""
    result = await db.execute(
        select(Analysis)
        .where(Analysis.sha256 == sha256)
        .where(Analysis.model_version == model_version)
        .where(Analysis.status == AnalysisStatus.COMPLETED)
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def update_analysis_status(
    db: AsyncSession,
    analysis_id: str,
    status: AnalysisStatus,
    error: str | None = None,
) -> None:
    """Update analysis status."""
    analysis = await get_analysis(db, analysis_id)
    if analysis:
        analysis.status = status
        if error:
            analysis.error = error
        await db.commit()


async def update_analysis_result(
    db: AsyncSession,
    analysis_id: str,
    verdict: Verdict,
    confidence: float,
    runtime_ms: int,
    total_frames: int | None = None,
) -> None:
    """Update analysis with results."""
    analysis = await get_analysis(db, analysis_id)
    if analysis:
        analysis.status = AnalysisStatus.COMPLETED
        analysis.verdict = verdict
        analysis.confidence = confidence
        analysis.runtime_ms = runtime_ms
        if total_frames is not None:
            analysis.total_frames = total_frames
        await db.commit()


async def create_frame(
    db: AsyncSession,
    analysis_id: str,
    frame_index: int,
    timestamp: float,
    score: float,
    overlay_path: str | None = None,
) -> Frame:
    """Create frame analysis record."""
    frame = Frame(
        analysis_id=analysis_id,
        frame_index=frame_index,
        timestamp=timestamp,
        score=score,
        overlay_path=overlay_path,
    )
    db.add(frame)
    await db.commit()
    return frame


async def get_frames_for_analysis(
    db: AsyncSession,
    analysis_id: str,
) -> list[Frame]:
    """Get all frames for an analysis."""
    result = await db.execute(
        select(Frame).where(Frame.analysis_id == analysis_id).order_by(Frame.frame_index)
    )
    return list(result.scalars().all())


async def create_asset(
    db: AsyncSession,
    analysis_id: str,
    kind: str,
    path: str,
) -> Asset:
    """Create asset record."""
    asset = Asset(
        analysis_id=analysis_id,
        kind=kind,
        path=path,
    )
    db.add(asset)
    await db.commit()
    return asset


async def get_assets_for_analysis(
    db: AsyncSession,
    analysis_id: str,
) -> list[Asset]:
    """Get all assets for an analysis."""
    result = await db.execute(select(Asset).where(Asset.analysis_id == analysis_id))
    return list(result.scalars().all())


async def get_recent_analyses(
    db: AsyncSession,
    limit: int = 20,
) -> list[Analysis]:
    """Get recent analyses for dashboard."""
    result = await db.execute(select(Analysis).order_by(Analysis.created_at.desc()).limit(limit))
    return list(result.scalars().all())
