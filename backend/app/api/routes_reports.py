"""
Report generation and download endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.api.deps import DBSession, Storage
from app.api.schemas import AnalysisStatus
from app.logging_config import get_logger
from app.persistence import crud

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/reports/{job_id}.pdf",
    summary="Download PDF report",
    description="Download the PDF forensic report for a completed video analysis.",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF report file",
        },
    },
)
async def download_report(
    job_id: str,
    db: DBSession,
    storage: Storage,
):
    """
    Download the PDF forensic report for a video analysis.

    Only available after job is completed.
    """
    analysis = await crud.get_analysis(db, job_id)

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report not available. Job status: {analysis.status.value}",
        )

    # Get report asset
    assets = await crud.get_assets_for_analysis(db, job_id)
    report_asset = next((a for a in assets if a.kind == "report"), None)

    if not report_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found for this job",
        )

    # Get file path
    file_path = storage.get_asset_path(report_asset.path)

    if not file_path.exists():
        logger.error(f"Report file not found: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found",
        )

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=f"deepfake_analysis_{job_id}.pdf",
        headers={
            "Content-Disposition": f'attachment; filename="deepfake_analysis_{job_id}.pdf"',
        },
    )


@router.get(
    "/assets/{path:path}",
    summary="Get asset",
    description="Retrieve generated assets like heatmaps and charts.",
)
async def get_asset(
    path: str,
    storage: Storage,
):
    """
    Retrieve a generated asset file.

    Assets include heatmap overlays, timeline charts, and frame images.
    """
    file_path = storage.get_asset_path(path)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset not found: {path}",
        )

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
    }
    content_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
    )
