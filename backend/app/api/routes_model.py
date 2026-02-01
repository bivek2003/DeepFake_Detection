"""
Model information endpoints.
"""

from fastapi import APIRouter

from app.api.deps import AppSettings, Registry
from app.api.schemas import ModelInfo
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Get model information",
    description="Get information about the currently loaded detection model.",
)
async def get_model_info(
    registry: Registry,
    settings: AppSettings,
) -> ModelInfo:
    """
    Get detailed information about the detection model.
    
    Includes version, calibration method, and performance metrics if available.
    """
    return ModelInfo(
        model_name=registry.model_name,
        model_version=registry.model_version,
        commit_hash=registry.commit_hash,
        calibration_method=registry.calibration_method,
        demo_mode=settings.demo_mode,
        device=registry.device,
        metrics=registry.metrics,
    )
