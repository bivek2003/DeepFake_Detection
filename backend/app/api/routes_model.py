"""
Model information endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import AppSettings, Registry
from app.api.schemas import ModelInfo
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ModelItem(BaseModel):
    id: str
    name: str
    description: str
    type: str
    size_mb: float
    active: bool


class SwitchModelRequest(BaseModel):
    filename: str


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


@router.get(
    "/model/list",
    response_model=list[ModelItem],
    summary="List available models",
    description="List all model checkpoints available for loading.",
)
async def list_models(
    registry: Registry,
    settings: AppSettings,
) -> list[ModelItem]:
    """List available models."""
    models = registry.list_models()

    # Mark active model
    # We infer active status by checking if metrics match
    # This is a heuristic until we store active filename explicitly
    current_metrics = registry.metrics or {}
    accuracy = current_metrics.get("accuracy", 0)

    for model in models:
        # Check active model
        if hasattr(registry, "active_model_filename") and registry.active_model_filename:
            if model["id"] == registry.active_model_filename:
                model["active"] = True
        # Fallback to heuristics if active_model_filename not set (e.g. initial load)
        elif not settings.demo_mode:
            if model["id"] == "model_m12_high_end.pt" and accuracy > 0.99:
                model["active"] = True
            elif model["id"] == "model_m8_standard.pt" and accuracy < 0.99 and accuracy > 0.90:
                model["active"] = True

    return models


@router.post(
    "/model/switch",
    summary="Switch active model",
    description="Switch the currently running model to a different checkpoint.",
)
async def switch_model(
    request: SwitchModelRequest,
    registry: Registry,
) -> dict:
    """Switch to a specific model."""
    try:
        await registry.switch_model(request.filename)
        return {"status": "success", "message": f"Switched to {request.filename}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found") from None
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
