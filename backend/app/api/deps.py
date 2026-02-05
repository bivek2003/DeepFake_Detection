"""
FastAPI dependencies for dependency injection.
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from app.ml.model_registry import ModelRegistry
from app.persistence.db import get_async_session
from app.services.storage import StorageService
from app.settings import Settings, get_settings

logger = get_logger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_async_session():
        yield session


async def get_storage_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageService:
    """Get storage service dependency."""
    return StorageService(
        upload_dir=settings.upload_dir,
        assets_dir=settings.assets_dir,
        backend=settings.storage_backend,
    )


async def get_model_registry(request: Request) -> ModelRegistry:
    """Get model registry from application state."""
    if not hasattr(request.app.state, "model_registry"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry not initialized",
        )
    return request.app.state.model_registry


def validate_file_size(
    settings: Annotated[Settings, Depends(get_settings)],
) -> int:
    """Get maximum file size in bytes."""
    return settings.max_file_size_bytes


def validate_image_content_type(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[str]:
    """Get allowed image content types."""
    return settings.allowed_image_types_list


def validate_video_content_type(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[str]:
    """Get allowed video content types."""
    return settings.allowed_video_types_list


# Type aliases for cleaner function signatures
DBSession = Annotated[AsyncSession, Depends(get_db_session)]
Storage = Annotated[StorageService, Depends(get_storage_service)]
Registry = Annotated[ModelRegistry, Depends(get_model_registry)]
AppSettings = Annotated[Settings, Depends(get_settings)]
