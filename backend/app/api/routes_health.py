"""
Health check endpoints.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.api.deps import DBSession, Registry
from app.api.schemas import HealthCheck, ReadinessCheck
from app.logging_config import get_logger
from app.services.caching import redis_client

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/healthz",
    response_model=HealthCheck,
    summary="Health check",
    description="Basic health check endpoint for liveness probes.",
)
async def health_check() -> HealthCheck:
    """Basic health check - always returns healthy if service is running."""
    return HealthCheck(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/readyz",
    response_model=ReadinessCheck,
    summary="Readiness check",
    description="Readiness check endpoint verifying all dependencies.",
)
async def readiness_check(
    db: DBSession,
    registry: Registry,
) -> ReadinessCheck:
    """
    Check if service is ready to accept requests.
    Verifies database, Redis, and model status.
    """
    # Check database
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # Check Redis
    redis_status = "healthy"
    try:
        client = await redis_client()
        if client:
            await client.ping()
        else:
            redis_status = "not configured"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"
    
    # Check model
    model_status = "healthy" if registry.is_initialized else "not initialized"
    
    # Determine overall status
    all_healthy = (
        db_status == "healthy"
        and redis_status in ("healthy", "not configured")
        and model_status == "healthy"
    )
    
    return ReadinessCheck(
        status="ready" if all_healthy else "not ready",
        database=db_status,
        redis=redis_status,
        model=model_status,
    )
