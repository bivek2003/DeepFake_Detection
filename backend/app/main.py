"""
FastAPI Application Entry Point
Deepfake Detection Platform - Defensive Media Forensics
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.api.deps import get_db_session
from app.api.rate_limit import RateLimitMiddleware
from app.api.routes_analyze import router as analyze_router
from app.api.routes_health import router as health_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_model import router as model_router
from app.api.routes_reports import router as reports_router
from app.logging_config import get_logger, setup_logging
from app.metrics import (
    ERRORS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    init_metrics,
)
from app.ml.model_registry import ModelRegistry
from app.persistence.db import create_tables, get_async_engine
from app.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Deepfake Detection Platform", extra={"version": __version__})
    
    # Initialize metrics
    init_metrics(version=__version__, commit="dev")
    
    # Initialize database tables
    logger.info("Initializing database...")
    engine = get_async_engine()
    await create_tables(engine)
    
    # Initialize ML model registry
    logger.info("Initializing model registry...")
    registry = ModelRegistry()
    await registry.initialize()
    app.state.model_registry = registry
    
    logger.info(
        "Application started",
        extra={
            "demo_mode": settings.demo_mode,
            "auth_enabled": settings.auth_enabled,
        },
    )
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Deepfake Detection API",
        description=(
            "Production-ready API for detecting manipulated media. "
            "This is a defensive forensics platform - NO deepfake generation capabilities."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting middleware
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    
    # Request timing middleware
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)
            
            # Add timing header
            response.headers["X-Process-Time"] = f"{duration:.4f}"
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            ERRORS_TOTAL.labels(
                type=type(e).__name__,
                endpoint=request.url.path,
            ).inc()
            raise
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "type": type(exc).__name__,
            },
        )
    
    # Prometheus metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return JSONResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])
    app.include_router(jobs_router, prefix="/api/v1", tags=["Jobs"])
    app.include_router(reports_router, prefix="/api/v1", tags=["Reports"])
    app.include_router(model_router, prefix="/api/v1", tags=["Model"])
    
    return app


# Create the app instance
app = create_app()
