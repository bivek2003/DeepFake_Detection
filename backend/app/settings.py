"""
Application settings using Pydantic Settings.
All configuration is loaded from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Mode
    demo_mode: bool = Field(default=True, description="Enable demo mode without real weights")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Authentication
    auth_enabled: bool = Field(default=False, description="Enable JWT authentication")
    jwt_secret_key: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=60, description="JWT expiration in minutes")

    # Database
    database_url: str = Field(
        default="postgresql://deepfake:deepfake_secret@localhost:5432/deepfake_detection",
        description="PostgreSQL connection string",
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", description="Celery result backend URL"
    )

    # File Upload
    max_file_size_mb: int = Field(default=100, description="Maximum upload size in MB")
    allowed_image_types: str = Field(
        default="image/jpeg,image/png,image/webp", description="Allowed image MIME types"
    )
    allowed_video_types: str = Field(
        default="video/mp4,video/avi,video/mov,video/webm", description="Allowed video MIME types"
    )
    upload_dir: str = Field(default="/app/uploads", description="Upload directory")
    assets_dir: str = Field(default="/app/assets", description="Assets directory")

    # ML Settings
    model_weights_path: str = Field(default="/app/weights", description="Model weights directory")
    device: str = Field(default="auto", description="Device: auto, cpu, cuda")
    batch_size: int = Field(default=8, description="Batch size for video processing")
    max_frames: int = Field(default=100, description="Maximum frames to sample from video")
    frame_sample_rate: int = Field(default=1, description="Sample every Nth frame")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="CORS allowed origins",
    )

    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")

    # Storage
    storage_backend: Literal["local", "s3"] = Field(default="local", description="Storage backend")
    s3_bucket: str | None = Field(default=None, description="S3 bucket name")
    s3_region: str | None = Field(default=None, description="S3 region")

    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @computed_field
    @property
    def allowed_image_types_list(self) -> list[str]:
        """List of allowed image MIME types."""
        return [t.strip() for t in self.allowed_image_types.split(",")]

    @computed_field
    @property
    def allowed_video_types_list(self) -> list[str]:
        """List of allowed video MIME types."""
        return [t.strip() for t in self.allowed_video_types.split(",")]

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        """List of CORS allowed origins."""
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
