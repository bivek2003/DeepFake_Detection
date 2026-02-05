"""
SQLAlchemy database models.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.api.schemas import AnalysisStatus, AnalysisType, Verdict
from app.persistence.db import Base


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


class Analysis(Base):
    """Analysis record for image or video."""

    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True)
    type = Column(Enum(AnalysisType), nullable=False)
    status = Column(Enum(AnalysisStatus), nullable=False, default=AnalysisStatus.PENDING)
    verdict = Column(Enum(Verdict), nullable=True)
    confidence = Column(Float, nullable=True)
    sha256 = Column(String(64), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    runtime_ms = Column(Integer, nullable=True)
    device = Column(String(20), nullable=False)
    error = Column(Text, nullable=True)
    upload_path = Column(String(500), nullable=True)
    total_frames = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    frames = relationship("Frame", back_populates="analysis", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="analysis", cascade="all, delete-orphan")


class Frame(Base):
    """Frame analysis data for video processing."""

    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # seconds
    score = Column(Float, nullable=False)
    overlay_path = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    analysis = relationship("Analysis", back_populates="frames")


class Asset(Base):
    """Generated assets (overlays, plots, reports)."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(50), nullable=False)  # overlay, plot, report, timeline_chart
    path = Column(String(500), nullable=False)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    analysis = relationship("Analysis", back_populates="assets")
