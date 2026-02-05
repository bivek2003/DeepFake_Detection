"""
Video inference module.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.api.schemas import Verdict
from app.logging_config import get_logger
from app.ml.explainability import generate_demo_heatmap
from app.ml.faces import extract_largest_face
from app.ml.model_registry import ModelRegistry
from app.ml.preprocess import numpy_to_bytes, preprocess_image_array
from app.settings import Settings

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Result for a single frame."""

    frame_index: int
    timestamp: float
    score: float
    frame_image: np.ndarray | None = None


@dataclass
class VideoAnalysisResult:
    """Result of video analysis."""

    verdict: Verdict
    confidence: float
    total_frames: int
    analyzed_frames: int
    frame_results: list[FrameResult]
    suspicious_frames: list[FrameResult]
    runtime_ms: int


def sample_frames(
    video_path: str | Path,
    max_frames: int = 100,
    sample_rate: int = 1,
) -> tuple[list[np.ndarray], list[float], int]:
    """
    Sample frames from video.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to sample
        sample_rate: Sample every Nth frame

    Returns:
        Tuple of (frames, timestamps, total_frame_count)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to sample
    step = max(1, total_frames // max_frames)
    step = max(step, sample_rate)

    frames = []
    timestamps = []
    frame_idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)
        timestamps.append(frame_idx / fps if fps > 0 else frame_idx)
        frame_idx += step

    cap.release()

    return frames, timestamps, total_frames


async def analyze_video(
    video_path: str | Path,
    registry: ModelRegistry,
    settings: Settings,
) -> VideoAnalysisResult:
    """
    Analyze video for deepfake detection.

    Args:
        video_path: Path to video file
        registry: Model registry
        settings: Application settings

    Returns:
        Video analysis result
    """
    start_time = time.time()

    # Sample frames
    frames, timestamps, total_frames = sample_frames(
        video_path,
        max_frames=settings.max_frames,
        sample_rate=settings.frame_sample_rate,
    )

    logger.info(f"Sampled {len(frames)} frames from video")

    frame_results = []

    # Process each frame
    for idx, (frame, timestamp) in enumerate(zip(frames, timestamps, strict=True)):
        # Extract face
        face = extract_largest_face(frame)

        if face is not None:
            input_tensor = preprocess_image_array(face)
        else:
            # Resize full frame
            resized = cv2.resize(frame, (224, 224))
            input_tensor = preprocess_image_array(resized)

        # Run inference
        real_prob, fake_prob = registry.predict(input_tensor)

        frame_results.append(
            FrameResult(
                frame_index=idx,
                timestamp=timestamp,
                score=fake_prob,
                frame_image=frame,
            )
        )

    # Aggregate results
    scores = [r.score for r in frame_results]
    avg_score = np.mean(scores) if scores else 0.5

    # Determine verdict based on average score
    if avg_score > 0.7:
        verdict = Verdict.FAKE
    elif avg_score < 0.3:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.UNCERTAIN

    confidence = max(avg_score, 1 - avg_score)

    # Get suspicious frames (top 5 by score)
    suspicious = sorted(frame_results, key=lambda x: x.score, reverse=True)[:5]

    runtime_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "Video analysis completed",
        extra={
            "verdict": verdict.value,
            "confidence": confidence,
            "total_frames": total_frames,
            "analyzed_frames": len(frame_results),
            "runtime_ms": runtime_ms,
        },
    )

    return VideoAnalysisResult(
        verdict=verdict,
        confidence=confidence,
        total_frames=total_frames,
        analyzed_frames=len(frame_results),
        frame_results=frame_results,
        suspicious_frames=suspicious,
        runtime_ms=runtime_ms,
    )


def generate_frame_heatmap(frame: np.ndarray) -> bytes:
    """Generate heatmap for a single frame."""
    overlay = generate_demo_heatmap(frame)
    return numpy_to_bytes(overlay, "png")
