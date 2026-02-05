"""
Image inference module.
"""

import time
from dataclasses import dataclass

from app.api.schemas import Verdict
from app.logging_config import get_logger
from app.metrics import MODEL_INFERENCE_DURATION
from app.ml.explainability import generate_heatmap_bytes
from app.ml.faces import extract_largest_face
from app.ml.model_registry import ModelRegistry
from app.ml.preprocess import bytes_to_numpy, preprocess_image_array, preprocess_image_bytes
from app.settings import Settings

logger = get_logger(__name__)


@dataclass
class ImageAnalysisResult:
    """Result of image analysis."""

    verdict: Verdict
    confidence: float
    real_prob: float
    fake_prob: float
    heatmap: bytes | None = None


async def analyze_image(
    content: bytes,
    registry: ModelRegistry,
    settings: Settings,
) -> ImageAnalysisResult:
    """
    Analyze image for deepfake detection.

    Args:
        content: Image file bytes
        registry: Model registry
        settings: Application settings

    Returns:
        Analysis result with verdict, confidence, and optional heatmap
    """
    start_time = time.time()

    # Load image
    image = bytes_to_numpy(content)

    # Try to extract face
    face = extract_largest_face(image)

    if face is not None:
        # Use extracted face for analysis
        input_tensor = preprocess_image_array(face)
    else:
        # Use full image if no face detected
        logger.debug("No face detected, using full image")
        input_tensor = preprocess_image_bytes(content)

    # Run inference
    real_prob, fake_prob = registry.predict(input_tensor)

    # Determine verdict
    if fake_prob > 0.7:
        verdict = Verdict.FAKE
    elif fake_prob < 0.3:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.UNCERTAIN

    confidence = max(real_prob, fake_prob)

    # Record inference time
    inference_time = time.time() - start_time
    MODEL_INFERENCE_DURATION.labels(model_name=registry.model_name).observe(inference_time)

    # Generate heatmap
    heatmap = generate_heatmap_bytes(
        image_bytes=content,
        model=registry.model,
        input_tensor=input_tensor,
        is_demo=settings.demo_mode,
    )

    logger.info(
        "Image analysis completed",
        extra={
            "verdict": verdict.value,
            "confidence": confidence,
            "fake_prob": fake_prob,
            "inference_time_ms": int(inference_time * 1000),
        },
    )

    return ImageAnalysisResult(
        verdict=verdict,
        confidence=confidence,
        real_prob=real_prob,
        fake_prob=fake_prob,
        heatmap=heatmap,
    )
