"""
Tests for heatmap generation.
"""

import io

import numpy as np
from PIL import Image

from app.ml.explainability import (
    create_heatmap_overlay,
    generate_demo_heatmap,
    generate_heatmap_bytes,
)


def test_demo_heatmap_generation():
    """Test demo heatmap generates valid image."""
    # Create test image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Generate heatmap
    overlay = generate_demo_heatmap(image)

    # Verify output
    assert overlay is not None
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_heatmap_overlay_creation():
    """Test heatmap overlay combines correctly."""
    # Create test image and heatmap
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap = np.random.rand(224, 224).astype(np.float32)

    # Create overlay
    overlay = create_heatmap_overlay(image, heatmap)

    # Verify output
    assert overlay is not None
    assert overlay.shape == image.shape


def test_heatmap_bytes_output():
    """Test heatmap generation returns valid PNG bytes."""
    # Create test image as bytes
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    # Generate heatmap bytes
    heatmap_bytes = generate_heatmap_bytes(image_bytes, is_demo=True)

    # Verify output is valid PNG
    assert heatmap_bytes is not None
    assert len(heatmap_bytes) > 0

    # Verify it's a valid image
    result_img = Image.open(io.BytesIO(heatmap_bytes))
    assert result_img.size == (224, 224)


def test_heatmap_different_input_sizes():
    """Test heatmap works with different input sizes."""
    for size in [(100, 100), (640, 480), (1920, 1080)]:
        # Create test image
        img = Image.new("RGB", size, color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        # Generate heatmap
        heatmap_bytes = generate_heatmap_bytes(image_bytes, is_demo=True)

        # Verify output exists
        assert heatmap_bytes is not None
        assert len(heatmap_bytes) > 0
