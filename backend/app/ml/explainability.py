"""
Explainability module for generating heatmaps.
Provides Grad-CAM style visualizations and demo mode fallbacks.
"""

import cv2
import numpy as np
import torch

from app.logging_config import get_logger
from app.ml.preprocess import bytes_to_numpy, numpy_to_bytes

logger = get_logger(__name__)


def generate_demo_heatmap(image: np.ndarray) -> np.ndarray:
    """
    Generate a demo heatmap using edge detection and blur difference.
    Creates visually plausible heatmaps without real model gradients.

    Args:
        image: Input image (BGR format)

    Returns:
        Heatmap overlay image (BGR format)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges (potential manipulation artifacts)
    edges = cv2.Canny(gray, 50, 150)

    # Apply Gaussian blur to original
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute difference (high-frequency content)
    diff = cv2.absdiff(gray, blurred)

    # Combine edge and difference signals
    combined = cv2.addWeighted(edges, 0.5, diff, 0.5, 0)

    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.GaussianBlur(combined, (11, 11), 0)

    # Normalize to 0-255
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)

    # Apply colormap (red for high attention)
    heatmap = cv2.applyColorMap(combined.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """Resolve target layer for Grad-CAM from DeepfakeDetector (EfficientNet) or similar."""
    # DeepfakeDetector has .backbone (timm EfficientNet)
    if hasattr(model, "backbone"):
        backbone = model.backbone
        # timm EfficientNet: last spatial layer is often conv_head or blocks[-1]
        if hasattr(backbone, "conv_head"):
            return backbone.conv_head
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            return backbone.blocks[-1]
        # Fallback: last module that produces 4D output
        for name, module in reversed(list(backbone.named_children())):
            if "conv" in name or "block" in name:
                return module
    if hasattr(model, "features"):
        return model.features
    return None


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str | None = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap from model gradients.
    Supports DeepfakeDetector (EfficientNet backbone) and models with .features.

    Args:
        model: PyTorch model
        input_tensor: Input tensor [1, C, H, W]
        target_layer: Name of target layer for Grad-CAM (optional, auto-detected)

    Returns:
        Heatmap as numpy array [H, W]
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    h, w = input_tensor.shape[2], input_tensor.shape[3]

    target = None
    if target_layer and hasattr(model, "named_modules"):
        named = dict(model.named_modules())
        target = named.get(target_layer)
    if target is None:
        target = _get_target_layer(model)

    if target is None:
        logger.debug("No target layer for Grad-CAM, using demo heatmap")
        return np.ones((h, w), dtype=np.float32) * 0.5

    activations = None
    gradients = None

    def forward_hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        nonlocal activations
        activations = output.detach()

    def backward_hook(module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        nonlocal gradients
        if grad_output[0] is not None:
            gradients = grad_output[0].detach()

    fh = target.register_forward_hook(forward_hook)
    bh = target.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        if input_tensor.requires_grad is False:
            input_tensor = input_tensor.requires_grad_(True)
        output = model(input_tensor)
        # Binary classifier: output [B, 1]; high value = fake
        if output.shape[-1] == 1:
            scalar = output[0, 0]
        else:
            scalar = output[0, 1] if output.shape[-1] > 1 else output[0, 0]
        model.zero_grad()
        scalar.backward()

        if gradients is None or activations is None:
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Handle 4D (B,C,H,W) or 2D (B,C) activations
        if activations.dim() == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
        else:
            return np.ones((h, w), dtype=np.float32) * 0.5

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )
        return cam.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}, using demo heatmap")
        return np.ones((h, w), dtype=np.float32) * 0.5
    finally:
        fh.remove()
        bh.remove()


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create heatmap overlay on image.

    Args:
        image: Original image (BGR format)
        heatmap: Heatmap array [H, W] with values 0-1
        alpha: Overlay alpha

    Returns:
        Overlaid image (BGR format)
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert to uint8
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def generate_heatmap_bytes(
    image_bytes: bytes,
    model: torch.nn.Module | None = None,
    input_tensor: torch.Tensor | None = None,
    is_demo: bool = True,
) -> bytes:
    """
    Generate heatmap overlay and return as PNG bytes.
    Falls back to demo heatmap if Grad-CAM fails.

    Args:
        image_bytes: Original image bytes
        model: PyTorch model (optional)
        input_tensor: Preprocessed input tensor (optional)
        is_demo: Whether to use demo heatmap generation

    Returns:
        Heatmap overlay as PNG bytes
    """
    image = bytes_to_numpy(image_bytes)

    if is_demo or model is None or input_tensor is None:
        overlay = generate_demo_heatmap(image)
    else:
        try:
            heatmap = generate_gradcam_heatmap(model, input_tensor)
            overlay = create_heatmap_overlay(image, heatmap)
        except Exception as e:
            logger.debug(f"Grad-CAM failed, using demo heatmap: {e}")
            overlay = generate_demo_heatmap(image)

    return numpy_to_bytes(overlay, "png")
