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


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str | None = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap from model gradients.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor [1, C, H, W]
        target_layer: Name of target layer for Grad-CAM
        
    Returns:
        Heatmap as numpy array [H, W]
    """
    # For demo model, return uniform heatmap
    if not hasattr(model, 'features') or target_layer is None:
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        return np.ones((h, w), dtype=np.float32) * 0.5
    
    # Store activations and gradients
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    # Register hooks
    target = dict(model.named_modules())[target_layer]
    fh = target.register_forward_hook(forward_hook)
    bh = target.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        model.eval()
        input_tensor.requires_grad = True
        output = model(input_tensor)
        
        # Backward pass on fake class
        model.zero_grad()
        output[0, 1].backward()  # Gradient w.r.t. fake class
        
        # Compute Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        
        return cam.squeeze().cpu().numpy()
        
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
    
    Args:
        image_bytes: Original image bytes
        model: PyTorch model (optional)
        input_tensor: Preprocessed input tensor (optional)
        is_demo: Whether to use demo heatmap generation
        
    Returns:
        Heatmap overlay as PNG bytes
    """
    # Load image
    image = bytes_to_numpy(image_bytes)
    
    if is_demo or model is None:
        # Use demo heatmap generation
        overlay = generate_demo_heatmap(image)
    else:
        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(model, input_tensor)
        overlay = create_heatmap_overlay(image, heatmap)
    
    # Convert to bytes
    return numpy_to_bytes(overlay, "png")
