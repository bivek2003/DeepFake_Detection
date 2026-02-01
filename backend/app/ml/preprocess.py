"""
Image preprocessing for model inference.
"""

import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.logging_config import get_logger

logger = get_logger(__name__)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard transforms for inference
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image_bytes(content: bytes) -> torch.Tensor:
    """
    Preprocess image from bytes for model inference.
    
    Args:
        content: Image file bytes
        
    Returns:
        Preprocessed tensor [1, 3, 224, 224]
    """
    # Load image
    image = Image.open(io.BytesIO(content))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transforms
    tensor = inference_transform(image)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def preprocess_image_array(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess numpy array image for model inference.
    
    Args:
        image: Image as numpy array (H, W, C) in BGR or RGB format
        
    Returns:
        Preprocessed tensor [1, 3, 224, 224]
    """
    # Convert BGR to RGB if needed (OpenCV format)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Apply transforms
    tensor = inference_transform(pil_image)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def preprocess_batch(images: list[np.ndarray]) -> torch.Tensor:
    """
    Preprocess batch of images.
    
    Args:
        images: List of images as numpy arrays
        
    Returns:
        Batch tensor [B, 3, 224, 224]
    """
    tensors = []
    
    for image in images:
        tensor = preprocess_image_array(image)
        tensors.append(tensor)
    
    return torch.cat(tensors, dim=0)


def bytes_to_numpy(content: bytes) -> np.ndarray:
    """Convert image bytes to numpy array."""
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def numpy_to_bytes(image: np.ndarray, format: str = "png") -> bytes:
    """Convert numpy array to image bytes."""
    if format.lower() == "png":
        _, buffer = cv2.imencode(".png", image)
    elif format.lower() in ("jpg", "jpeg"):
        _, buffer = cv2.imencode(".jpg", image)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return buffer.tobytes()
