"""
Face detection and extraction utilities.
"""

import cv2
import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(
    image: np.ndarray,
    min_size: tuple[int, int] = (80, 80),
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in image using Haar cascade.
    
    Args:
        image: Input image (BGR format)
        min_size: Minimum face size
        scale_factor: Scale factor for detection
        min_neighbors: Minimum neighbors for detection
        
    Returns:
        List of (x, y, w, h) bounding boxes
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    
    return [tuple(f) for f in faces]


def extract_face(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 0.2,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Extract and resize face from image.
    
    Args:
        image: Input image
        bbox: Face bounding box (x, y, w, h)
        padding: Padding factor around face
        output_size: Output image size
        
    Returns:
        Extracted face image
    """
    x, y, w, h = bbox
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    # Extract face region
    face = image[y1:y2, x1:x2]
    
    # Resize
    face = cv2.resize(face, output_size)
    
    return face


def extract_largest_face(
    image: np.ndarray,
    padding: float = 0.2,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray | None:
    """
    Detect and extract the largest face from image.
    
    Args:
        image: Input image
        padding: Padding factor around face
        output_size: Output image size
        
    Returns:
        Extracted face image or None if no face detected
    """
    faces = detect_faces(image)
    
    if not faces:
        logger.debug("No faces detected in image")
        return None
    
    # Find largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    
    return extract_face(image, largest, padding, output_size)


def extract_all_faces(
    image: np.ndarray,
    padding: float = 0.2,
    output_size: tuple[int, int] = (224, 224),
) -> list[np.ndarray]:
    """
    Detect and extract all faces from image.
    
    Args:
        image: Input image
        padding: Padding factor around faces
        output_size: Output image size
        
    Returns:
        List of extracted face images
    """
    faces = detect_faces(image)
    
    return [
        extract_face(image, bbox, padding, output_size)
        for bbox in faces
    ]
